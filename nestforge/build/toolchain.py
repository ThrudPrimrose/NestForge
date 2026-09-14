# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""What the machine's toolchains can actually do: compiler families, OpenMP runtimes, and
C-signature parsing. No DaCe. Every answer is discovered rather than assumed, and subprocess probes
are cached (``typed=True``)."""

from __future__ import annotations

import ctypes
import ctypes.util  # a SUBMODULE: `import ctypes` alone does not bind it, and lib_findable needs it
import functools
import os
import re
import shutil
import subprocess
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Type, Union

C_SCALAR = {
    "int32_t": ctypes.c_int32,
    "int64_t": ctypes.c_int64,
    "int": ctypes.c_int,
    "float": ctypes.c_float,
    "double": ctypes.c_double,
    "bool": ctypes.c_bool,
}

C_PTR = {"float": ctypes.c_float, "double": ctypes.c_double, "int32_t": ctypes.c_int32, "int64_t": ctypes.c_int64}

DEFAULT_COMPILER = "g++"

#: Records a shared library in DT_NEEDED only when something references it. Every link nest-forge performs
#: passes it before its libraries, instead of trusting a distribution's default.
AS_NEEDED = "-Wl,--as-needed"

#: The ONE C++ standard the whole tree compiles against; override per call only to TEST, never to ship.
CXX_STD = "c++20"

DEFAULT_FLAGS = ["-O3", "-march=native", f"-std={CXX_STD}", "-fPIC", "-shared"]


@functools.lru_cache(maxsize=None, typed=True)
def compiler_family(compiler: str) -> str:
    """OpenMP-relevant compiler family: ``llvm`` (clang/flang, icx/icpx/ifx), or ``gnu``
    (gcc/gfortran, default)."""
    b = Path(compiler).name.lower()
    if "clang" in b or "flang" in b or b.startswith(("icx", "icpx", "ifx")):
        return "llvm"
    return "gnu"


#: OpenMP ABI a family emits -- ``gomp`` (GCC ``GOMP_*``) or ``kmpc`` (LLVM/oneAPI ``__kmpc_*``).
COMPILER_ABI = {"gnu": "gomp", "llvm": "kmpc"}

#: Runtimes selectable via -fopenmp=<name> on clang/flang/icx; gcc links any runtime explicitly via -l<soname>.
LLVM_SELECTABLE = frozenset({"libomp", "libgomp", "libiomp5"})


@dataclass(slots=True)
class OpenMPRuntime:
    """One OpenMP runtime the whole program links. ``libomp`` is default: LLVM-selectable by name and
    GOMP_*-compatible, so gcc- and clang-built libraries share one thread pool."""

    name: str = "libomp"  # selected by name on LLVM (``-fopenmp=<name>``)
    soname: str = "omp"  # ``-l<soname>`` for explicit linking
    #: ``-L`` for the runtime; None -> discovered via linkable_lib_dir. ``""`` forces bare ``-l<soname>``.
    lib_dir: Optional[str] = None
    #: ABIs this runtime implements; libgomp is GOMP_*-only, unusable by a kmpc compiler (clang/nvc++).
    provides: frozenset = frozenset({"kmpc", "gomp"})

    def compatible(self, compiler: str) -> bool:
        """True if ``compiler`` can LINK this runtime: llvm selects by name from the LLVM-selectable
        set; gnu links any gomp-ABI runtime."""
        fam = compiler_family(compiler)
        if fam == "llvm":
            return self.name in LLVM_SELECTABLE and COMPILER_ABI["llvm"] in self.provides
        return COMPILER_ABI["gnu"] in self.provides

    def check(self, compiler: str) -> None:
        if self.compatible(compiler):
            return
        fam = compiler_family(compiler)
        if fam == "llvm":
            if COMPILER_ABI["llvm"] not in self.provides:
                raise ValueError(
                    f"{Path(compiler).name} emits the 'kmpc' OpenMP ABI, which {self.name} does not "
                    f"implement (it provides {sorted(self.provides)}); libgomp is gomp-only. Use a "
                    f"kmpc runtime (libomp/libiomp5)."
                )
            raise ValueError(
                f"{Path(compiler).name} selects the OpenMP runtime by name and only knows "
                f"{sorted(LLVM_SELECTABLE)}; {self.name} is not name-selectable by an LLVM compiler. "
                f"Use libomp/libiomp5, or build with gcc (which links {self.name} via -l{self.soname})."
            )
        raise ValueError(
            f"{Path(compiler).name} emits the 'gomp' OpenMP ABI, which {self.name} does not implement "
            f"(it provides {sorted(self.provides)}). Use a gomp-capable runtime "
            f"(libomp/libiomp5 carry a GOMP-compat layer; libgomp is gomp-only)."
        )

    def compile_flags(self, compiler: str) -> List[str]:
        """Flags to compile a translation unit with OpenMP against this runtime."""
        self.check(compiler)
        fam = compiler_family(compiler)
        if fam == "llvm":  # pick the runtime by name
            return [f"-fopenmp={self.name}"]
        return ["-fopenmp"]  # gnu: runtime fixed at link, not by this flag

    def link_flags(self, compiler: str) -> List[str]:
        """Flags to link a program against THIS runtime only (avoids dual-runtime oversubscription)."""
        self.check(compiler)
        pinned, library = self.link_location(compiler)
        # -L alone leaves no RUNPATH; ctypes.CDLL fails to open the lib after build without -rpath too
        libdir = [f"-L{pinned}", f"-Wl,-rpath,{pinned}"] if pinned else []
        if compiler_family(compiler) == "llvm":
            return [f"-fopenmp={self.name}", *libdir]
        # gnu: link the runtime EXPLICITLY (bare -fopenmp would pull libgomp instead)
        return [*libdir, library]

    def link_location(self, compiler: str) -> Tuple[Optional[str], str]:
        """``(-L directory or None, library flag)``. An explicit ``lib_dir`` wins (pin a spack/module runtime; ``""``
        forces a bare ``-l<soname>``); otherwise both are discovered, see :func:`runtime_library`."""
        if self.lib_dir is not None:
            return self.lib_dir, f"-l{self.soname}"
        return linkable_lib_dir(self.soname, compiler), library_flag(self.soname, compiler)


#: icx auto-links libsvml/libimf/libirng/libintlc off-path with NO RUNPATH; probing this one finds the set.
SUPPORT_LIB_PROBE = "svml"


@functools.lru_cache(maxsize=None, typed=True)
def support_rpath_flags(compiler: str) -> Tuple[str, ...]:
    """-Wl,-rpath for the compiler's own auto-linked support libs (icx svml/imf/irng/intlc), or () if none."""
    found = driver_lib_path(SUPPORT_LIB_PROBE, compiler)
    return ("-Wl,-rpath,%s" % found.parent,) if found else ()


#: Ready-made OpenMP runtimes; libomp/libgomp/libiomp5 share the GOMP ABI.
LIBOMP = OpenMPRuntime(name="libomp", soname="omp")

LIBGOMP = OpenMPRuntime(
    name="libgomp", soname="gomp", provides=frozenset({"gomp"})
)  # GOMP-only; unusable by a kmpc compiler

LIBIOMP5 = OpenMPRuntime(name="libiomp5", soname="iomp5")

#: name -> runtime, for a config/CLI knob.
OPENMP_RUNTIMES = {"libomp": LIBOMP, "libgomp": LIBGOMP, "libiomp5": LIBIOMP5}


def env_library_dirs() -> List[str]:
    """Dirs from LD_LIBRARY_PATH/LIBRARY_PATH; find_library only consults ldconfig, missing these."""
    dirs: List[str] = []
    for var in ("LD_LIBRARY_PATH", "LIBRARY_PATH"):
        dirs += [d for d in os.environ.get(var, "").split(os.pathsep) if d]
    return dirs


#: Drivers to ask where a runtime lives when the target compiler cannot find it; clang-first since libomp.
LIB_PROBE_DRIVERS = ("clang++", "clang", "g++", "gcc")

#: Ceiling on a toolchain PROBE (asking a driver/loader, not compiling); an unbounded one hangs the sweep.
PROBE_TIMEOUT_S: float = 15.0


@functools.lru_cache(maxsize=None, typed=True)
def driver_lib_path(soname: str, compiler: str) -> Optional[Path]:
    """Where ``compiler`` resolves ``lib<soname>.so``, or ``None`` (a different question from what
    ldconfig/find_library find); cached since it sits on the hot flag-composition path."""
    try:
        out = subprocess.run(
            [compiler, f"-print-file-name=lib{soname}.so"], capture_output=True, text=True, timeout=PROBE_TIMEOUT_S
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    if not out or out == f"lib{soname}.so":
        return None
    # normalize LEXICALLY, never resolve(): libomp.so is often a symlink into another dir
    path = Path(os.path.normpath(out))
    return path if path.exists() else None


def driver_search_dirs(compiler: str) -> List[str]:
    """Library directories ``compiler`` itself searches, via -print-search-dirs."""
    try:
        out = subprocess.run(
            [compiler, "-print-search-dirs"], capture_output=True, text=True, timeout=PROBE_TIMEOUT_S
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    for line in out.splitlines():
        if line.startswith("libraries:"):
            raw = line.split(":", 1)[1].strip().lstrip("=")
            return [os.path.normpath(d) for d in raw.split(os.pathsep) if d]
    return []


#: ldconfig by name AND full path: /usr/sbin is off the default non-root PATH on Debian-family slim images.
LDCONFIG_EXES = ("ldconfig", "/usr/sbin/ldconfig", "/sbin/ldconfig")


@functools.lru_cache(maxsize=None, typed=True)
def ldconfig_output() -> str:
    """``ldconfig -p`` output, or "". sbin is off the non-root PATH on slim images, so full paths are tried too."""
    for exe in LDCONFIG_EXES:
        try:
            out = subprocess.run([exe, "-p"], capture_output=True, text=True, timeout=PROBE_TIMEOUT_S).stdout
        except (OSError, subprocess.SubprocessError):
            continue
        if out:
            return out
    return ""


def ldconfig_dirs(soname: str) -> List[str]:
    """Directories the loader cache lists for lib<soname>. The linker needs the -dev .so symlink, which
    ldconfig does not index, but it shares a directory with the versioned .so.N ldconfig does index."""
    out = ldconfig_output()
    if not out:
        return []
    dirs: List[str] = []
    for line in out.splitlines():
        if f"lib{soname}.so" not in line or "=>" not in line:
            continue
        d = os.path.dirname(line.split("=>")[-1].strip())
        if d and d not in dirs:
            dirs.append(d)
    return dirs


#: Common install layouts, tried only after driver/loader queries come up empty; hints, not truth.
LIB_DIR_HINT_ROOTS = ("/usr/lib", "/usr/lib64")

LIB_DIR_HINTS = ("/usr/lib64", "/usr/local/lib64", "/usr/local/lib")


def llvm_version(path: Path) -> Tuple[int, ...]:
    """Version tuple of an llvm-N[.M] dir, or (-1,); string sort ranks llvm-9 above llvm-21, so parsed as ints."""
    parts = path.parent.name.partition("llvm-")[2].split(".")
    if not parts or not parts[0].isdigit():
        return (-1,)
    return tuple(int(p) for p in parts if p.isdigit())


def hint_dirs() -> List[str]:
    """Guessed library dirs, newest LLVM first, ranked ACROSS roots (per-root sorting would put
    /usr/lib/llvm-14 ahead of /usr/lib64/llvm-18) with path as a stable tiebreaker for glob order."""
    found = [p for root in LIB_DIR_HINT_ROOTS for p in Path(root).glob("llvm-*/lib*")]
    ranked = sorted({str(p) for p in found}, key=lambda d: (llvm_version(Path(d)), d), reverse=True)
    return ranked + [d for d in LIB_DIR_HINTS if d not in ranked]


def linker_finds(soname: str, compiler: str = DEFAULT_COMPILER) -> bool:
    return driver_lib_path(soname, compiler) is not None


#: LLVM drivers asked where an LLVM runtime lives: distributions install libomp under the LLVM prefix, off g++'s path.
LLVM_DRIVERS = ("clang++", "clang")


@functools.lru_cache(maxsize=None, typed=True)
def llvm_config_libdir() -> Optional[str]:
    """``llvm-config --libdir`` of the LLVM on PATH, or ``None``."""
    if shutil.which("llvm-config") is None:
        return None
    try:
        out = subprocess.run(["llvm-config", "--libdir"], capture_output=True, text=True, timeout=PROBE_TIMEOUT_S)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def shared_object_in(directory: str, soname: str) -> Optional[Path]:
    """``lib<soname>.so`` in ``directory``, else its highest-versioned ``lib<soname>.so.N``."""
    unversioned = Path(directory) / f"lib{soname}.so"
    if unversioned.exists():
        return unversioned
    versioned = sorted(Path(directory).glob(f"lib{soname}.so.*"))
    return versioned[-1] if versioned else None


def library_search_dirs(soname: str) -> Iterator[str]:
    """Where a runtime may sit once no driver names it: the environment's library paths, each probe driver's
    own search dirs, the loader cache ranked by LLVM version, then common layouts."""
    yield from env_library_dirs()  # explicit intent (spack/module) outranks anything inferred
    for probe in LIB_PROBE_DRIVERS:
        if shutil.which(probe):
            yield from driver_search_dirs(probe)
    # ldconfig lists dirs in cache order, not version order; rank like hint_dirs or llvm-14 wins over 18
    yield from sorted(ldconfig_dirs(soname), key=lambda d: (llvm_version(Path(d)), d), reverse=True)
    yield from hint_dirs()


def runtime_library_candidates(soname: str, compiler: str) -> Iterator[Optional[Path]]:
    yield driver_lib_path(soname, compiler)
    for driver in LLVM_DRIVERS:
        if driver != compiler and shutil.which(driver):
            yield driver_lib_path(soname, driver)
    libdir = llvm_config_libdir()
    if libdir is not None:
        yield shared_object_in(libdir, soname)
    for directory in library_search_dirs(soname):
        yield shared_object_in(directory, soname)


def runtime_library(soname: str, compiler: str) -> Optional[Path]:
    """The shared object ``lib<soname>`` links from for ``compiler``: its own driver first, then an LLVM driver
    on PATH and ``llvm-config --libdir``, then the environment, loader and layout search. A versioned
    ``lib<soname>.so.N`` stands in for a missing ``lib<soname>.so``; ``None`` if nothing provides it."""
    return next((path for path in runtime_library_candidates(soname, compiler) if path is not None), None)


def library_flag(soname: str, compiler: str) -> str:
    """``-l<soname>``, or ``-l:<file>`` when only a versioned shared object provides the runtime."""
    found = runtime_library(soname, compiler)
    return f"-l:{found.name}" if found is not None and found.name != f"lib{soname}.so" else f"-l{soname}"


@functools.lru_cache(maxsize=None, typed=True)
def linkable_lib_dir(soname: str, compiler: str = DEFAULT_COMPILER) -> Optional[str]:
    """The -L directory needed to link lib<soname>, or None if ``compiler`` finds it unaided or nothing provides
    it. Loader and linker search different paths, so the directory comes from :func:`runtime_library`."""
    if shutil.which(compiler) is None:
        return None  # no linker to ask; a guessed -L would be worse than none
    if linker_finds(soname, compiler):
        return None
    found = runtime_library(soname, compiler)
    return str(found.parent) if found is not None else None


def lib_linkable(soname: str, compiler: str = DEFAULT_COMPILER) -> bool:
    """True if -l<soname> resolves at link time; unlike find_library, which a versioned .so.5 satisfies too."""
    return linker_finds(soname, compiler) or linkable_lib_dir(soname, compiler) is not None


def lib_findable(soname: str, lib_dir: Optional[str]) -> bool:
    """True if lib<soname> is in lib_dir, an env loader path, or the system loader path (matches .so.N too)."""
    for d in ([lib_dir] if lib_dir else []) + env_library_dirs():
        p = Path(d)
        if (p / f"lib{soname}.a").exists() or any(p.glob(f"lib{soname}.so*")):
            return True
    return ctypes.util.find_library(soname) is not None


def runtime_installed(rt: OpenMPRuntime) -> bool:
    """True if the runtime's shared object can be found."""
    return lib_findable(rt.soname, rt.lib_dir)


@functools.lru_cache(maxsize=None, typed=True)
def usable_openmp(compiler: str) -> Optional[OpenMPRuntime]:
    """The ONE OpenMP runtime ``compiler`` can actually link, preferring libomp. Never a bare -fopenmp
    (gcc/clang would each link a different default, doubling thread pools in a mixed-compiler sweep): a
    runtime-less build makes the compiler silently drop the OpenMP pragma and run serial. None if nothing links."""
    for rt in OPENMP_RUNTIMES.values():  # deliberately libomp-first
        if not rt.compatible(compiler):
            continue
        if lib_linkable(rt.soname, compiler):
            return rt
    return None


#: The two ctypes shapes a kernel parameter can take: a scalar type, or a pointer to one.
CType = Union[Type[ctypes._SimpleCData], Type[ctypes._Pointer]]


@dataclass(slots=True)
class Param:
    name: str
    ctype: CType
    is_pointer: bool


def parse_params(param_str: str) -> List[Param]:
    """Parse a C parameter list into typed params; skips the leading N_state_t *__state handle."""
    params: List[Param] = []
    for raw in split_params(param_str):
        # strip qualifiers as whole WORDS: a substring strip would corrupt names like `const_term`
        tok = re.sub(r"\b(?:const|__restrict__)\b", "", raw).strip()
        if not tok or tok.endswith("_state_t *__state") or tok.endswith("_state_t* __state"):
            continue
        is_ptr = "*" in tok
        name = re.split(r"[\s*]+", tok)[-1]
        base = tok[: tok.rfind(name)].replace("*", "").strip()
        if is_ptr:
            # an unmapped base type would guess a width silently -- an ABI bug ctypes can't catch -- so refuse
            ptr_ctype = C_PTR.get(base)
            if ptr_ctype is None:
                raise ValueError(
                    f"parameter {name!r} of entry point is a pointer to C type {base!r}, which has no "
                    f"ctypes mapping (known: {sorted(C_PTR)}); add it to C_PTR"
                )
            params.append(Param(name, ctypes.POINTER(ptr_ctype), True))
        else:
            ctype = C_SCALAR.get(base)
            if ctype is None:
                raise ValueError(
                    f"parameter {name!r} of entry point has C type {base!r}, which has no ctypes "
                    f"mapping (known: {sorted(C_SCALAR)}); add it to C_SCALAR"
                )
            params.append(Param(name, ctype, False))
    return params


def split_params(param_str: str) -> List[str]:
    out, depth, cur = [], 0, ""
    for ch in param_str:
        if ch in "(<":
            depth += 1
        elif ch in ")>":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur)
    return out


def raw_signature(text: str, symbol: str, lang: str = "c") -> str:
    """The parameter-declaration text between the parens of the kernel entry, verbatim. One regex shared
    by every consumer that parses a kernel entry definition, anchored on the ``void`` return and opening
    brace since a looser ``\\b<symbol>\\s*\\(`` also matches a doc comment merely naming the function;
    raises LookupError if the entry is absent."""
    if lang == "fortran":
        m = re.search(rf"subroutine\s+{re.escape(symbol)}\s*\((.*?)\)", text, re.S | re.I)
    else:
        # `[^)]*` not `(.*?)`: non-greedy backtracks across a preceding prototype, capturing a bogus span
        m = re.search(rf"void\s+{re.escape(symbol)}\s*\(([^)]*)\)\s*\{{", text, re.S)
    if not m:
        raise LookupError(f"entry {symbol} not found in the emitted {lang} source")
    return m.group(1)


def signature(code: str, symbol: str) -> str:
    """The parameter list of symbol(...) in code; unlike raw_signature, matches a non-void DaCe declaration."""
    m = re.search(rf"{re.escape(symbol)}\s*\((.*?)\)", code, re.S)
    if not m:
        raise LookupError(f"entry point {symbol} not found in generated code")
    return m.group(1)


# whole-toolchain discovery, PATH only; lives here, not a perf driver, so querying "which compilers does
# this box have" does not drag in a dace import via perf/tsvc_arena
@dataclass(slots=True)
class Toolchain:
    """One discovered toolchain family: C compiler, optional C++ compiler, and where it was found."""

    name: str
    cc: str
    cxx: Optional[str]  # None -> no native column

    @property
    def family(self) -> str:
        """OpenMP-runtime family of the C compiler (icx -> llvm)."""
        return compiler_family(self.cc)

    @property
    def fp_family(self) -> str:
        """Flag-matrix FP family: Intel is its own (defaults to -fp-model=fast) despite being clang-based."""
        return "intel" if self.name == "intel" else self.family


#: family label -> (C compiler exe, C++ compiler exe).
FAMILY_EXES = {
    "gcc": ("gcc", "g++"),
    "clang": ("clang", "clang++"),
    "intel": ("icx", "icpx"),
}
#: user tokens (compiler names/aliases) -> family label.
ALIASES = {
    "gcc": "gcc", "g++": "gcc", "gnu": "gcc",
    "clang": "clang", "clang++": "clang", "llvm": "clang",
    "icx": "intel", "icpx": "intel", "intel": "intel", "oneapi": "intel",
}  # yapf: disable


def discover_toolchains(requested: str = "auto") -> List[Toolchain]:
    """Discover toolchain families on PATH ("auto"/"all" -> gcc/clang/intel); C compiler required, C++
    optional."""
    tokens = list(FAMILY_EXES) if requested.strip() in ("", "auto", "all") else requested.split()
    families: List[str] = []
    for t in tokens:
        fam = ALIASES.get(t.strip())
        if fam is None:
            warnings.warn(f"unknown compiler token {t!r}; known: {sorted(ALIASES)}")
        elif fam not in families:
            families.append(fam)
    out: List[Toolchain] = []
    for fam in families:
        cc_exe, cxx_exe = FAMILY_EXES[fam]
        cc = shutil.which(cc_exe)
        if cc is None:
            warnings.warn(f"{fam}: C compiler {cc_exe!r} not found on PATH; skipping this family")
            continue
        cxx = shutil.which(cxx_exe)
        if cxx is None:
            warnings.warn(f"{fam}: C++ compiler {cxx_exe!r} not found; native-baseline column disabled for {fam}")
        out.append(Toolchain(name=fam, cc=cc, cxx=cxx))
    return out


@dataclass(frozen=True, slots=True)
class CudaToolchain:
    """One nvcc found on PATH, with its CUDA release and the directory its ``libcudart`` lives in."""

    nvcc: str
    release: str
    cudart_dir: str

    @property
    def name(self) -> str:
        return f"nvcc-{self.release}"


def path_executables(exe: str) -> List[str]:
    """Every distinct ``exe`` on PATH, resolved through symlinks, in PATH order."""
    found: Dict[str, None] = {}
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        candidate = Path(directory) / exe
        if candidate.is_file() and os.access(candidate, os.X_OK):
            found.setdefault(str(candidate.resolve()), None)
    return list(found)


@functools.lru_cache(maxsize=None, typed=True)
def nvcc_release(nvcc: str) -> str:
    """``major.minor`` of the CUDA toolkit ``nvcc`` belongs to."""
    out = subprocess.run([nvcc, "--version"], capture_output=True, text=True, timeout=60).stdout
    match = re.search(r"release (\d+\.\d+)", out)
    if match is None:
        raise LookupError(f"{nvcc} --version names no CUDA release: {out[:200]!r}")
    return match.group(1)


@functools.lru_cache(maxsize=None, typed=True)
def cudart_dir(nvcc: str) -> str:
    """The directory ``nvcc`` links ``libcudart`` from, read off its own verbose link of a probe library."""
    with tempfile.TemporaryDirectory(prefix="nf_cudart_") as scratch:
        source = Path(scratch) / "probe.cu"
        source.write_text("int nf_cudart_probe() { return 0; }\n")
        probe = str(Path(scratch) / "probe.so")
        link = [
            nvcc,
            "-v",
            "-shared",
            "-Xcompiler=-fPIC",
            nvcc_linker_flag(AS_NEEDED),
            str(source),
            "-o",
            probe,
            "-lcudart",
        ]
        proc = subprocess.run(link, capture_output=True, text=True, timeout=COMPILE_TIMEOUT_S)
    for directory in re.findall(r"(?<=-L)\S+", proc.stdout + proc.stderr):
        candidate = Path(directory.strip('"'))
        if (candidate / "libcudart.so").exists():
            return str(candidate.resolve())
    raise LookupError(f"{nvcc} names no directory holding libcudart.so in its link line")


def nvcc_linker_flag(flag: str) -> str:
    """A ``-Wl,`` linker flag spelled for nvcc, which rejects ``-Wl,`` and forwards ``-Xlinker=`` instead."""
    return "-Xlinker=" + flag.removeprefix("-Wl,")


def discover_cuda_toolchains() -> List[CudaToolchain]:
    """Every nvcc on PATH, one toolchain per distinct compiler."""
    return [CudaToolchain(nvcc, nvcc_release(nvcc), cudart_dir(nvcc)) for nvcc in path_executables("nvcc")]


def cudart_link_flags(directory: str) -> List[str]:
    """Link ``libcudart`` from ``directory`` and find it there again at load time."""
    return [f"-L{directory}", "-lcudart", f"-Wl,-rpath,{directory}"]


def needed_libraries(shared: Path) -> List[str]:
    """The ``NEEDED`` sonames of a shared object, in ``readelf -d`` order."""
    out = subprocess.run(["readelf", "-d", str(shared)], capture_output=True, text=True, check=True).stdout
    return re.findall(r"\(NEEDED\)\s+Shared library: \[([^\]]+)\]", out)


#: The archiver every build uses; nothing in the tree passes -flto, so no LTO-plugin-aware ar/gcc-ar is needed.
AR = "ar"

#: Wall-clock ceiling for a single compile/link/archive command; a stuck compile freezes the whole sweep rank.
COMPILE_TIMEOUT_S: float = float(os.environ.get("NF_COMPILE_TIMEOUT", "900"))

#: Distinct warning TEXTS reported per tool before the rest are only counted (unbounded dedup grows forever).
WARN_BUDGET: int = 5

#: tool name -> (distinct texts already reported (insertion order, as a dict), total suppressed past the budget).
WARNED: Dict[str, Tuple[Dict[str, None], int]] = {}


def warning_kinds(stderr: str) -> str:
    """The distinct [-Wflag] kinds in stderr, or its first line; keyed on kind, not text, to dedup across cells."""
    kinds = sorted({m.group(1) for m in re.finditer(r"\[-W([a-z0-9-]+)\]", stderr)})
    return ", ".join(kinds) if kinds else stderr.strip().splitlines()[0][:120]


def warn_once(tool: str, stderr: str) -> None:
    """Report a SUCCEEDING command's warnings, bounded -- unbounded this printed a multi-KB block per
    compiled cell (hundreds of MB across a sweep). Past budget, kinds are only counted; warning_summary
    reports the total."""
    kinds = warning_kinds(stderr)
    seen, suppressed = WARNED.setdefault(tool, ({}, 0))
    if kinds in seen:
        WARNED[tool] = (seen, suppressed + 1)
        return
    if len(seen) >= WARN_BUDGET:
        WARNED[tool] = (seen, suppressed + 1)
        return
    seen[kinds] = None
    WARNED[tool] = (seen, suppressed)
    warnings.warn(f"{tool} warnings [{kinds}]:\n{stderr[-2000:]}")


def warning_summary() -> List[str]:
    """One line per tool naming what was reported and how many further warnings were only counted."""
    out = []
    for tool, (seen, suppressed) in sorted(WARNED.items()):
        line = f"{tool}: {len(seen)} warning kind(s) reported ({', '.join(sorted(seen))})"
        if suppressed:
            line += f"; {suppressed} further warning(s) suppressed"
        out.append(line)
    return out


def run(cmd: List[str], timeout: Optional[float] = COMPILE_TIMEOUT_S) -> None:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        # child already SIGKILLed; surface as a normal build failure so the sweep moves on
        raise RuntimeError(
            f"command timed out after {timeout:.0f}s: {' '.join(cmd[:2])} ... "
            f"(pathological compile/link; ceiling is NF_COMPILE_TIMEOUT)"
        )
    if p.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd[:2])} ...\n{p.stderr[-2000:]}")
    if p.stderr.strip():
        warn_once(Path(cmd[0]).name, p.stderr)
