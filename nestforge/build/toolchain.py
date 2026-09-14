# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""What the machine's toolchains can actually do: compiler families, OpenMP runtimes, vector-math
libraries, linkers, ccache, and C-signature parsing. No DaCe. Every answer is discovered rather than
assumed, and subprocess probes are cached (``typed=True``)."""
from __future__ import annotations

import ctypes
import ctypes.util  # a SUBMODULE: `import ctypes` alone does not bind it, and lib_findable needs it
import functools
import glob
import os
import re
import shutil
import subprocess
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_C_SCALAR = {
    "int32_t": ctypes.c_int32,
    "int64_t": ctypes.c_int64,
    "int": ctypes.c_int,
    "float": ctypes.c_float,
    "double": ctypes.c_double,
    "bool": ctypes.c_bool
}

_C_PTR = {"float": ctypes.c_float, "double": ctypes.c_double, "int32_t": ctypes.c_int32, "int64_t": ctypes.c_int64}

DEFAULT_COMPILER = "g++"

#: The ONE C++ standard the whole tree compiles against; override per call only to TEST, never to ship.
CXX_STD = "c++20"

DEFAULT_FLAGS = ["-O3", "-march=native", f"-std={CXX_STD}", "-fPIC", "-shared"]


@functools.lru_cache(maxsize=None, typed=True)
def compiler_family(compiler: str) -> str:
    """OpenMP-relevant compiler family: ``llvm`` (clang/flang, icx/icpx/ifx), ``intel-classic``
    (icc/icpc/ifort), ``nvidia`` (nvc/nvc++/nvfortran), or ``gnu`` (gcc/gfortran, default)."""
    b = Path(compiler).name.lower()
    if "clang" in b or "flang" in b or b.startswith(("icx", "icpx", "ifx")):
        return "llvm"
    if b.startswith(("icc", "icpc", "ifort")):
        return "intel-classic"
    if b.startswith(("nvc", "nvfortran", "pgcc", "pgfortran")):
        return "nvidia"
    return "gnu"


#: OpenMP ABI a family emits -- ``gomp`` (GCC ``GOMP_*``) or ``kmpc`` (LLVM/Intel ``__kmpc_*``, incl. nvc/nvc++).
_COMPILER_ABI = {"gnu": "gomp", "llvm": "kmpc", "intel-classic": "kmpc", "nvidia": "kmpc"}

#: Runtimes selectable via -fopenmp=<name> on clang/flang/icx; gcc links any runtime explicitly via -l<soname>.
_LLVM_SELECTABLE = frozenset({"libomp", "libgomp", "libiomp5"})


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
        """True if ``compiler`` can LINK this runtime: nvidia/intel-classic hard-link their own native
        runtime only; llvm selects by name from the LLVM-selectable set; gnu links any gomp-ABI runtime."""
        fam = compiler_family(compiler)
        if fam == "nvidia":
            return self.name == "libnvomp"
        if fam == "intel-classic":
            return self.name == "libiomp5"
        if fam == "llvm":
            return self.name in _LLVM_SELECTABLE and _COMPILER_ABI["llvm"] in self.provides
        return _COMPILER_ABI["gnu"] in self.provides

    def check(self, compiler: str) -> None:
        if self.compatible(compiler):
            return
        fam = compiler_family(compiler)
        if fam == "nvidia":
            raise ValueError(f"{Path(compiler).name} (NVIDIA HPC) links OpenMP only through '-mp', which uses its "
                             f"native libnvomp; it cannot link {self.name}. Use the libnvomp runtime for nvc/nvc++, "
                             f"or drop the NVIDIA compiler from this runtime's sweep.")
        if fam == "intel-classic":
            raise ValueError(f"{Path(compiler).name} (classic Intel) links OpenMP through '-qopenmp', which uses its "
                             f"native libiomp5; it cannot link {self.name}. Use the libiomp5 runtime for icc/icpc, "
                             f"or drop the classic Intel compiler from this runtime's sweep.")
        if fam == "llvm":
            if _COMPILER_ABI["llvm"] not in self.provides:
                raise ValueError(f"{Path(compiler).name} emits the 'kmpc' OpenMP ABI, which {self.name} does not "
                                 f"implement (it provides {sorted(self.provides)}); libgomp is gomp-only. Use a "
                                 f"kmpc runtime (libomp/libiomp5).")
            raise ValueError(f"{Path(compiler).name} selects the OpenMP runtime by name and only knows "
                             f"{sorted(_LLVM_SELECTABLE)}; {self.name} is not name-selectable by an LLVM compiler. "
                             f"Use libomp/libiomp5, or build with gcc (which links {self.name} via -l{self.soname}).")
        raise ValueError(f"{Path(compiler).name} emits the 'gomp' OpenMP ABI, which {self.name} does not implement "
                         f"(it provides {sorted(self.provides)}). Use a gomp-capable runtime "
                         f"(libomp/libiomp5/libnvomp carry a GOMP-compat layer; libgomp is gomp-only).")

    def compile_flags(self, compiler: str) -> List[str]:
        """Flags to compile a translation unit with OpenMP against this runtime."""
        self.check(compiler)
        fam = compiler_family(compiler)
        if fam == "llvm":  # pick the runtime by name
            return [f"-fopenmp={self.name}"]
        if fam == "intel-classic":
            return ["-qopenmp"]
        if fam == "nvidia":
            return ["-mp"]  # hard-links native libnvomp; no -fopenmp=<lib> switch
        return ["-fopenmp"]  # gnu: runtime fixed at link, not by this flag

    def link_flags(self, compiler: str) -> List[str]:
        """Flags to link a program against THIS runtime only (avoids dual-runtime oversubscription)."""
        self.check(compiler)
        fam = compiler_family(compiler)
        # explicit lib_dir wins (pin a spack/module runtime, "" forces bare -l<soname>); else discover it
        pinned = self.lib_dir if self.lib_dir is not None else linkable_lib_dir(self.soname, compiler)
        # -L alone leaves no RUNPATH; ctypes.CDLL fails to open the lib after build without -rpath too
        libdir = [f"-L{pinned}", f"-Wl,-rpath,{pinned}"] if pinned else []
        if fam == "llvm":
            return [f"-fopenmp={self.name}", *libdir]
        if fam == "intel-classic":
            return ["-qopenmp", *libdir]
        if fam == "nvidia":
            return ["-mp", *libdir]
        # gnu: link the runtime EXPLICITLY (bare -fopenmp would pull libgomp instead)
        return [*libdir, f"-l{self.soname}"]


#: icx auto-links libsvml/libimf/libirng/libintlc off-path with NO RUNPATH; probing this one finds the set.
SUPPORT_LIB_PROBE = "svml"


@functools.lru_cache(maxsize=None, typed=True)
def support_rpath_flags(compiler: str) -> Tuple[str, ...]:
    """-Wl,-rpath for the compiler's own auto-linked support libs (icx svml/imf/irng/intlc), or () if none."""
    found = driver_lib_path(SUPPORT_LIB_PROBE, compiler)
    return ("-Wl,-rpath,%s" % found.parent, ) if found else ()


#: Ready-made OpenMP runtimes; libomp/libgomp/libiomp5 share the GOMP ABI, libnvomp only via nvc -mp.
LIBOMP = OpenMPRuntime(name="libomp", soname="omp")

LIBGOMP = OpenMPRuntime(name="libgomp", soname="gomp",
                        provides=frozenset({"gomp"}))  # GOMP-only; unusable by a kmpc compiler

LIBIOMP5 = OpenMPRuntime(name="libiomp5", soname="iomp5")

LIBNVOMP = OpenMPRuntime(name="libnvomp", soname="nvomp")

#: name -> runtime, for a config/CLI knob.
OPENMP_RUNTIMES = {"libomp": LIBOMP, "libgomp": LIBGOMP, "libiomp5": LIBIOMP5, "libnvomp": LIBNVOMP}


def env_library_dirs() -> List[str]:
    """Dirs from LD_LIBRARY_PATH/LIBRARY_PATH/DYLD_*; find_library only consults ldconfig, missing these."""
    dirs: List[str] = []
    for var in ("LD_LIBRARY_PATH", "LIBRARY_PATH", "DYLD_LIBRARY_PATH", "DYLD_FALLBACK_LIBRARY_PATH"):
        dirs += [d for d in os.environ.get(var, "").split(os.pathsep) if d]
    return dirs


#: Drivers to ask where a runtime lives when the target compiler cannot find it; clang-first since libomp.
_LIB_PROBE_DRIVERS = ("clang++", "clang", "g++", "gcc")

#: Ceiling on a toolchain PROBE (asking a driver/loader, not compiling); an unbounded one hangs the sweep.
PROBE_TIMEOUT_S: float = 15.0


@functools.lru_cache(maxsize=None, typed=True)
def driver_lib_path(soname: str, compiler: str) -> Optional[Path]:
    """Where ``compiler`` resolves ``lib<soname>.so``, or ``None`` (a different question from what
    ldconfig/find_library find); cached since it sits on the hot flag-composition path."""
    try:
        out = subprocess.run([compiler, f"-print-file-name=lib{soname}.so"],
                             capture_output=True,
                             text=True,
                             timeout=PROBE_TIMEOUT_S).stdout.strip()
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
        out = subprocess.run([compiler, "-print-search-dirs"], capture_output=True, text=True,
                             timeout=PROBE_TIMEOUT_S).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    for line in out.splitlines():
        if line.startswith("libraries:"):
            raw = line.split(":", 1)[1].strip().lstrip("=")
            return [os.path.normpath(d) for d in raw.split(os.pathsep) if d]
    return []


#: ldconfig by name AND full path: /usr/sbin is off the default non-root PATH on Debian-family slim images.
_LDCONFIG_EXES = ("ldconfig", "/usr/sbin/ldconfig", "/sbin/ldconfig")


@functools.lru_cache(maxsize=None, typed=True)
def ldconfig_output() -> str:
    """``ldconfig -p`` output, or "". sbin is off the non-root PATH on slim images, so full paths are tried too."""
    for exe in _LDCONFIG_EXES:
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
_LIB_DIR_HINT_ROOTS = ("/usr/lib", "/usr/lib64")

_LIB_DIR_HINTS = ("/usr/lib64", "/usr/local/lib64", "/usr/local/lib")


def llvm_version(path: Path) -> Tuple[int, ...]:
    """Version tuple of an llvm-N[.M] dir, or (-1,); string sort ranks llvm-9 above llvm-21, so parsed as ints."""
    parts = path.parent.name.partition("llvm-")[2].split(".")
    if not parts or not parts[0].isdigit():
        return (-1, )
    return tuple(int(p) for p in parts if p.isdigit())


def hint_dirs() -> List[str]:
    """Guessed library dirs, newest LLVM first, ranked ACROSS roots (per-root sorting would put
    /usr/lib/llvm-14 ahead of /usr/lib64/llvm-18) with path as a stable tiebreaker for glob order."""
    found = [p for root in _LIB_DIR_HINT_ROOTS for p in Path(root).glob("llvm-*/lib*")]
    ranked = sorted({str(p) for p in found}, key=lambda d: (llvm_version(Path(d)), d), reverse=True)
    return ranked + [d for d in _LIB_DIR_HINTS if d not in ranked]


def linker_finds(soname: str, compiler: str = DEFAULT_COMPILER) -> bool:
    return driver_lib_path(soname, compiler) is not None


@functools.lru_cache(maxsize=None, typed=True)
def linkable_lib_dir(soname: str, compiler: str = DEFAULT_COMPILER) -> Optional[str]:
    """The -L directory needed to link lib<soname>, or None if the linker already finds it. Loader and
    linker search different paths (Ubuntu's libomp-dev symlink can miss the link path while ldconfig
    still reports it installed). Tries: env path, sibling drivers, then a layout guess."""
    if shutil.which(compiler) is None:
        return None  # no linker to ask; a guessed -L would be worse than none
    if linker_finds(soname, compiler):
        return None
    for d in env_library_dirs():  # explicit intent (spack/module) outranks anything inferred
        p = Path(d)
        if (p / f"lib{soname}.so").exists() or (p / f"lib{soname}.a").exists() or (p / f"lib{soname}.dylib").exists():
            return d
    for probe in _LIB_PROBE_DRIVERS:
        if probe != compiler and shutil.which(probe):
            found = driver_lib_path(soname, probe)
            if found is not None:
                return str(found.parent)
    # nothing resolved it yet: fall back to driver search dirs, a hardcoded ladder would go stale
    for probe in _LIB_PROBE_DRIVERS:
        if shutil.which(probe):
            for d in driver_search_dirs(probe):
                if (Path(d) / f"lib{soname}.so").exists():
                    return d
    # ldconfig lists dirs in cache order, not version order; rank like hint_dirs or llvm-14 wins over 18
    for d in sorted(ldconfig_dirs(soname), key=lambda d: (llvm_version(Path(d)), d), reverse=True):
        if (Path(d) / f"lib{soname}.so").exists():
            return d
    for d in hint_dirs():  # last resort: common layouts, only for a runtime NO query above admitted to
        if (Path(d) / f"lib{soname}.so").exists():
            return d
    return None


def lib_linkable(soname: str, compiler: str = DEFAULT_COMPILER) -> bool:
    """True if -l<soname> resolves at link time; unlike find_library, which a versioned .so.5 satisfies too."""
    return linker_finds(soname, compiler) or linkable_lib_dir(soname, compiler) is not None


def lib_findable(soname: str, lib_dir: Optional[str]) -> bool:
    """True if lib<soname> is in lib_dir, an env loader path, or the system loader path (matches .so.N too)."""
    for d in ([lib_dir] if lib_dir else []) + env_library_dirs():
        p = Path(d)
        if (p / f"lib{soname}.a").exists() or (p / f"lib{soname}.dylib").exists() or any(p.glob(f"lib{soname}.so*")):
            return True
    return ctypes.util.find_library(soname) is not None


def runtime_installed(rt: OpenMPRuntime) -> bool:
    """True if the runtime's shared object can be found; libnvomp lives off the default path, so without
    a lib_dir it reads as not-installed."""
    return lib_findable(rt.soname, rt.lib_dir)


@functools.lru_cache(maxsize=None, typed=True)
def usable_openmp(compiler: str) -> Optional[OpenMPRuntime]:
    """The ONE OpenMP runtime ``compiler`` can actually link, preferring libomp. Never a bare -fopenmp
    (gcc/clang would each link a different default, doubling thread pools in a mixed-compiler sweep): a
    runtime-less build makes the compiler silently drop the OpenMP pragma and run serial. None if nothing links."""
    for rt in OPENMP_RUNTIMES.values():  # deliberately libomp-first
        if not rt.compatible(compiler):
            continue
        # intel-classic/nvidia hard-link their own runtime through -qopenmp/-mp; there is no -l to resolve.
        if compiler_family(compiler) in ("intel-classic", "nvidia") or lib_linkable(rt.soname, compiler):
            return rt
    return None


#: clang/icx -fveclib token per veclib; sleef reuses libmvec's token (x86 has no -fveclib=SLEEF), svml is __svml_*.
_CLANG_VECLIB = {"sleef": "libmvec", "libmvec": "libmvec", "svml": "SVML"}

#: Intel oneAPI roots holding libsvml (+ libintlc/libimf/libirng), off the default path; globbed for */lib.
_INTEL_ONEAPI_ROOTS = ("/opt/intel/oneapi/compiler", "/opt/intel/oneapi")


@functools.lru_cache(maxsize=None, typed=True)
def veclib_lib_dir(soname: str, compiler: str) -> Optional[str]:
    """Directory holding lib<soname> for -L/-rpath, or None on the default path (driver, oneAPI, SLEEF prefix)."""
    found = driver_lib_path(soname, compiler)
    if found is not None:
        return str(found.parent)
    dirs: List[str] = []
    for root in _INTEL_ONEAPI_ROOTS:
        dirs += sorted((str(p) for p in Path(root).glob("*/lib")), reverse=True)
    prefix = os.environ.get("NF_SLEEF_PREFIX")
    if prefix:
        dirs.append(str(Path(prefix) / "lib"))
    dirs += [str(Path.home() / ".local" / "lib"), "/usr/local/lib"]
    for d in dirs:
        if any(Path(d).glob(f"lib{soname}.so*")):
            return d
    return None


@dataclass(slots=True)
class VectorMathLib:
    """SIMD elementary-math library an autovec loop calls (libmvec/sleef emit _ZGV*, svml emits __svml_*)."""
    name: str
    soname: Optional[str]  # -l<soname> for the vector symbols (None: toolchain/glibc provides)
    lib_dir: Optional[str] = None  # explicit -L override; None resolves via veclib_lib_dir

    def compatible(self, compiler: str) -> bool:
        fam = compiler_family(compiler)
        if fam == "llvm":  # -fveclib=libmvec (also SLEEF's path) or -fveclib=SVML
            return self.name in ("libmvec", "sleef", "svml")
        if fam == "gnu":  # gcc emits _ZGV* under fast-math; libmvec/SLEEF satisfy it
            return self.name in ("libmvec", "sleef")  # NOT svml: gcc never emits __svml_*
        if fam == "intel-classic":
            return self.name == "svml"  # classic icc emits SVML natively
        return False  # nvidia: uses its own -Mvect

    def check(self, compiler: str) -> None:
        if not self.compatible(compiler):
            raise ValueError(f"{Path(compiler).name} ({compiler_family(compiler)}) cannot use the {self.name} "
                             f"vector math library; try a compatible compiler or a different veclib.")

    def compile_flags(self, compiler: str) -> List[str]:
        self.check(compiler)
        if compiler_family(compiler) == "llvm":  # SVML -> __svml_*, else glibc _ZGV*
            return [f"-fveclib={_CLANG_VECLIB[self.name]}"]
        return []  # gnu: -ffast-math autovec already emits _ZGV*; intel-classic: SVML native

    def link_flags(self, compiler: str) -> List[str]:
        self.check(compiler)
        if not self.soname:
            return []
        libdir = self.lib_dir or veclib_lib_dir(self.soname, compiler)
        search = [f"-L{libdir}", f"-Wl,-rpath,{libdir}"] if libdir else []
        if self.name == "svml":
            search.append("-Wl,--disable-new-dtags")  # transitive libintlc needs DT_RPATH, not RUNPATH
        # pin NEEDED regardless of link-line position (else a veclib -l before the object is dropped)
        return [*search, f"-Wl,--push-state,--no-as-needed,-l{self.soname},--pop-state"]


SLEEF = VectorMathLib(name="sleef", soname="sleefgnuabi")  # GNU-ABI lib, exports _ZGV* symbols

LIBMVEC = VectorMathLib(name="libmvec", soname="mvec")

SVML = VectorMathLib(name="svml", soname="svml")  # Intel SVML runtime

#: name -> vector-math library, for a config/CLI knob.
VECTOR_LIBS = {"sleef": SLEEF, "libmvec": LIBMVEC, "svml": SVML}


def vectorlib_installed(vl: VectorMathLib) -> bool:
    """True if the vector library is findable; soname-less entries are always present."""
    if not vl.soname:
        return True
    return lib_findable(vl.soname, vl.lib_dir) or veclib_lib_dir(vl.soname, DEFAULT_COMPILER) is not None


#: glibc vector-ABI prefixes: ``_ZGV<isa><mask><lanes>v_``. x86 ``b/c/d/e`` = SSE/AVX/AVX2/AVX512,
#: aarch64 ``n/s`` = NEON/SVE; ``N`` unmasked, ``M`` masked (what ``omp simd`` emits).
GLIBC_VECTOR_PREFIXES: Tuple[str, ...] = ("_ZGVbN2v_", "_ZGVcN4v_", "_ZGVdN4v_", "_ZGVeN8v_", "_ZGVbM2v_", "_ZGVcM4v_",
                                          "_ZGVdM4v_", "_ZGVeM8v_", "_ZGVnN2v_", "_ZGVsMxv_")

#: Two-arg elementals: glibc mangles the extra operand as an extra v (_ZGVbN2vv_pow), missing a unary prefix.
BINARY_VECTOR_OPS: frozenset = frozenset({"pow", "atan2", "hypot", "fmod"})

#: Elementals the veclib probe exercises; a library is only credited for the ones it actually serves.
VECLIB_PROBE_OPS: Tuple[str, ...] = ("sin", "cos", "pow", "log", "exp", "tan", "atan")


def veclib_symbol_candidates(veclib: str, op: str) -> Tuple[str, ...]:
    """Every packed symbol veclib could emit for op; libmvec/sleef are indistinguishable on purpose (the
    link line decides who serves the call); SVML suffixes the lane count, so the stem matches as a prefix."""
    if veclib == "svml":
        return (f"__svml_{op}", )
    if veclib in ("libmvec", "sleef"):
        prefixes = GLIBC_VECTOR_PREFIXES
        if op in BINARY_VECTOR_OPS:
            prefixes = tuple(p[:-1] + "v_" for p in prefixes)  # trailing `v_` -> `vv_`
        return tuple(prefix + op for prefix in prefixes)
    return ()


def nm_symbols(path: str, dynamic_only: bool) -> str:
    """nm output for path: undefined imports, or exports when dynamic_only; both spellings are tried."""
    flavours = (["-D", "--defined-only"], ["--defined-only"]) if dynamic_only else (["-u"], ["-Du"])
    for extra in flavours:
        try:
            done = subprocess.run(["nm", *extra, path], capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.SubprocessError):
            return ""
        if done.returncode == 0 and done.stdout.strip():
            return done.stdout
    return ""


def nm_symbol_names(path: str, dynamic_only: bool) -> frozenset:
    """Symbol names from nm, version suffix stripped (_ZGVdN4v_sin@@GLIBC_2.22 -> _ZGVdN4v_sin)."""
    names = set()
    for line in nm_symbols(path, dynamic_only).splitlines():
        parts = line.split()
        if len(parts) >= 2:  # a bare ``file.o:`` header has one field
            names.add(parts[-1].split("@", 1)[0])
    return frozenset(names)


def serves_op(names: frozenset, veclib: str, op: str) -> bool:
    """Whether ``names`` holds a packed entry point of veclib for op. Whole-name match, not substring
    (``tan`` matched ``_ZGVdN4v_tanh``); SVML is the exception, so only digits may follow its stem."""
    if veclib == "svml":
        stem = f"__svml_{op}"
        return any(n.startswith(stem) and (len(n) == len(stem) or n[len(stem)].isdigit()) for n in names)
    return bool(names & frozenset(veclib_symbol_candidates(veclib, op)))


def packed_ops_called(veclib: str, obj_path: str, ops: Tuple[str, ...] = VECLIB_PROBE_OPS) -> Tuple[str, ...]:
    """Which of ``ops`` ``obj_path`` actually calls through ``veclib``'s packed entry points."""
    names = nm_symbol_names(obj_path, dynamic_only=False)
    return tuple(op for op in ops if serves_op(names, veclib, op))


def veclib_library_path(vl: VectorMathLib, compiler: str) -> Optional[str]:
    """The library file the link would resolve, so its exports can be inspected."""
    if not vl.soname:
        return None
    found = driver_lib_path(vl.soname, compiler)
    if found is not None:
        return str(found)
    lib_dir = vl.lib_dir or veclib_lib_dir(vl.soname, compiler)
    if lib_dir:
        candidates = sorted(Path(lib_dir).glob(f"lib{vl.soname}.so*"))
        if candidates:
            return str(candidates[0])
    return None


@dataclass(slots=True)
class Param:
    name: str
    ctype: object  # a ctypes type
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
        base = tok[:tok.rfind(name)].replace("*", "").strip()
        if is_ptr:
            params.append(Param(name, ctypes.POINTER(_C_PTR.get(base, ctypes.c_double)), True))
        else:
            # an unmapped type would guess a width silently -- an ABI bug ctypes can't catch -- so refuse
            ctype = _C_SCALAR.get(base)
            if ctype is None:
                raise ValueError(f"parameter {name!r} of entry point has C type {base!r}, which has no ctypes "
                                 f"mapping (known: {sorted(_C_SCALAR)}); add it to _C_SCALAR")
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
    m = re.search(rf"{symbol}\s*\((.*?)\)", code, re.S)
    if not m:
        raise LookupError(f"entry point {symbol} not found in generated code")
    return m.group(1)


def clang_major_via_preprocessor(compiler: str) -> Optional[int]:
    """Underlying clang major via __clang_major__, for icx/icpx/ifx whose --version hides it; None if unknown."""
    try:
        p = subprocess.run([compiler, "-dM", "-E", "-x", "c", "/dev/null"],
                           capture_output=True,
                           text=True,
                           timeout=PROBE_TIMEOUT_S)
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r"#define __clang_major__ (\d+)", p.stdout)
    return int(m.group(1)) if m else None


@functools.lru_cache(maxsize=None, typed=True)
def compiler_version(compiler: str) -> Tuple[int, int]:
    """The compiler's (major, minor) version from --version; (0, 0) if unparseable, never a guess."""
    try:
        p = subprocess.run([compiler, "--version"], capture_output=True, text=True, timeout=PROBE_TIMEOUT_S)
    except (OSError, subprocess.SubprocessError):
        return (0, 0)
    out = f"{p.stdout}\n{p.stderr}"
    fam = compiler_family(compiler)
    if fam in ("llvm", "intel-classic"):
        m = re.search(r"clang version (\d+)\.(\d+)", out)
        if m:
            return (int(m.group(1)), int(m.group(2)))
        # icx/icpx/ifx hide the clang version behind their own banner; ask the preprocessor instead
        cmaj = clang_major_via_preprocessor(compiler)
        return (cmaj, 0) if cmaj is not None else (0, 0)
    if fam == "gnu":
        m = re.search(r"\bg(?:cc|\+\+)?[^\n]*?\b(\d+)\.(\d+)\.\d+\b", out) or re.search(r"\b(\d+)\.(\d+)\.\d+\b", out)
        if m:
            return (int(m.group(1)), int(m.group(2)))
    return (0, 0)


# whole-toolchain discovery (PATH + spack + vendor install roots); lives here, not a perf driver, so
# querying "which compilers does this box have" does not drag in a dace import via perf/tsvc_arena
@dataclass(slots=True)
class Toolchain:
    """One discovered toolchain family: C compiler, optional C++ compiler, and where it was found."""
    name: str
    cc: str
    cxx: Optional[str]  # None -> no native column
    source: str  # "path" | "spack"

    @property
    def family(self) -> str:
        """OpenMP-runtime family of the C compiler (icx -> llvm)."""
        return compiler_family(self.cc)

    @property
    def fp_family(self) -> str:
        """Flag-matrix FP family: Intel is its own (defaults to -fp-model=fast) despite being clang-based."""
        return "intel" if self.name == "intel" else self.family


#: family label -> (C compiler exe, C++ compiler exe).
_FAMILY_EXES = {
    "gcc": ("gcc", "g++"),
    "clang": ("clang", "clang++"),
    "nvhpc": ("nvc", "nvc++"),
    "intel": ("icx", "icpx")
}
#: user tokens (compiler names/aliases) -> family label.
_ALIASES = {
    "gcc": "gcc", "g++": "gcc", "gnu": "gcc",
    "clang": "clang", "clang++": "clang", "llvm": "clang",
    "nvc": "nvhpc", "nvc++": "nvhpc", "nvhpc": "nvhpc", "nvidia": "nvhpc",
    "icx": "intel", "icpx": "intel", "intel": "intel", "oneapi": "intel",
}  # yapf: disable


def spack_bin_dirs() -> List[Path]:
    """bin dirs of spack-installed gcc/llvm/nvhpc, so an installed-but-unloaded compiler is still discoverable."""
    if not shutil.which("spack"):
        return []
    dirs: List[Path] = []
    try:
        out = subprocess.run(["spack", "find", "--paths", "--no-groups"], capture_output=True, text=True,
                             timeout=25).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].split("@")[0] in ("gcc", "llvm", "nvhpc"):
            bindir = Path(parts[-1]) / "bin"
            if bindir.is_dir():
                dirs.append(bindir)
    return dirs


def spack_compiler_bin_dirs() -> List[Path]:
    """bin dirs of every compiler spack has REGISTERED, distinct from installed packages."""
    if not shutil.which("spack"):
        return []
    try:
        listing = subprocess.run(["spack", "compiler", "list"], capture_output=True, text=True, timeout=25).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    specs = []
    for line in listing.splitlines():
        line = line.strip()
        if not line or line.startswith("==>") or line.startswith("--"):
            continue
        specs += [tok for tok in line.split() if "@" in tok]
    dirs: List[Path] = []
    for spec in specs[:12]:
        try:
            info = subprocess.run(["spack", "compiler", "info", spec], capture_output=True, text=True,
                                  timeout=15).stdout
        except (OSError, subprocess.SubprocessError):
            continue
        for line in info.splitlines():
            if "=" not in line:
                continue
            path = line.split("=", 1)[1].strip()
            if path and os.path.isabs(path):
                d = Path(path).parent
                if d.is_dir() and d not in dirs:
                    dirs.append(d)
    return dirs


#: Install roots of vendor toolchains off PATH; NF_EXTRA_COMPILER_DIRS (colon-separated) prepends a site prefix.
_VENDOR_COMPILER_GLOBS = (
    "/opt/intel/oneapi/compiler/*/bin",  # icx/icpx/ifx; NOT 'latest' -- can point at an ifx-only version
    "/opt/nvidia/hpc_sdk/Linux_x86_64/*/compilers/bin",  # nvc / nvc++ / nvfortran
)


def vendor_compiler_bin_dirs() -> List[Path]:
    """bin dirs of vendor toolchains at their default location but not on PATH (Intel oneAPI, NVIDIA
    HPC). setvars.sh is deliberately not sourced: the arena dlopens libraries in-process, so a shell
    LD_LIBRARY_PATH would not reach the loader (rpath is baked in instead). Newest version first."""
    dirs: List[Path] = []
    for d in os.environ.get("NF_EXTRA_COMPILER_DIRS", "").split(os.pathsep):
        p = Path(d)
        if d and p.is_dir():
            dirs.append(p)
    for pattern in _VENDOR_COMPILER_GLOBS:
        for d in sorted((Path(x) for x in glob.glob(pattern)), reverse=True):
            if d.is_dir() and d not in dirs:
                dirs.append(d)
    return dirs


def which_on_path(exe: str, extra_dirs: List[Path]) -> Optional[str]:
    """exe on PATH, else under one of extra_dirs (the spack + vendor install bins)."""
    found = shutil.which(exe)
    if found:
        return found
    for d in extra_dirs:
        cand = d / exe
        if cand.is_file() and os.access(cand, os.X_OK):
            return str(cand)
    return None


def discover_toolchains(requested: str = "auto") -> List[Toolchain]:
    """Discover toolchain families ("auto"/"all" -> gcc/clang/nvhpc); C compiler required, C++ optional."""
    tokens = list(_FAMILY_EXES) if requested.strip() in ("", "auto", "all") else requested.split()
    families: List[str] = []
    for t in tokens:
        fam = _ALIASES.get(t.strip())
        if fam is None:
            warnings.warn(f"unknown compiler token {t!r}; known: {sorted(_ALIASES)}")
        elif fam not in families:
            families.append(fam)
    # installed AND registered: a spack-default host may register a compiler outside `spack find`'s prefix
    extra_dirs = spack_bin_dirs()
    for d in spack_compiler_bin_dirs() + vendor_compiler_bin_dirs():
        if d not in extra_dirs:
            extra_dirs.append(d)
    out: List[Toolchain] = []
    for fam in families:
        cc_exe, cxx_exe = _FAMILY_EXES[fam]
        cc = which_on_path(cc_exe, extra_dirs)
        if cc is None:
            warnings.warn(f"{fam}: C compiler {cc_exe!r} not found (PATH, spack or vendor default); skipping "
                          f"this family")
            continue
        cxx = which_on_path(cxx_exe, extra_dirs)
        if cxx is None:
            warnings.warn(f"{fam}: C++ compiler {cxx_exe!r} not found; native-baseline column disabled for {fam}")
        source = "path" if shutil.which(cc_exe) else "vendor/spack"
        out.append(Toolchain(name=fam, cc=cc, cxx=cxx, source=source))
    return out


@functools.lru_cache(maxsize=None, typed=True)
def ar_for(compiler: str) -> str:
    """The LTO-plugin-aware ar (gcc-ar/llvm-ar) when present, so an -flto object stays linkable; plain ar otherwise."""
    cand = {"gnu": "gcc-ar", "llvm": "llvm-ar"}.get(compiler_family(compiler), "ar")
    return cand if shutil.which(cand) else "ar"


#: Minimum compiler version accepting -fuse-ld=<linker>, per family (icx/icpx report as modern LLVM).
_LINKER_MIN: Dict[str, Dict[str, Tuple[int, int]]] = {
    "mold": {
        "gnu": (12, 1),
        "llvm": (12, 0)
    },
    "lld": {
        "gnu": (9, 0),
        "llvm": (3, 0),
        "intel-classic": (0, 0)
    },
    "gold": {
        "gnu": (0, 0),
        "llvm": (3, 0),
        "intel-classic": (0, 0)
    },
}


def linker_supported(compiler: str, linker: str) -> bool:
    fam = compiler_family(compiler)
    floor = _LINKER_MIN.get(linker, {}).get(fam)
    return floor is not None and compiler_version(compiler) >= floor


def fat_lto_flags(compiler: str) -> List[str]:
    """Flags for a FAT-LTO object (bitcode + real code), or [] if this compiler cannot (warns, skips LTO)."""
    fam = compiler_family(compiler)
    if fam == "gnu":
        return ["-flto", "-ffat-lto-objects"]
    if fam == "llvm" and compiler_version(compiler) >= (18, 0):
        return ["-flto", "-ffat-lto-objects"]
    reason = ("clang < 18 has no -ffat-lto-objects" if fam == "llvm" else
              "classic icc uses -ipo, not fat LTO" if fam == "intel-classic" else "no fat-LTO support")
    warnings.warn(f"{Path(compiler).name}: {reason}; archiving the node library without LTO "
                  f"(the .so still links from real machine code and runs correctly).")
    return []


#: Wall-clock ceiling for a single compile/link/archive command; a stuck compile freezes the whole sweep rank.
COMPILE_TIMEOUT_S: float = float(os.environ.get("NF_COMPILE_TIMEOUT", "900"))

#: Distinct warning TEXTS reported per tool before the rest are only counted (unbounded dedup grows forever).
WARN_BUDGET: int = 5

#: tool name -> (distinct texts already reported, total suppressed after the budget).
_warned: Dict[str, Tuple[set, int]] = {}


def warning_kinds(stderr: str) -> str:
    """The distinct [-Wflag] kinds in stderr, or its first line; keyed on kind, not text, to dedup across cells."""
    kinds = sorted({m.group(1) for m in re.finditer(r"\[-W([a-z0-9-]+)\]", stderr)})
    return ", ".join(kinds) if kinds else stderr.strip().splitlines()[0][:120]


def warn_once(tool: str, stderr: str) -> None:
    """Report a SUCCEEDING command's warnings, bounded -- unbounded this printed a multi-KB block per
    compiled cell (hundreds of MB across a sweep). Past budget, kinds are only counted; warning_summary
    reports the total."""
    kinds = warning_kinds(stderr)
    seen, suppressed = _warned.setdefault(tool, (set(), 0))
    if kinds in seen:
        _warned[tool] = (seen, suppressed + 1)
        return
    if len(seen) >= WARN_BUDGET:
        _warned[tool] = (seen, suppressed + 1)
        return
    seen.add(kinds)
    _warned[tool] = (seen, suppressed)
    warnings.warn(f"{tool} warnings [{kinds}]:\n{stderr[-2000:]}")


def warning_summary() -> List[str]:
    """One line per tool naming what was reported and how many further warnings were only counted."""
    out = []
    for tool, (seen, suppressed) in sorted(_warned.items()):
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
        raise RuntimeError(f"command timed out after {timeout:.0f}s: {' '.join(cmd[:2])} ... "
                           f"(pathological compile/link; ceiling is NF_COMPILE_TIMEOUT)")
    if p.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd[:2])} ...\n{p.stderr[-2000:]}")
    if p.stderr.strip():
        warn_once(Path(cmd[0]).name, p.stderr)


#: Fast alternative linkers, FASTEST FIRST. Default ``bfd`` ``ld`` is always the fallback (not listed).
_FAST_LINKERS = ("mold", "lld", "gold")


@functools.lru_cache(maxsize=None, typed=True)
def ccache_available() -> bool:
    """Whether a compiler cache is installed and not disabled. ccache keys on PREPROCESSED SOURCE, not
    path, so the same generated kernel recompiled from a fresh temp dir is a cache hit. NF_NO_CCACHE=1
    forces it off."""
    if os.environ.get("NF_NO_CCACHE"):
        return False
    return shutil.which("ccache") is not None


def ccache_prefix(use_ccache: Optional[bool]) -> List[str]:
    """Launcher prefix for a compiler invocation: ["ccache"] or []. None means AUTO. Pass False from any
    path that MEASURES compile time: a cache hit returns in ~0s, making the toolchain-cost measurement
    meaningless rather than merely faster."""
    if use_ccache is False:
        return []
    return ["ccache"] if ccache_available() else []


@functools.lru_cache(maxsize=None, typed=True)
def available_linkers() -> Dict[str, str]:
    """Fast alternative linkers installed, fastest first: name -> backing binary path."""
    found: Dict[str, str] = {}
    for ld in _FAST_LINKERS:
        p = shutil.which(ld) or shutil.which(f"ld.{ld}")
        if p:
            found[ld] = p
    return found


def fastest_linker(compiler: str) -> List[str]:
    """-fuse-ld=<linker> for the fastest installed linker this compiler accepts (mold > lld > gold), or
    []. NVIDIA has no -fuse-ld switch. NOT cached: the result depends on compiler_version, which tests
    monkeypatch."""
    if compiler_family(compiler) == "nvidia":
        return []
    for ld in available_linkers():  # dict preserves the fastest-first order of _FAST_LINKERS
        if linker_supported(compiler, ld):
            return [f"-fuse-ld={ld}"]
    return []
