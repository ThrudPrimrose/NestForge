# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Owns the DaCe build (BUILD.md): codegen + compile/link with one compiler, call via ctypes
(manual init/program/exit) -- not ``dace.compile()``, whose ``__call__`` re-marshals args and
confounds timing. Entry points: ``__dace_init_N``/``__program_N``/``__dace_exit_N``.

What the machine's toolchains can DO -- compiler families, OpenMP runtimes, vector-math libraries,
linkers, ccache, C-signature parsing -- lives in :mod:`nestforge.build.toolchain`, which mentions dace nowhere.
That half was the majority of this file and is used by five modules that never build an SDFG, so asking
it a question no longer imports the codegen stack.
"""
from __future__ import annotations

import contextlib
import copy
import ctypes
import functools
from _ctypes import dlclose  # release a built .so mapping (BuiltSDFG.unload)
import time
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

import dace
from dace.codegen import codegen
from dace.codegen import compiler as dace_compiler
from dace.transformation.auto.auto_optimize import set_fast_implementations

from nestforge.build.toolchain import (CXX_STD, DEFAULT_COMPILER, DEFAULT_FLAGS, OpenMPRuntime, Param, VectorMathLib,
                                       ar_for, ccache_prefix, fastest_linker, fat_lto_flags, parse_params, run,
                                       signature, support_rpath_flags, usable_openmp)

# TODO(blas): a BLAS/LAPACK axis (openblas/mkl/blis/nvpl/accelerate) the same way -- discovery exists
# (arena.discover_blas_libraries); missing is threading a chosen BLAS into the link line + a prune step.


@functools.lru_cache(maxsize=None, typed=True)
def dace_runtime_include() -> Path:
    """The ``-I`` directory holding DaCe's runtime headers (``dace/runtime/include``)."""
    inc = Path(dace.__file__).parent / "runtime" / "include"
    if not inc.is_dir():
        raise FileNotFoundError(f"DaCe runtime include not found at {inc}")
    return inc


@dataclass(slots=True)
class BuiltSDFG:
    """A nest-forge-built DaCe ``.so`` with its entry points bound and init/exit managed."""
    name: str
    so_path: Path
    _lib: ctypes.CDLL
    _init_params: List[Param]
    _prog_params: List[Param]
    #: wall time of the OPTIMIZATION phase (DaCe codegen + C++ emission), distinct from the compile below.
    codegen_seconds: float = 0.0
    #: wall time of the post-optimization COMPILE (compiler/linker turning C++ into the ``.so``).
    compile_seconds: float = 0.0
    _handle: Optional[ctypes.c_void_p] = field(default=None, repr=False)

    def init(self, sizes: Dict[str, int]) -> None:
        fn = self._lib[f"__dace_init_{self.name}"]  # ctypes CDLL indexing (not getattr) binds the entry point
        fn.restype = ctypes.c_void_p
        fn.argtypes = [p.ctype for p in self._init_params]
        # each param's OWN ctype: a hardcoded width would mismatch (jacobi's int N vs gemm's int64_t NI)
        self._handle = ctypes.c_void_p(fn(*[p.ctype(int(sizes[p.name])) for p in self._init_params]))

    def bind_program(self, buffers: Dict[str, np.ndarray], sizes: Dict[str, int]) -> Tuple[Any, list]:
        """Bind ``__program_N`` and its ctypes args ONCE; return ``(fn, args)``, so a timed rep loop calls
        ``fn(*args)`` with no per-rep marshaling. ``init`` must have run; ``buffers`` must stay alive."""
        fn = self._lib[f"__program_{self.name}"]
        fn.restype = None
        fn.argtypes = [ctypes.c_void_p] + [p.ctype for p in self._prog_params]
        args = [self._handle]
        for p in self._prog_params:
            if p.is_pointer:
                args.append(buffers[p.name].ctypes.data_as(p.ctype))
            elif p.name in buffers:  # a DaCe Scalar passed by value
                args.append(p.ctype(buffers[p.name].item()))
            else:  # a size symbol
                args.append(p.ctype(int(sizes[p.name])))
        return fn, args

    def program(self, buffers: Dict[str, np.ndarray], sizes: Dict[str, int]) -> None:
        """Call ``__program_N(handle, args...)`` once, in place (init must have run)."""
        fn, args = self.bind_program(buffers, sizes)
        fn(*args)

    def unload(self) -> None:
        """Release the ``dlopen`` mapping (file may be deleted after). Prevents a long sweep from
        accumulating one live mapping per kernel."""
        if self._lib is not None:
            dlclose(self._lib._handle)
            self._lib = None

    def close(self) -> None:
        if self._handle is not None:
            fn = self._lib[f"__dace_exit_{self.name}"]
            fn.restype = ctypes.c_int
            fn.argtypes = [ctypes.c_void_p]
            fn(self._handle)
            self._handle = None

    def run(self, buffers: Dict[str, np.ndarray], sizes: Dict[str, int]) -> None:
        """One-shot init -> program -> exit (for correctness; for timing, init once + loop program)."""
        self.init(sizes)
        try:
            self.program(buffers, sizes)
        finally:
            self.close()


def config_has(*path) -> bool:
    """True when the running DaCe config schema DEFINES the key at ``path`` (``Config.get`` raises on an
    unknown key), so the codegen axis degrades gracefully instead of crashing."""
    try:
        dace.config.Config.get(*path)
        return True
    except (KeyError, TypeError):
        return False


#: Codegen-implementation axis (``compiler.cpu.implementation``): ``experimental`` is the readable
#: constexpr-index-fn codegen (nest-forge's DEFAULT); ``legacy`` is the classic connector-based codegen.
CODEGEN_IMPLS = ("experimental", "legacy")
#: nest-forge defaults to DaCe's NEW (human-readable) codegen when the running DaCe build supports it.
DEFAULT_CODEGEN_IMPL = "experimental"


def default_codegen_impl() -> str:
    """Codegen impl used when the caller specifies none: ``experimental`` if this DaCe build supports
    ``compiler.cpu.implementation``, else ``legacy``."""
    return DEFAULT_CODEGEN_IMPL if config_has("compiler", "cpu", "implementation") else "legacy"


def codegen_impls_available() -> Tuple[str, ...]:
    """Codegen-implementation values THIS DaCe build supports, default first: both when the schema has
    ``compiler.cpu.implementation``, else just ``('legacy',)``. The driver sweeps exactly this tuple."""
    return CODEGEN_IMPLS if config_has("compiler", "cpu", "implementation") else ("legacy", )


@contextlib.contextmanager
def codegen_config(codegen_impl: str) -> Iterator[None]:
    """Scope the DaCe codegen config for ONE ``generate_code`` call: pin ``emit_tree_reductions`` true and
    select the CPU codegen ``implementation``. Raises for ``experimental`` on a build lacking the key,
    rather than silently emitting legacy and mislabelling it."""
    with dace.config.temporary_config():
        dace.config.Config.set("compiler", "emit_tree_reductions", value=True)
        if config_has("compiler", "cpu", "implementation"):
            dace.config.Config.set("compiler", "cpu", "implementation", value=codegen_impl)
        elif codegen_impl != "legacy":
            raise ValueError(f"codegen_impl={codegen_impl!r} requested, but this DaCe build has no "
                             "'compiler.cpu.implementation' key (needs the experimental-codegen branch)")
        yield


def generate_program_folder(sdfg: dace.SDFG, out_dir: Path, codegen_impl: Optional[str] = None) -> Tuple[Path, str]:
    """Lay out DaCe's compilable source tree (``src/cpu/<name>.cpp`` + ``include/``) via DaCe's own
    ``generate_program_folder`` so relative includes resolve, but WITHOUT letting DaCe compile it.

    :param codegen_impl: ``experimental`` | ``legacy``; ``None`` -> :func:`default_codegen_impl`.
    :returns: (the C++ Frame source path, sdfg name).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    with codegen_config(codegen_impl or default_codegen_impl()):
        code_objects = codegen.generate_code(sdfg)
    folder = Path(dace_compiler.generate_program_folder(sdfg, code_objects, str(out_dir)))
    frame = folder / "src" / "cpu" / f"{sdfg.name}.cpp"
    if not frame.exists():  # fall back to whatever CPU Frame the layout produced
        frame = next(folder.glob("src/cpu/*.cpp"))
    return frame, sdfg.name


def include_flags(folder: Path) -> List[str]:
    """Header search paths: the generated ``include/`` and DaCe's runtime include."""
    return [f"-I{folder / 'include'}", f"-I{dace_runtime_include()}"]


@dataclass(slots=True)
class BuildOptions:
    """Toolchain + optimization knobs for the owned build (:func:`build_sdfg` / :func:`compare_link_modes`
    take this instead of a long parameter list). Each axis is independent."""
    compiler: str = DEFAULT_COMPILER
    flags: Optional[List[str]] = None  # None -> DEFAULT_FLAGS
    expand_libnodes: bool = False  # expand library nodes to naive loops ("without libnodes" variant)
    fast_libnodes: bool = False  # instead of expanding, pick the fast library impl (OpenBLAS/MKL)
    blas_link: Optional[List[str]] = None  # link flags for the chosen BLAS (e.g. ['-lopenblas'])
    openmp: Optional[OpenMPRuntime] = None  # the one mandated runtime to link (per-compiler flags)
    link_external: bool = False  # link the nest as a separate static .a (else a monolithic single TU)
    lto: bool = False  # enable LTO: -flto (monolithic) / fat-LTO object in the .a (external)
    veclib: Optional[VectorMathLib] = None  # SLEEF / libmvec / SVML, a separate axis from flags/openmp
    # DaCe CPU codegen: 'experimental' (DEFAULT where available) | 'legacy'; downgrades on an older build.
    codegen_impl: str = field(default_factory=default_codegen_impl)
    # DaCe multi-dim tile-op vectorizer config applied before codegen; None = no vectorization. Typed as
    # object to keep the vectorizer import lazy.
    vectorize: Optional[object] = None
    # Extra link arguments appended AFTER the frame object -- the extern nest-variant libs (absolute .so
    # path + its -Wl,-rpath) for the differential swap path. nest-forge compiles the generated frame
    # directly (bypassing DaCe's CMake), so ``ExternLibEnv``'s libraries are NOT auto-linked; the caller
    # passes them here instead. Placed after the frame so ld resolves the extern-C entry symbols.
    extra_link: Optional[List[str]] = None
    # Compiler cache: None = AUTO (use ccache when installed), False = never. Set False for any build whose
    # compile_seconds is reported as a measurement -- a cache hit reports ~0s (see ccache_prefix).
    use_ccache: Optional[bool] = None

    def resolved_flags(self) -> List[str]:
        """``flags`` (or :data:`DEFAULT_FLAGS`), with the C++ standard and ``-Wall`` guaranteed.

        The standard is a REQUIREMENT of the DaCe runtime headers, not an optimization knob: ``types.h``
        uses ``std::bit_cast`` unguarded, so anything below C++20 fails to compile. A caller overriding
        ``flags`` for one axis (``-O2``, a veclib) must not silently lose it -- an explicit ``-std=`` is
        honored, an absent one is filled in.

        ``-Wall`` but deliberately NOT ``-Werror``: this compiles DACE-GENERATED C++, so a warning is a
        codegen signal to read, not a reason to fail a measurement we do not control the source of.
        :func:`toolchain.run` prints what it produces. Filled in the same way as ``-std=``, since most
        callers pass their own ``flags`` for one axis and would otherwise lose it; an explicit ``-w``
        (silence) is honored."""
        flags = list(self.flags if self.flags is not None else DEFAULT_FLAGS)
        if not any(f.startswith("-std=") for f in flags):
            flags.append(f"-std={CXX_STD}")
        if "-Wall" not in flags and "-w" not in flags:
            flags.append("-Wall")
        return flags


def set_fast_libnodes(sdfg: dace.SDFG) -> None:
    """Select the fastest AVAILABLE library-node implementation (OpenBLAS/MKL/LAPACK) for every library
    node, instead of lowering to naive loops. Link flags come via :attr:`BuildOptions.blas_link`.

    TODO(lib-axis): generalize into a per-node "try every backend" sweep, keeping the timed winner."""
    set_fast_implementations(sdfg, dace.dtypes.DeviceType.CPU)


@dataclass(slots=True)
class BuildCommands:
    """The argument groups every compile and link of one build shares, resolved before any clock starts."""
    compiler: str
    ccache: List[str]
    cflags: List[str]  # resolved flags minus the link-only -shared
    compile_extra: List[str]  # OpenMP + veclib compile halves, then the include path
    ld: List[str]
    link_libs: List[str]  # after the object: ld resolves left to right


def build_commands(folder: Path, opts: BuildOptions) -> BuildCommands:
    """Resolve the compiler, OpenMP runtime, veclib, linker and cache for one build of ``folder``."""
    compiler = opts.compiler
    # dace emits `#pragma omp parallel for` for every multicore map; a build without OpenMP runs it serially.
    omp = opts.openmp or usable_openmp(compiler)
    if omp is None:
        warnings.warn(f"{Path(compiler).name} can link no OpenMP runtime; building SERIAL -- any parallel "
                      "map in this SDFG is emitted as an ignored pragma and the timing is single-threaded")
    omp_c = omp.compile_flags(compiler) if omp else []
    omp_l = omp.link_flags(compiler) if omp else []
    vec_c = opts.veclib.compile_flags(compiler) if opts.veclib else []
    vec_l = opts.veclib.link_flags(compiler) if opts.veclib else []
    # icx auto-links libsvml/libimf off the loader path with no RUNPATH; without this dlopen fails.
    libs = [*omp_l, *vec_l, *(opts.blas_link or []), *(opts.extra_link or []), *support_rpath_flags(compiler)]
    return BuildCommands(compiler=compiler,
                         ccache=ccache_prefix(opts.use_ccache),
                         cflags=[f for f in opts.resolved_flags() if f != "-shared"],
                         compile_extra=[*omp_c, *vec_c, *include_flags(folder)],
                         ld=fastest_linker(compiler),
                         link_libs=libs)


def build_archive(sources: Sequence[Path], folder: Path, archive: Path, shared: Path, opts: BuildOptions) -> float:
    """Compile ``sources`` to ``<archive dir>/<stem>.o`` against ``folder``'s headers, archive them, and link
    ``shared`` from the whole archive; returns toolchain wall seconds."""
    cmds = build_commands(folder, opts)
    lto_c = fat_lto_flags(opts.compiler) if opts.lto else []
    ar = ar_for(opts.compiler)
    objs = [archive.parent / f"{src.stem}.o" for src in sources]
    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive.exists():
        archive.unlink()  # ar r APPENDS; start clean so a rebuild doesn't stack stale members
    t0 = time.perf_counter()
    for src, obj in zip(sources, objs):
        run([*cmds.ccache, cmds.compiler, *cmds.cflags, *lto_c, "-c", *cmds.compile_extra, str(src), "-o", str(obj)])
    run([ar, "rcs", str(archive), *[str(obj) for obj in objs]])
    # link the objects' real code (not -flto, so no compile flags: clang warns on unused FP ones)
    run([
        cmds.compiler, "-shared", *cmds.ld, "-Wl,--export-dynamic", "-Wl,--whole-archive",
        str(archive), "-Wl,--no-whole-archive", *cmds.link_libs, "-o",
        str(shared)
    ])
    return time.perf_counter() - t0


def compile(frame: Path, folder: Path, name: str, opts: BuildOptions) -> Tuple[Path, float]:
    """Compile the generated frame into ``lib<name>.so``; return (path, toolchain wall_seconds only).
    ``link_external=False`` builds one TU monolithically; ``True`` goes through :func:`build_archive`.

    Compiling and linking are always separate commands: a compile is cacheable, a link is not, and ccache
    declines any command that does both."""
    so = folder / f"lib{name}.so"
    if opts.link_external:
        return so, build_archive([frame], folder, folder / f"lib{name}_nest.a", so, opts)
    cmds = build_commands(folder, opts)
    obj = folder / f"{name}.o"
    lto_f = ["-flto"] if opts.lto else []
    t0 = time.perf_counter()
    run([*cmds.ccache, cmds.compiler, *cmds.cflags, *lto_f, "-c", *cmds.compile_extra, str(frame), "-o", str(obj)])
    run([cmds.compiler, "-shared", *cmds.cflags, *lto_f, *cmds.ld, str(obj), *cmds.link_libs, "-o", str(so)])
    return so, time.perf_counter() - t0


def apply_vectorizer(sdfg: dace.SDFG, config: object) -> None:
    """Apply the DaCe multi-dim tile-op CPU vectorizer to ``sdfg`` in place. Force-expands tile library
    nodes to tasklets regardless of ``config`` (no ``dace.compile`` here to lower them later). Lazy
    import: eager would close an import cycle."""
    import dataclasses
    from dace.transformation.passes.vectorization import VectorizeCPUMultiDim
    VectorizeCPUMultiDim(dataclasses.replace(config, expand_tile_nodes=True)).apply_pass(sdfg, {})


@dataclass(slots=True)
class GeneratedProgram:
    """The optimization phase's output: emitted source, not yet compiled.

    Split out of :func:`build_sdfg` so a search can hash ``source`` and SKIP a compile that would emit
    byte-identical code -- the cheapest possible screen, since a knob whose flip changes nothing here
    cannot change a timing.
    """
    frame: Path  # the frame .cpp DaCe emitted
    name: str
    source: str
    codegen_seconds: float

    @property
    def folder(self) -> Path:
        return self.frame.parent.parent.parent  # <out>/src/cpu/x.cpp -> <out>


def generate_program(sdfg: dace.SDFG, out_dir: Path, opts: Optional[BuildOptions] = None) -> GeneratedProgram:
    """Run the optimization phase only: apply the configured passes and emit the program folder."""
    opts = opts or BuildOptions()
    t_opt = time.perf_counter()
    sdfg = copy.deepcopy(sdfg)
    if opts.expand_libnodes:
        sdfg.expand_library_nodes()
    elif opts.fast_libnodes:  # keep the library nodes, but pick the fast (OpenBLAS/MKL) implementation
        set_fast_libnodes(sdfg)
    if opts.vectorize is not None:
        apply_vectorizer(sdfg, opts.vectorize)
    frame, name = generate_program_folder(sdfg, out_dir, opts.codegen_impl)
    return GeneratedProgram(frame=frame,
                            name=name,
                            source=frame.read_text(),
                            codegen_seconds=time.perf_counter() - t_opt)


def compile_program(gen: GeneratedProgram, opts: Optional[BuildOptions] = None) -> BuiltSDFG:
    """Compile + link an already-generated program and bind its entry points."""
    opts = opts or BuildOptions()
    init_params = parse_params(signature(gen.source, f"__dace_init_{gen.name}"))
    prog_params = parse_params(signature(gen.source, f"__program_{gen.name}"))
    so, compile_seconds = compile(gen.frame, gen.folder, gen.name, opts)
    return BuiltSDFG(name=gen.name,
                     so_path=so,
                     _lib=ctypes.CDLL(str(so)),
                     _init_params=init_params,
                     _prog_params=prog_params,
                     codegen_seconds=gen.codegen_seconds,
                     compile_seconds=compile_seconds)


def build_sdfg(sdfg: dace.SDFG, out_dir: Path, opts: Optional[BuildOptions] = None) -> BuiltSDFG:
    """Generate + compile + link an SDFG ourselves; return a :class:`BuiltSDFG` carrying
    ``codegen_seconds``/``compile_seconds`` timing.

    :param opts: toolchain + optimization knobs; ``None`` uses all defaults (g++, monolithic, no veclib) --
        including a RESOLVED OpenMP runtime, since dace emits ``#pragma omp parallel for`` for every
        multicore map and a build without one runs that schedule serially. A caller comparing against a
        serial baseline must therefore pin ``OMP_NUM_THREADS=1`` rather than assume no OpenMP.
    """
    opts = opts or BuildOptions()
    return compile_program(generate_program(sdfg, out_dir, opts), opts)


@dataclass(slots=True)
class LinkTimings:
    """Optimization time and the two post-optimization compile times isolated on ONE codegen."""
    codegen_seconds: float  # the optimization (DaCe codegen) phase, run once
    compile_seconds_monolithic: float  # WITHOUT external linking (single TU)
    compile_seconds_external: float  # WITH external linking (static .a -> .so)


def compare_link_modes(sdfg: dace.SDFG, out_dir: Path, opts: Optional[BuildOptions] = None) -> LinkTimings:
    """Generate the code ONCE, then compile that same frame both monolithically and externally, so
    ``compile_seconds`` is the only thing that differs. ``opts``' link mode is overridden per build; its
    other axes apply to both."""
    opts = opts or BuildOptions()
    # the SAME optimization phase build_sdfg runs, so the two link modes compare the code the caller's
    # opts actually produce -- an inlined copy here silently dropped the vectorizer from the comparison
    gen = generate_program(sdfg, out_dir, opts)
    # compile time IS the result here, so the cache is forced OFF: a hit would report ~0s and make the
    # monolithic-vs-external comparison meaningless.
    _, mono = compile(gen.frame, gen.folder, gen.name, replace(opts, link_external=False, use_ccache=False))
    _, ext = compile(gen.frame, gen.folder, gen.name, replace(opts, link_external=True, use_ccache=False))
    return LinkTimings(codegen_seconds=gen.codegen_seconds,
                       compile_seconds_monolithic=mono,
                       compile_seconds_external=ext)
