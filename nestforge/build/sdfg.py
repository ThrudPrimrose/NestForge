# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Owns the DaCe build: codegen then compile/link via ctypes (manual init/program/exit), not
``dace.compile()``, whose ``__call__`` re-marshals args and confounds timing."""
from __future__ import annotations

import contextlib
import copy
import ctypes
import functools
from _ctypes import dlclose  # release a built .so mapping (BuiltSDFG.unload)
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

import dace
from dace.codegen import codegen
from dace.codegen import compiler as dace_compiler

from nestforge.build.toolchain import (CXX_STD, DEFAULT_COMPILER, DEFAULT_FLAGS, OpenMPRuntime, Param, ar_for,
                                       parse_params, run, signature, support_rpath_flags, usable_openmp)


@functools.lru_cache(maxsize=None, typed=True)
def dace_runtime_include() -> Path:
    """The ``-I`` directory holding DaCe's runtime headers."""
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
    #: wall time of DaCe codegen + C++ emission (the optimization phase).
    codegen_seconds: float = 0.0
    #: wall time of the compiler/linker turning C++ into the .so.
    compile_seconds: float = 0.0
    _handle: Optional[ctypes.c_void_p] = field(default=None, repr=False)

    def init(self, sizes: Dict[str, int]) -> None:
        fn = self._lib[f"__dace_init_{self.name}"]  # ctypes CDLL indexing (not getattr) binds the entry point
        fn.restype = ctypes.c_void_p
        fn.argtypes = [p.ctype for p in self._init_params]
        # each param's OWN ctype: a hardcoded width would mismatch (jacobi's int N vs gemm's int64_t NI)
        self._handle = ctypes.c_void_p(fn(*[p.ctype(int(sizes[p.name])) for p in self._init_params]))

    def bind_program(self, buffers: Dict[str, np.ndarray], sizes: Dict[str, int]) -> Tuple[Any, list]:
        """Bind ``__program_N`` and its ctypes args once, so a timed rep loop calls ``fn(*args)`` with no per-rep marshaling."""
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
        """Release the dlopen mapping so a long sweep does not accumulate one live mapping per kernel."""
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


@contextlib.contextmanager
def codegen_config() -> Iterator[None]:
    """Scope the DaCe codegen config for one ``generate_code`` call."""
    with dace.config.temporary_config():
        dace.config.Config.set("compiler", "emit_tree_reductions", value=True)
        yield


def generate_program_folder(sdfg: dace.SDFG, out_dir: Path) -> Tuple[Path, str]:
    """Lay out DaCe's compilable source tree (``src/cpu/<name>.cpp`` + ``include/``), without letting DaCe compile it."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with codegen_config():
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
    """Toolchain + optimization knobs for the owned build; each axis is independent."""
    compiler: str = DEFAULT_COMPILER
    flags: Optional[List[str]] = None  # None -> DEFAULT_FLAGS
    expand_libnodes: bool = False
    blas_link: Optional[List[str]] = None
    openmp: Optional[OpenMPRuntime] = None
    link_external: bool = False  # link the nest as a separate static .a (else a monolithic single TU)
    # object, not the vectorizer's own config type, to keep the vectorizer import lazy
    vectorize: Optional[object] = None
    # extern nest-variant libs appended after the frame object, for the differential swap path: nest-forge
    # bypasses DaCe's CMake, so ExternLibEnv's libraries are not auto-linked and must be passed here
    extra_link: Optional[List[str]] = None

    def resolved_flags(self) -> List[str]:
        """``flags`` (or :data:`DEFAULT_FLAGS`), with the C++ standard and ``-Wall`` guaranteed."""
        # the DaCe runtime headers need C++20 (std::bit_cast unguarded); fill in -std= only if the caller's
        # own flags did not already set one, so overriding flags for one axis does not silently lose it
        flags = list(self.flags if self.flags is not None else DEFAULT_FLAGS)
        if not any(f.startswith("-std=") for f in flags):
            flags.append(f"-std={CXX_STD}")
        if "-Wall" not in flags and "-w" not in flags:
            flags.append("-Wall")
        return flags


@dataclass(slots=True)
class BuildCommands:
    compiler: str
    cflags: List[str]
    compile_extra: List[str]
    link_libs: List[str]  # after the object: the linker resolves left to right


def build_commands(folder: Path, opts: BuildOptions) -> BuildCommands:
    compiler = opts.compiler
    # dace emits `#pragma omp parallel for` for every multicore map; a build without OpenMP runs it serially.
    omp = opts.openmp or usable_openmp(compiler)
    if omp is None:
        warnings.warn(f"{Path(compiler).name} can link no OpenMP runtime; building SERIAL -- any parallel "
                      "map in this SDFG is emitted as an ignored pragma and the timing is single-threaded")
    omp_c = omp.compile_flags(compiler) if omp else []
    omp_l = omp.link_flags(compiler) if omp else []
    # icx auto-links libsvml/libimf off the loader path with no RUNPATH; without this dlopen fails.
    libs = [*omp_l, *(opts.blas_link or []), *(opts.extra_link or []), *support_rpath_flags(compiler)]
    return BuildCommands(compiler=compiler,
                         cflags=[f for f in opts.resolved_flags() if f != "-shared"],
                         compile_extra=[*omp_c, *include_flags(folder)],
                         link_libs=libs)


def build_archive(sources: Sequence[Path], folder: Path, archive: Path, shared: Path, opts: BuildOptions) -> float:
    """Compile ``sources`` against ``folder``'s headers, archive them, and link ``shared`` from the whole archive."""
    cmds = build_commands(folder, opts)
    ar = ar_for(opts.compiler)
    objs = [archive.parent / f"{src.stem}.o" for src in sources]
    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive.exists():
        archive.unlink()  # ar r APPENDS; start clean so a rebuild doesn't stack stale members
    t0 = time.perf_counter()
    for src, obj in zip(sources, objs):
        run([cmds.compiler, *cmds.cflags, "-c", *cmds.compile_extra, str(src), "-o", str(obj)])
    run([ar, "rcs", str(archive), *[str(obj) for obj in objs]])
    run([
        cmds.compiler, "-shared", "-Wl,--export-dynamic", "-Wl,--whole-archive",
        str(archive), "-Wl,--no-whole-archive", *cmds.link_libs, "-o",
        str(shared)
    ])
    return time.perf_counter() - t0


def compile(frame: Path, folder: Path, name: str, opts: BuildOptions) -> Tuple[Path, float]:
    """Compile the generated frame into ``lib<name>.so``; ``link_external`` picks monolithic vs :func:`build_archive`."""
    so = folder / f"lib{name}.so"
    if opts.link_external:
        return so, build_archive([frame], folder, folder / f"lib{name}_nest.a", so, opts)
    cmds = build_commands(folder, opts)
    obj = folder / f"{name}.o"
    t0 = time.perf_counter()
    run([cmds.compiler, *cmds.cflags, "-c", *cmds.compile_extra, str(frame), "-o", str(obj)])
    run([cmds.compiler, "-shared", *cmds.cflags, str(obj), *cmds.link_libs, "-o", str(so)])
    return so, time.perf_counter() - t0


def apply_vectorizer(sdfg: dace.SDFG, config: object) -> None:
    """Apply the DaCe multi-dim tile-op CPU vectorizer to ``sdfg`` in place."""
    import dataclasses  # lazy: closes an import cycle
    from dace.transformation.passes.vectorization import VectorizeCPUMultiDim
    VectorizeCPUMultiDim(dataclasses.replace(config, expand_tile_nodes=True)).apply_pass(sdfg, {})


@dataclass(slots=True)
class GeneratedProgram:
    """The optimization phase's output: emitted source, not yet compiled."""
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
    if opts.vectorize is not None:
        apply_vectorizer(sdfg, opts.vectorize)
    frame, name = generate_program_folder(sdfg, out_dir)
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
    """Generate + compile + link an SDFG ourselves. Resolves a real OpenMP runtime by default, so a caller
    comparing against serial must pin ``OMP_NUM_THREADS=1`` rather than assume none is linked."""
    opts = opts or BuildOptions()
    return compile_program(generate_program(sdfg, out_dir, opts), opts)
