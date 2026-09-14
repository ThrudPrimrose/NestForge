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
from typing import Any, cast
from collections.abc import Iterator, Sequence

import numpy as np

import dace
from dace.codegen import codegen
from dace.codegen import compiler as dace_compiler

from nestforge.build.toolchain import (
    AR,
    CXX_STD,
    DEFAULT_COMPILER,
    DEFAULT_FLAGS,
    AS_NEEDED,
    LIBOMP,
    OpenMPRuntime,
    Param,
    cudart_dir,
    cudart_link_flags,
    parse_params,
    run,
    runtime_library,
    signature,
    support_rpath_flags,
    usable_openmp,
)


@functools.lru_cache(maxsize=None, typed=True)
def dace_runtime_include() -> Path:
    """The ``-I`` directory holding DaCe's runtime headers."""
    inc = Path(dace.__file__).parent / "runtime" / "include"
    if not inc.is_dir():
        raise FileNotFoundError(f"DaCe runtime include not found at {inc}")
    return inc


@dataclass(slots=True)
class BuiltSDFG:
    """A nest-forge-built DaCe ``.so`` with its entry points bound and init/exit managed. ``lib`` turns
    ``None`` once :meth:`unload` has released the dlopen mapping; every method that needs it fails with a
    clear error rather than a bare ``TypeError`` on ``None[...]``."""

    name: str
    so_path: Path
    lib: ctypes.CDLL | None
    init_params: list[Param]
    prog_params: list[Param]
    #: wall time of DaCe codegen + C++ emission (the optimization phase).
    codegen_seconds: float = 0.0
    #: wall time of the compiler/linker turning C++ into the .so.
    compile_seconds: float = 0.0
    handle: ctypes.c_void_p | None = field(default=None, repr=False)

    def init(self, sizes: dict[str, int]) -> None:
        if self.lib is None:
            raise RuntimeError(f"{self.name}: init() called after unload(); the compiled library is not mapped")
        fn = self.lib[f"__dace_init_{self.name}"]  # ctypes CDLL indexing (not getattr) binds the entry point
        fn.restype = ctypes.c_void_p
        fn.argtypes = [p.ctype for p in self.init_params]
        # each param's OWN ctype: a hardcoded width would mismatch (jacobi's int N vs gemm's int64_t NI)
        self.handle = ctypes.c_void_p(fn(*[p.ctype(int(sizes[p.name])) for p in self.init_params]))

    def bind_program(self, buffers: dict[str, np.ndarray], sizes: dict[str, int]) -> tuple[Any, list]:
        """Bind ``__program_N`` and its ctypes args once, so a timed rep loop calls ``fn(*args)`` with no per-rep marshaling."""
        if self.lib is None:
            raise RuntimeError(f"{self.name}: bind_program() called after unload(); the compiled library is not mapped")
        fn = self.lib[f"__program_{self.name}"]
        fn.restype = None
        fn.argtypes = [ctypes.c_void_p] + [p.ctype for p in self.prog_params]
        args: list[Any] = [self.handle]
        for p in self.prog_params:
            if p.is_pointer:
                # is_pointer guarantees ctype is a Pointer type (parse_params); the cast reflects that.
                args.append(buffers[p.name].ctypes.data_as(cast(type[ctypes._Pointer], p.ctype)))
            elif p.name in buffers:  # a DaCe Scalar passed by value
                args.append(p.ctype(buffers[p.name].item()))
            else:  # a size symbol
                args.append(p.ctype(int(sizes[p.name])))
        return fn, args

    def program(self, buffers: dict[str, np.ndarray], sizes: dict[str, int]) -> None:
        """Call ``__program_N(handle, args...)`` once, in place (init must have run)."""
        fn, args = self.bind_program(buffers, sizes)
        fn(*args)

    def unload(self) -> None:
        """Release the dlopen mapping so a long sweep does not accumulate one live mapping per kernel."""
        if self.lib is not None:
            dlclose(self.lib._handle)
            self.lib = None

    def close(self) -> None:
        """Run ``__dace_exit`` on a live handle. A no-op with no handle open; raises if the library was
        unloaded while a handle was still open, since there is then nothing left to call ``__dace_exit`` on."""
        if self.handle is None:
            return
        if self.lib is None:
            raise RuntimeError(
                f"{self.name}: close() called after unload() with a handle still open; the library needed "
                "to run __dace_exit is no longer mapped -- call close() before unload()"
            )
        fn = self.lib[f"__dace_exit_{self.name}"]
        fn.restype = ctypes.c_int
        fn.argtypes = [ctypes.c_void_p]
        fn(self.handle)
        self.handle = None

    def run(self, buffers: dict[str, np.ndarray], sizes: dict[str, int]) -> None:
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


def generate_program_folder(sdfg: dace.SDFG, out_dir: Path) -> tuple[Path, str]:
    """Lay out DaCe's compilable source tree (``src/cpu/<name>.cpp`` + ``include/``), without letting DaCe compile it."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with codegen_config():
        code_objects = codegen.generate_code(sdfg)
    folder = Path(dace_compiler.generate_program_folder(sdfg, code_objects, str(out_dir)))
    frame = folder / "src" / "cpu" / f"{sdfg.name}.cpp"
    if not frame.exists():  # fall back to whatever CPU Frame the layout produced
        frame = next(folder.glob("src/cpu/*.cpp"))
    return frame, sdfg.name


def include_flags(folder: Path) -> list[str]:
    """Header search paths: the generated ``include/`` and DaCe's runtime include."""
    return [f"-I{folder / 'include'}", f"-I{dace_runtime_include()}"]


@dataclass(slots=True)
class BuildOptions:
    """Toolchain + optimization knobs for the owned build; each axis is independent."""

    compiler: str = DEFAULT_COMPILER
    flags: list[str] | None = None  # None -> DEFAULT_FLAGS
    expand_libnodes: bool = False
    openmp: OpenMPRuntime | None = None
    link_external: bool = False  # link the nest as a separate static .a (else a monolithic single TU)
    # object, not the vectorizer's own config type, to keep the vectorizer import lazy
    vectorize: object | None = None

    def resolved_flags(self) -> list[str]:
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
    cflags: list[str]
    compile_extra: list[str]
    link_libs: list[str]  # after the object: the linker resolves left to right


def build_commands(folder: Path | None, opts: BuildOptions) -> BuildCommands:
    compiler = opts.compiler
    # dace emits `#pragma omp parallel for` for every multicore map; a build without OpenMP runs it serially.
    omp = opts.openmp or usable_openmp(compiler)
    if omp is None:
        warnings.warn(
            f"{Path(compiler).name} can link no OpenMP runtime; building SERIAL -- any parallel "
            "map in this SDFG is emitted as an ignored pragma and the timing is single-threaded"
        )
    omp_c = omp.compile_flags(compiler) if omp else []
    omp_l = omp.link_flags(compiler) if omp else []
    # icx auto-links libsvml/libimf off the loader path with no RUNPATH; without this dlopen fails.
    libs = [*omp_l, *support_rpath_flags(compiler)]
    return BuildCommands(
        compiler=compiler,
        cflags=[f for f in opts.resolved_flags() if f != "-shared"],
        compile_extra=[*omp_c, *(include_flags(folder) if folder is not None else [])],
        link_libs=libs,
    )


def build_archive(
    sources: Sequence[Path], folder: Path | None, archive: Path, shared: Path, opts: BuildOptions
) -> float:
    """Compile ``sources`` (against ``folder``'s headers, if given), archive them, and link ``shared`` from the
    whole archive."""
    cmds = build_commands(folder, opts)
    objs = [archive.parent / f"{src.stem}.o" for src in sources]
    archive.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    for src, obj in zip(sources, objs):
        run([cmds.compiler, *cmds.cflags, "-c", *cmds.compile_extra, str(src), "-o", str(obj)])
    archive_and_link(objs, archive, shared, opts.compiler, cmds.link_libs)
    return time.perf_counter() - t0


def archive_and_link(
    objects: Sequence[Path], archive: Path, shared: Path, linker: str, link_libs: Sequence[str]
) -> None:
    """Archive ``objects`` into ``archive`` and link ``shared`` from the whole archive."""
    if archive.exists():
        archive.unlink()  # ar r APPENDS; start clean so a rebuild doesn't stack stale members
    run([AR, "rcs", str(archive), *[str(obj) for obj in objects]])
    whole = ["-Wl,--export-dynamic", "-Wl,--whole-archive", str(archive), "-Wl,--no-whole-archive"]
    link_shared(linker, whole, link_libs, shared)


def link_shared(linker: str, inputs: Sequence[str], link_libs: Sequence[str], shared: Path) -> None:
    """The one shared-library link: :data:`AS_NEEDED` ahead of ``inputs``, then the libraries."""
    run([linker, "-shared", AS_NEEDED, *inputs, *link_libs, "-o", str(shared)])


def build_cuda_archive(source: Path, archive: Path, shared: Path, nvcc: str, flags: Sequence[str]) -> float:
    """Compile one CUDA unit with ``nvcc``, archive it, and link ``shared`` with the host compiler against the
    ``libcudart`` that ``nvcc`` itself links."""
    obj = archive.parent / f"{source.stem}.o"
    archive.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    run([nvcc, *[f for f in flags if f != "-shared"], "-c", str(source), "-o", str(obj)])
    archive_and_link([obj], archive, shared, DEFAULT_COMPILER, cudart_link_flags(cudart_dir(nvcc)))
    return time.perf_counter() - t0


def compile(frame: Path, folder: Path, name: str, opts: BuildOptions) -> tuple[Path, float]:
    """Compile the generated frame into ``lib<name>.so``; ``link_external`` picks monolithic vs :func:`build_archive`."""
    so = folder / f"lib{name}.so"
    if opts.link_external:
        return so, build_archive([frame], folder, folder / f"lib{name}_nest.a", so, opts)
    cmds = build_commands(folder, opts)
    obj = folder / f"{name}.o"
    t0 = time.perf_counter()
    run([cmds.compiler, *cmds.cflags, "-c", *cmds.compile_extra, str(frame), "-o", str(obj)])
    link_shared(cmds.compiler, [*cmds.cflags, str(obj)], cmds.link_libs, so)
    return so, time.perf_counter() - t0


def program_compiler() -> str:
    """The C++ compiler DaCe's program build runs: ``compiler.cpu.executable``, else CMake's ``c++``."""
    return str(dace.config.Config.get("compiler", "cpu", "executable") or "c++")


def libomp_cmake_args(compiler: str) -> list[str]:
    """CMake cache values under which DaCe's ``find_package(OpenMP)`` resolves LLVM libomp for ``compiler``."""
    library = runtime_library(LIBOMP.soname, compiler)
    if library is None:
        raise LookupError(
            f"no lib{LIBOMP.soname} for {compiler}: neither it, an LLVM driver, llvm-config nor the library search path "
            "names one, and the process's one OpenMP runtime is libomp"
        )
    return [f"-DOpenMP_CXX_LIB_NAMES={LIBOMP.soname}", f"-DOpenMP_{LIBOMP.soname}_LIBRARY={library}"]


def compile_linked_program(sdfg: dace.SDFG, build_folder: Path) -> Any:
    """Compile a program that links kernel libraries, with libomp as its one OpenMP runtime; the CMake
    setting holds for this compile only."""
    extra = [dace.config.Config.get("compiler", "extra_cmake_args"), *libomp_cmake_args(program_compiler())]
    # DaCe folds compiler.linker.args into the one CMAKE_SHARED_LINKER_FLAGS it passes, after extra_cmake_args
    linker_args = [dace.config.Config.get("compiler", "linker", "args"), AS_NEEDED]
    sdfg.build_folder = str(build_folder)
    with dace.config.set_temporary("compiler", "extra_cmake_args", value=" ".join(arg for arg in extra if arg)):
        with dace.config.set_temporary("compiler", "linker", "args", value=" ".join(arg for arg in linker_args if arg)):
            return sdfg.compile()


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


def generate_program(sdfg: dace.SDFG, out_dir: Path, opts: BuildOptions | None = None) -> GeneratedProgram:
    """Run the optimization phase only: apply the configured passes and emit the program folder."""
    opts = opts or BuildOptions()
    t_opt = time.perf_counter()
    sdfg = copy.deepcopy(sdfg)
    if opts.expand_libnodes:
        sdfg.expand_library_nodes()
    if opts.vectorize is not None:
        apply_vectorizer(sdfg, opts.vectorize)
    frame, name = generate_program_folder(sdfg, out_dir)
    return GeneratedProgram(
        frame=frame, name=name, source=frame.read_text(), codegen_seconds=time.perf_counter() - t_opt
    )


def compile_program(gen: GeneratedProgram, opts: BuildOptions | None = None) -> BuiltSDFG:
    """Compile + link an already-generated program and bind its entry points."""
    opts = opts or BuildOptions()
    init_params = parse_params(signature(gen.source, f"__dace_init_{gen.name}"))
    prog_params = parse_params(signature(gen.source, f"__program_{gen.name}"))
    so, compile_seconds = compile(gen.frame, gen.folder, gen.name, opts)
    return BuiltSDFG(
        name=gen.name,
        so_path=so,
        lib=ctypes.CDLL(str(so)),
        init_params=init_params,
        prog_params=prog_params,
        codegen_seconds=gen.codegen_seconds,
        compile_seconds=compile_seconds,
    )


def build_sdfg(sdfg: dace.SDFG, out_dir: Path, opts: BuildOptions | None = None) -> BuiltSDFG:
    """Generate + compile + link an SDFG ourselves. Resolves a real OpenMP runtime by default, so a caller
    comparing against serial must pin ``OMP_NUM_THREADS=1`` rather than assume none is linked."""
    opts = opts or BuildOptions()
    return compile_program(generate_program(sdfg, out_dir, opts), opts)
