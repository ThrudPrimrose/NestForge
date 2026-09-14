# Build

[../README.md](../README.md) · related: [4 Optimize Kernels](phases/4-optimize-kernels.md) ·
[5 Sweep Configurations](phases/5-sweep-configurations.md)

NestForge does not call `dace.compile()` for kernels: `nestforge/build/sdfg.py` and
`toolchain.py` generate the SDFG's source, compile and link it with one chosen compiler and flag
set, and `arena.py` calls the result directly. This keeps the DaCe-backend competitor and
the offloaded kernels on the same compiler and flags, so phase 5's timings compare codegen against
codegen rather than against `CompiledSDFG`'s own marshaling overhead.

## Owning the compile

`generate_program_folder` runs `codegen.generate_code(sdfg)` and writes the `CodeObject`s to disk
instead of letting DaCe build them. `compile` in `sdfg.py` then invokes the chosen compiler
directly, with DaCe's runtime headers (`dace_runtime_include`) on the include path.

## Manual init / run / exit

A DaCe-generated shared object exposes three C-linkage entry points for an SDFG named `N`:
`__dace_init_N`, `__program_N`, `__dace_exit_N`. `BuiltSDFG` (`sdfg.py`) binds all three through
`ctypes.CDLL` and calls them in that order: init allocates the SDFG's state and returns an opaque
handle, `__program_N` runs the kernel and is the only call phase 5 times, and exit frees the state.
`BuiltSDFG.unload` releases the `.so` mapping with `dlclose` when a build is discarded.

## Fork isolation

`nestforge/build/isolation.py` runs a freshly compiled kernel in a forked child
(`run_isolated`), so a segfault or a runaway loop in generated code cannot take down the process
driving the sweep. `os.fork()` duplicates only the calling thread, so a live OpenMP thread pool
across the fork deadlocks the child; `pause_openmp_pools` tears down every already-loaded
runtime's pool first. The parent enforces a wall-clock timeout and kills a child that does not
finish in time.

## Static archives and shared objects

`build_archive` in `sdfg.py` is the one archive path: it compiles translation units to objects,
archives them, and links a shared twin from the archive with `--whole-archive`. Kernel optimization
(`nestforge/phases/kernel.py`) builds `lib<kernel>.a` from DaCe's frame plus a generated wrapper TU
that defines the kernel's single `extern "C"` entry (init, run, exit). The twin `lib<kernel>.so`
exists only for ctypes validation and timing; the parent links the archive through `ExternalCall`.
Each kernel gets its own archive, since DaCe sorts the parent's link flags and a shared archive
would lose members.

## One OpenMP runtime

`nestforge.build.toolchain.OpenMPRuntime` names the single OpenMP runtime a build links against
(default `libomp`, since it is LLVM-selectable and also implements the GOMP ABI, so a GCC-built and
a Clang-built object can share one thread pool). `OpenMPRuntime.check` raises before compiling a
translation unit against a runtime a given compiler cannot actually link (classic Intel hard-links
its own runtime; LLVM selects by name; GNU accepts any GOMP-ABI runtime), which is how a
mixed-compiler build is kept off a mixed-runtime link.
