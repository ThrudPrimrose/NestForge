# Build

[Overview](../README.md) · related: [4 Optimize Kernels](phases/4-optimize-kernels.md),
[5 Sweep Configurations](phases/5-sweep-configurations.md)

`nestforge/build/` compiles with one chosen compiler and flag set and calls the result through
ctypes, so timings compare generated code, free of `CompiledSDFG` marshaling.

## Program builds

`generate_program_folder` writes DaCe's generated code to disk, and `sdfg.py` compiles it with DaCe's
runtime headers on the include path. `BuiltSDFG` binds the three C entries of an SDFG `N`
(`__dace_init_N`, `__program_N`, `__dace_exit_N`), calls them in order and times only `__program_N`;
`unload` closes the `.so`. `compile_linked_program` builds a program that links kernel libraries.

## Kernel libraries

`build_archive` compiles a kernel's CPF unit into `lib<kernel>.a` and links a shared twin with
`--whole-archive` for ctypes validation and timing. The program links the archive through
`ExternalCall`. Every kernel gets its own archive, because DaCe sorts the program's link flags and a
shared archive would lose members.

## Fork isolation

`run_isolated` (`isolation.py`) runs fresh code in a forked child under a wall-clock timeout, so a
crash or hang in generated code becomes a recorded result. `os.fork()` copies only the calling
thread, so `pause_openmp_pools` shuts down loaded OpenMP thread pools first.

## Runtime libraries

Every library and program links the runtimes it needs by name and never relies on the host process
having them loaded.

- **OpenMP.** LLVM libomp is the process's one runtime. It serves g++ code through its `GOMP_*` entry
  points and clang++ or icpx code through `__kmpc_*`, so every compiler shares one thread pool.
  `OpenMPRuntime.check` refuses a compiler that cannot link the runtime. The program link swaps
  DaCe's default runtime for libomp, for that compile only.
- **CUDA.** A GPU kernel links cudart from the nvcc that built it.

`ExternalCall` carries these link items, and the program links them after its objects.
