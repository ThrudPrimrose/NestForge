# Build

[../README.md](../README.md) · related: [3 Kernel optimization](phases/3-kernel-opt.md) ·
[4 Codegen variants](phases/4-codegen-variants.md)

NestForge does not call `dace.compile()` for the arena: `nestforge/build/sdfg.py`,
`toolchain.py` and `arena.py` generate the SDFG's source, compile and link it with one chosen
compiler and flag set, and call the result directly. This keeps the DaCe-backend competitor and
the offloaded kernels on the same compiler and flags, so phase 4's timings compare codegen against
codegen rather than against `CompiledSDFG`'s own marshaling overhead.

## Owning the compile

`generate_program_folder` runs `codegen.generate_code(sdfg)` and writes the `CodeObject`s to disk
instead of letting DaCe build them. `compile` in `sdfg.py` then invokes the chosen compiler
directly, with DaCe's runtime headers (`dace_runtime_include`) on the include path.

## Manual init / run / exit

A DaCe-generated shared object exposes three C-linkage entry points for an SDFG named `N`:
`__dace_init_N`, `__program_N`, `__dace_exit_N`. `BuiltSDFG` (`sdfg.py`) binds all three through
`ctypes.CDLL` and calls them in that order: init allocates the SDFG's state and returns an opaque
handle, `__program_N` runs the kernel and is the only call phase 4 times, and exit frees the state.
`BuiltSDFG.unload` releases the `.so` mapping with `dlclose` when a build is discarded.

## Fork isolation

`nestforge/build/isolation.py` runs a freshly compiled kernel in a forked child
(`run_isolated`), so a segfault or a runaway loop in generated code cannot take down the process
driving the sweep. `os.fork()` duplicates only the calling thread, so a live OpenMP thread pool
across the fork deadlocks the child; `pause_openmp_pools` tears down every already-loaded
runtime's pool first. The parent enforces a wall-clock timeout and kills a child that does not
finish in time.

## Static archives and shared objects

`nestforge/build/arena.py` offers two ways to hand a winning kernel to a parent build:
`archive_objects` bundles one nest's objects into `lib<name>_nest.a`, and `link_shared` links them
into `lib<name>_nest.so` instead. A `.so` resolves symbols at `dlopen` time and survives DaCe
sorting the parent's link flags; a static archive does not, so several nests never share one
archive. `build_winner_archive` recompiles a winning cell's source with its chosen compiler and FP
mode into a fresh static archive for that purpose.

## One OpenMP runtime

`nestforge.build.toolchain.OpenMPRuntime` names the single OpenMP runtime a build links against
(default `libomp`, since it is LLVM-selectable and also implements the GOMP ABI, so a GCC-built and
a Clang-built object can share one thread pool). `OpenMPRuntime.check` raises before compiling a
translation unit against a runtime a given compiler cannot actually link (NVIDIA and classic Intel
hard-link their own runtime; LLVM selects by name; GNU accepts any GOMP-ABI runtime), which is how
a mixed-compiler build is kept off a mixed-runtime link.

## LTO for the archive path

Passing `lto=True` (`BuildOptions.lto` in `sdfg.py`) compiles a fat-LTO object (bitcode plus real
machine code) and archives it with the LTO-plugin-aware archiver (`gcc-ar`/`llvm-ar`), so a later
`-flto` link can still inline into it; a plain `ar` would drop the bitcode section. Fat-LTO is
available on GCC and Clang; other families fall back to a plain object with a warning.
