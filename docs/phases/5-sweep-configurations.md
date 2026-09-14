# Phase 5: Sweep Configurations

prev: [4 Optimize Kernels](4-optimize-kernels.md) · feedback: [Analyze](feedback.md)

Phase 5 compiles each kernel's phase-4 source into variants and keeps the fastest one that matches
the NumPy oracle. This is the variant search from the Vectra paper, applied per kernel.

- **CPU axes.** Compiler (gcc, clang, icx) x FP mode x vectorizer cost model, over the
  DaCe-generated C++.
- **GPU axes.** CUDA toolchains (nvcc, clang) x a few flag sets.
- Variants that compile to the same object are timed once.
- Every variant runs in a forked child, so a crash is a recorded result.

The winners link into the parent program as static archives, giving one binary with one OpenMP
runtime.

| | |
|---|---|
| default | brute force over all axes the toolchain supports |
| session | `sweep_configurations(kernel_id, sizes, reps, compilers)` links the winner |
| code | `nestforge/phases/variants.py`, `nestforge/build/arena.py`, `nestforge/build/flags.py` |
| status | CPU sweep over DaCe C++ implemented (`enumerate_variants`, `select_variant`); CUDA pending |
