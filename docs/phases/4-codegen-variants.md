# Phase 4: Codegen variants

prev: [3 Kernel optimization](3-kernel-opt.md) · feedback: [(e), (g)](feedback.md)

Phase 4 compiles each kernel's phase-3 source into variants and keeps the fastest one that matches
the NumPy oracle. This is the variant search from the Vectra paper, applied per kernel.

- **CPU axes.** Compiler (gcc, clang, icx, nvc) x FP mode x vectorizer cost model x vector math
  library, over the DaCe-generated C++.
- **GPU axes.** CUDA toolchains (nvcc, clang) x a few flag sets.
- Variants that compile to the same object are timed once.
- Every variant runs in a forked child, so a crash is a recorded result.

The winners link into the parent program as static archives, giving one binary with one OpenMP
runtime.

| | |
|---|---|
| default | brute force over all axes the toolchain supports |
| code | `nestforge/phases/variants.py`, `nestforge/build/arena.py`, `nestforge/build/flags.py` |
| status | CPU sweep exists over translator C; moving to DaCe C++ and CUDA |
