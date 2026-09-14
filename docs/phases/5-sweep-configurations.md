# Phase 5: Sweep Configurations

prev: [4 Optimize Kernels](4-optimize-kernels.md) · feedback: [Analyze](feedback.md)

Phase 5 compiles each kernel's phase 4 unit into variants, checks every variant against the NumPy
oracle, and keeps the fastest correct one.

- **CPU.** Compiler (GNU, LLVM, oneAPI) × FP mode (strict-ieee, contract-fma, fast-math) × vectorizer
  cost model.
- **GPU.** Every nvcc on PATH × FP mode (strict-ieee, contract-fma), all with `-arch=native`; no cost
  model.

Variants that compile to the same object are timed once. Each variant runs in a forked child, so a
crash is a recorded result. The winner's compiler, FP mode, cost model, flags and time form the nest's
JSON configuration, and the session links the winner into the program. [FP modes](../fp-and-vectorization.md)
lists the flags.

| | |
|---|---|
| default | brute force over every axis the toolchains support |
| session | `sweep_configurations(kernel_id, sizes, reps, compilers)` |
| code | `nestforge/phases/variants.py`, `nestforge/build/arena.py`, `nestforge/build/flags.py` |
