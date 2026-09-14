# Phase 3: Kernel optimization

prev: [2.5 Offloading](2.5-offload.md) · next: [4 Codegen variants](4-codegen-variants.md)

Phase 3 produces each kernel's implementation, one kernel at a time. Every kernel ends up as a static
library `lib<kernel>.a` with a single `extern "C"` entry, which the parent program links. The
kernel's NumPy reference is the correctness oracle for every implementation.

- **Default (DaCe).** CPU kernels get DaCe's vectorizer with one default configuration, then
  `finalize_for_target`. GPU kernels are offloaded and finalized for the GPU. DaCe's generated
  program gets a small generated wrapper (init, run, exit) behind the C entry.
- **Kernel agent.** The agent receives the kernel as NumPy, C++ or Fortran plus its boundary, and
  returns source or a library that exposes the same entry. HPCAgent-Bench runs the agent.

| | |
|---|---|
| default | DaCe vectorizer + `finalize_for_target(device)` |
| output | kernel source for phase 4, or a finished `lib<kernel>.a` |
| code | `nestforge/phases/kernel.py`, `nestforge/build/sdfg.py` |
| status | planned |
