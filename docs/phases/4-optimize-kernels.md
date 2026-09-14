# Phase 4: Optimize Kernels

prev: [3 Offload](3-offload.md) · next: [5 Sweep Configurations](5-sweep-configurations.md)

Phase 4 produces each kernel's implementation, one kernel at a time. Every kernel ends up as a static
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
| output | kernel source for phase 5, or a finished `lib<kernel>.a` |
| code | `nestforge/phases/kernel.py`, `nestforge/build/sdfg.py` |
| status | planned |
