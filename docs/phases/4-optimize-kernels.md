# Phase 4: Optimize Kernels

prev: [3 Offload](3-offload.md) · next: [5 Sweep Configurations](5-sweep-configurations.md)

Phase 4 produces each kernel's implementation, one kernel at a time. Every kernel ends up as a static
library `lib<kernel>.a` with a single `extern "C"` entry, which the parent program links. The
kernel's NumPy reference is the correctness oracle for every implementation.

- **Default (CPF).** DaCe's `cpf.render` writes the kernel as one standalone canonical parallel form
  unit. The unit needs no DaCe header or runtime and defines the kernel's C entry itself: C++ with
  OpenMP pragmas on CPU, CUDA on GPU. A GPU kernel is offloaded whole, so its entry takes device
  pointers and the program's phase 3 copies feed it. Before rendering, the kernel copy is
  finalized for its device and its integer symbols become `int64_t`, so the entry takes exactly what
  the `ExternalCall` prototype passes: arrays by pointer, a read-only scalar input by value. The DaCe
  vectorizer is not applied; the compiler vectorizes, and phase 5 sweeps its cost models.
- **Kernel agent.** The agent receives the kernel as NumPy, C++ or Fortran plus its boundary, and
  returns source or a library that exposes the same entry. HPCAgent-Bench runs the agent.

| | |
|---|---|
| default | `offload_to_gpu` on GPU, `finalize_for_target(device)`, `cpf.render(kernel, language="c++" or "cuda")` |
| session | `optimize_kernel(kernel_id)`, `set_kernel(kernel_id, lib_path, symbol, abi_order)` |
| output | the kernel's CPF unit, or a finished `lib<kernel>.a` |
| code | `nestforge/phases/kernel.py`, `nestforge/build/sdfg.py` |
| status | CPU and GPU implemented (`cpu_schedule`, `gpu_schedule`, `schedule_kernel`, `build_kernel_library`, `validate_kernel`) |
