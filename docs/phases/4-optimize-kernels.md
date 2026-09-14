# Phase 4: Optimize Kernels

prev: [3 Offload](3-offload.md) · next: [5 Sweep Configurations](5-sweep-configurations.md)

Phase 4 builds each kernel into a static library `lib<kernel>.a` with one `extern "C"` entry, which
the program links. The kernel's NumPy reference is the oracle for every implementation.

- **Default.** DaCe's `cpf.render` writes the kernel as one standalone CPF unit with no DaCe header
  or runtime: C++ with OpenMP on CPU, CUDA on GPU. Before rendering, the kernel copy is finalized for
  its device and its integer symbols widen to `int64_t`, so the entry takes exactly what
  `ExternalCall` passes: arrays by pointer, read-only scalars by value. A GPU kernel is offloaded
  whole, and its entry takes the device pointers that phase 3's copies fill. The compiler vectorizes;
  phase 5 sweeps its cost models.
- **Kernel agent.** Receives the kernel as NumPy, C++ or Fortran plus its boundary, and returns source
  or a library with the same entry. HPCAgent-Bench runs the agent.

| | |
|---|---|
| default | `finalize_for_target(device)`, then `cpf.render(kernel, language="c++" or "cuda")` |
| session | `optimize_kernel(kernel_id)`, `set_kernel(kernel_id, lib_path, symbol, abi_order)` |
| output | the kernel's CPF unit, or a finished `lib<kernel>.a` |
| code | `nestforge/phases/kernel.py`, `nestforge/build/sdfg.py` |
