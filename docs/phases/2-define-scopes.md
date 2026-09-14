# Phase 2: Define Scopes

prev: [1 Shape Kernels](1-shape-kernels.md) · next: [3 Offload](3-offload.md)

Phase 2 decides which scopes leave the program as external kernels. Each scope is outlined into a
standalone SDFG and replaced by an `ExternalCall` library node that carries the kernel's NumPy
reference, manifest and boundary. Until phase 4 binds a compiled library, the node runs through
DaCe's reference expansion.

The default makes one kernel per parallel top-level map, wherever the map sits in the control flow,
time loops included. A purely sequential nest yields no kernel.

Scalar inputs cross the boundary by value. Lowering refuses a host length-1 array input; only a
length-1 GPU array, a device pointer, may stand in for a scalar.

| | |
|---|---|
| default | `lower_nests_to_external_call(sdfg)` |
| preview | `offload_candidates(sdfg)` lists the parallel top-level maps without mutating |
| code | `nestforge/phases/scopes.py`, `nestforge/ir/extract.py`, `nestforge/ir/libnode.py` |

The [placement analysis](feedback.md) can send the program back here when kernels need different
boundaries.
