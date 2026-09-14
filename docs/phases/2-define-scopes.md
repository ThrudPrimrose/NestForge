# Phase 2: Define Scopes

prev: [1 Shape Kernels](1-shape-kernels.md) · next: [3 Offload](3-offload.md)

Phase 2 decides which scopes leave the program as external kernels. Each chosen scope is outlined
into a standalone SDFG and replaced by an `ExternalCall` library node. The node carries the kernel's
NumPy reference, its manifest and its boundary, and it still runs through the DaCe reference
expansion until phase 4 gives it a compiled library.

The default kernel granularity is one scope per parallel top-level map: every parallel map that is
not nested inside another map becomes its own kernel, wherever it sits in the control flow
(including inside a time loop). A nest with no parallel top-level map, such as a purely sequential
loop, yields no scope.

| | |
|---|---|
| default | `lower_nests_to_external_call(sdfg)` |
| preview | `offload_candidates(sdfg)` lists the parallel top-level maps, without mutating |
| code | `nestforge/phases/scopes.py`, `nestforge/ir/extract.py`, `nestforge/ir/libnode.py` |

[Offloading](3-offload.md) can send the program back here when a placement needs different scopes.
