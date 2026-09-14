# Phase 2: Define Scopes

prev: [1 Shape Kernels](1-shape-kernels.md) · next: [3 Offload](3-offload.md)

Phase 2 decides which scopes leave the program as external kernels. Each chosen scope is outlined
into a standalone SDFG and replaced by an `ExternalCall` library node. The node carries the kernel's
NumPy reference, its manifest and its boundary, and it still runs through the DaCe reference
expansion until phase 4 gives it a compiled library.

The scope unit sets the kernel granularity: a single map, a whole state, or a control-flow region
(`map`, `state`, `cfg`). The default skips map wrappers that only schedule inner maps.

| | |
|---|---|
| default | `lower_nests_to_external_call(sdfg, "skip-taskloops")` |
| preview | `offload_candidates(sdfg, unit)` lists what a unit would extract, without mutating |
| code | `nestforge/phases/scopes.py`, `nestforge/ir/extract.py`, `nestforge/ir/libnode.py` |

[Offloading](3-offload.md) can send the program back here when a placement needs different scopes.
