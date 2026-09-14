# Phase 0: Normalize

[Overview](../../README.md) · next: [1 Shape Kernels](1-shape-kernels.md)

Normalization runs once on every program and is never searched. The CPU target is always on; the GPU
target is opt-in. DaCe canonicalization runs with the target's preset up to, but excluding, the `fuse`
stage. The result is the canonical parallel form (CPF): parallel loops are maps, reductions and scans
are lifted, and statements are distributed.

The early `coalesce` stage already fuses simple map chains. The `fuse` stage and everything after it
belong to phase 1, which keeps the fusion decision open.

| | |
|---|---|
| input | SDFG from the Python or Fortran frontend |
| output | CPF SDFG, not yet fused |
| default | `normalize(sdfg, Targets(gpu=...))` |
| stages | `dace.transformation.passes.canonicalize.stage_labels(target)` |
| code | `nestforge/phases/normalize.py` |
