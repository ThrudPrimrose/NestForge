# Phase 0: Normalize

[Overview](../../README.md) · next: [1 Shape Kernels](1-shape-kernels.md)

Normalization runs on every program and is not searched. It picks the targets first: the CPU is always on, the
GPU is opt-in. It then runs DaCe canonicalization with that target's preset, up to but excluding the
`fuse` stage. The result is the canonical parallel form (CPF): loops that can run in parallel are
maps, reductions and scans are lifted, and statements are distributed.

Canonicalization's early `coalesce` stage already fuses simple map chains; that stays here. The
`fuse` stage and everything after it belong to phase 1, so the fusion decision stays open.

| | |
|---|---|
| input | SDFG from the Python or Fortran frontend |
| output | CPF SDFG, not yet fused |
| default | `normalize(sdfg, Targets(gpu=...))` |
| code | `nestforge/phases/normalize.py` |

Stage list: `dace.transformation.passes.canonicalize.stage_labels(target)`.
