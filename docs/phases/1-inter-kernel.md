# Phase 1: Inter-kernel schedule

prev: [0 Normalize](0-normalize.md) · next: [2 Scope definition](2-scope-def.md)

Phase 1 decides how coarse the computation is: which maps and loops fuse and which split. The
scheduling agent does not edit graph nodes. It reads two views and requests moves.

- **Structure.** An indented text tree of control-flow regions, states and kernels, with each
  kernel's iteration domain and read/write sets. A kernel body can be rendered as NumPy, C++ or
  Fortran.
- **Cost.** Symbolic work and depth per scope, and a symbolic operational intensity (OI). OI divides
  work by the bytes a scope moves, under a simple cache model: a map (parallel region) caches
  perfectly, a loop caches nothing.

Moves are single-pair DaCe transformations, each legality-checked before it applies: vertical and
horizontal map fusion, loop fusion, state fusion (to merge the region two nests sit in), and fission
down to statements.

| | |
|---|---|
| default | `full_fusion(sdfg, targets)`: canonicalization's `fuse` stage and the stages after it |
| hand-chosen granularity | apply moves, then `finish_schedule(sdfg, targets)` |
| code | `nestforge/phases/schedule.py`, `nestforge/ir/introspect.py` |
| status | moves and default built; work/depth and OI views planned |
