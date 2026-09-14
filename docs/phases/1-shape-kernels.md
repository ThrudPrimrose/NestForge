# Phase 1: Shape Kernels

prev: [0 Normalize](0-normalize.md) · next: [2 Define Scopes](2-define-scopes.md)

Phase 1 decides which maps and loops fuse and which split. The scheduling agent never edits graph
nodes; it reads two views and requests moves.

- **Structure.** `Session.describe()` prints control-flow regions, states and nests with their
  iteration domains and read/write sets. `Session.kernel_source` renders a nest as NumPy, C++ or
  Fortran.
- **Cost.** `describe(metrics=True)` adds symbolic work, depth and operational intensity (OI) per
  scope. OI divides work by the bytes a scope moves under a simple cache model: a map caches
  perfectly, a loop caches nothing.

A move is one existing DaCe transformation, legal only when that transformation's `can_be_applied`
accepts it. `list_moves(kind)` returns the legal moves as `{kind, labels, epoch}`. `apply_move(kind,
labels, epoch)` applies one and returns a `MoveResult` whose status is `applied`, `illegal`,
`not-implemented`, `not-found` or `stale`. Labels are the tree labels `describe()` prints, and its
first line shows the epoch.

| kind | labels | DaCe |
|---|---|---|
| `loop-fusion` | first, second loop | `LoopFusion` |
| `map-fusion` | two maps | `MapFusionVertical` through an intermediate, else `MapFusionHorizontal` |
| `map-fission` | map | `MapFission` on a map whose body is one nested SDFG |
| `interchange-map-map` | outer, inner map | `MapInterchange` |
| `interchange-loop-map` | loop, its one map | `MoveLoopIntoMap`: the map becomes outer |
| `loop-fission`, `interchange-loop-loop`, `interchange-map-loop` | | not implemented |

`fission_all` splits the whole program to statement granularity, loops included. The id-based calls
stay: `list_fusions` / `fuse`, `list_fissions` / `fission` and state fusion via `fuse_regions`.

| | |
|---|---|
| default | `full_fusion(sdfg, targets)`: canonicalization's `fuse` stage and the stages after it |
| hand-chosen | apply moves, then `finish_schedule(sdfg, targets)` |
| code | `nestforge/phases/schedule.py`, `nestforge/ir/introspect.py` |
