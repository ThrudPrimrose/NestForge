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

Moves are single-pair DaCe transformations, each checked for legality before it applies: vertical
and horizontal map fusion, loop fusion, state fusion, and fission of one map's independent output
groups (`list_fissions` / `fission`). `fission_all` splits the whole program to statement
granularity.

| | |
|---|---|
| default | `full_fusion(sdfg, targets)`: canonicalization's `fuse` stage and the stages after it |
| hand-chosen | apply moves, then `finish_schedule(sdfg, targets)` |
| code | `nestforge/phases/schedule.py`, `nestforge/ir/introspect.py` |
