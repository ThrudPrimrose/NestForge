---
name: dace
description: >-
  Domain knowledge for reading an SDFG or requesting a transformation on one --
  the graph model, memlets and write-conflict resolution, scope symbols, the
  numpy-style rank rules a slice or index changes, what map/loop fusion and
  fission require to stay legal, and how to build a small Python-frontend
  reproducer. Use whenever inspecting, fusing, fissioning, or otherwise
  rewriting a DaCe SDFG.
---

# dace

An SDFG is a dataflow-and-control-flow intermediate representation. This page
covers the model an agent needs to read one and judge whether a graph move is
legal, not how to contribute to the DaCe project itself.

## The SDFG model

An SDFG is a control-flow region: a directed graph of control-flow blocks
connected by interstate edges, each carrying a condition and a set of symbol
assignments. A block is one of:

- an **SDFGState** -- a dataflow multigraph of nodes connected by memlet
  edges with named connectors;
- a nested **ControlFlowRegion**, **LoopRegion** (init / condition / update,
  like a `for` or `while`), or **ConditionalBlock** (an ordered list of
  `(condition, region)` branches, `if`/`elif`/`else`).

The hierarchy is recursive: regions nest inside regions, and a **NestedSDFG**
node inside a state holds an entire child SDFG. A method that walks only the
top region's blocks misses everything inside a loop body or a branch; walking
"all control-flow regions" is what reaches those.

A **library node** stands for a coarse operation (a reduction, a BLAS call, a
scan) and expands into real dataflow later. It can carry several expansions --
a portable "pure" one plus one or more device- or instruction-set-specific
ones. They can legitimately expose different connectors, so a check against
the pure expansion says nothing about whether an ISA-specific expansion is
correct; verify the expansion that will actually run.

## Maps are data-parallel scopes

A map is data-parallel by definition: every iteration must be independent of
every other. A cross-iteration dependence inside a map is a data race, not a
detail to be validated away -- that computation belongs in a sequential loop
(a `LoopRegion`, or a map scheduled `Sequential`), not inside a parallel map.
A schedule type (`Sequential`, `CPU_Multicore`, `GPU_Device`, ...) says how a
map is executed; a storage type (`CPU_Heap`, `CPU_ThreadLocal`, `GPU_Global`,
`GPU_Shared`, `Register`) says where its data lives. Neither changes the
data-parallel contract above.

## Memlets: data edges and ordering edges

A memlet is an edge annotated with the data it moves: which array, which
subset, and (for a write) whether it accumulates. Two things about memlets
that are easy to get backward:

- **An empty memlet carries no data.** It is a pure ordering edge -- happens
  before, nothing more -- used where one node must run before another with no
  value passing between them (a tasklet before a scope, a control-flow
  ordering constraint). Do not treat one as degenerate or interchangeable
  with a normal data edge; removing it can leave two nodes free to execute in
  either order.
- **A write-conflict-resolution (WCR) memlet is a read-modify-write to the
  destination**, not a plain write: the runtime effectively does
  `*dst = wcr(*dst, value)`. A pass that treats a WCR edge as an ordinary
  store silently drops the accumulation it depends on -- and because it reads
  before it writes, WCR is where an accumulator is threaded through a
  transformation, not a place that can be paved over with a plain copy.

## Scope symbols are not SDFG symbols

A loop's iteration variable, a map's parameters, and a map entry's dynamic
input connectors (any connector that is not a plain `IN_*`/`OUT_*`
pass-through) are defined by their enclosing scope, not by the SDFG. They are
never in the SDFG's symbol table, and adding one there directly shadows the
real, scoped definition and injects a spurious free symbol into the SDFG's
call signature. Getting a scope symbol's type means walking the chain of
enclosing scopes (loops and maps) from the outside in and accumulating what
each one defines -- there is no shortcut that reads it off the SDFG.

## Index versus slice: rank rules

Indexing and slicing look similar and do different things to shape, exactly
where a length-1 case makes them visibly differ:

- an integer index **drops** the axis: `a[0:N, 0]` has shape `(N,)`;
- a slice **keeps** the axis, even at length 1: `a[0:N, 0:1]` has shape
  `(N, 1)`.

A size-1 axis from a slice broadcasts (every position along it reads the same
element); a dropped axis instead lines up against a different axis of the
other operand. Conflating the two is a silent shape bug, not a cosmetic
difference -- `a[:, 0:1] + b` and `a[:, 0] + b` compute different things.
A generic "squeeze all size-1 dimensions" operation cannot tell a slice
singleton from an index singleton apart, so it is only safe where that
provenance was tracked separately; applying it blindly to a subset can
silently collapse a real axis of a matrix into a vector of the wrong values.

## What map/loop fusion and fission require

**Map fusion** merges two adjacent maps into one. It is legal only when:

- the two maps iterate the **same range** (or one can be permuted/sliced to
  match the other exactly) -- fusing maps with different bounds changes which
  outputs exist;
- the schedule types agree (fusing a sequential map into a parallel one, or
  vice versa, changes execution semantics, not just structure);
- any intermediate array produced by the first map and consumed by the second
  has **no other reader that depends on it existing as a separate, fully
  materialized array** between the two maps -- fusing turns it into a
  per-iteration value (or a small local buffer), so an outside reader of the
  old full array breaks;
- no dependency is introduced that would force one iteration of the fused map
  to see another iteration's result -- since a map is data-parallel, this
  usually holds automatically, but a WCR edge crossing the fusion boundary
  must still resolve to the same accumulation, not a different one.

**Map fission** is the reverse: splitting one map into several. It is legal
whenever the body actually decomposes into independent pieces (which it does
for a data-parallel map, definitionally) -- the caveat is any transient
declared inside the original map body: fission usually needs a copy or a
redeclaration of that transient per resulting map, since a single shared
buffer can no longer be reused across maps that may run separately.

**Loop fusion** merges two sequential loop regions into one. Unlike maps,
loop bodies may depend on iteration order, so fusion additionally requires:

- the same trip count and the same iteration step;
- no loop-carried dependence that fusing would reorder -- a write at
  iteration *i* of the first loop that a later iteration of the second loop
  reads must still see that write happen at the same relative point once the
  two bodies interleave; if fusing changes that relative order, the loops are
  not fusable without changing the computed values.

## Building a small reproducer with the Python frontend

The cheapest way to check a claim about the graph model is to build the
smallest program that exhibits it and look at the SDFG directly:

```python
import dace, numpy as np

N = dace.symbol('N', dtype=dace.int64)

@dace.program
def k(a: dace.float64[N], out: dace.float64[N]):
    out[:] = a * 2.0

sdfg = k.to_sdfg(simplify=False)   # parse only -- no compiler invoked
sdfg.validate()                    # catches most hand-built-graph mistakes
```

`simplify=False` is the right default for a reproducer: simplification can
fuse away the very state or memlet a question is about. Turn it on
separately to check whether the behavior survives it.

To look at the result:

```python
sdfg.save('k.sdfg', readable=True)   # JSON, diffable; open in an SDFG viewer
sdfg.view()                          # opens it directly
print(sdfg.to_json()[:2000])         # quick text inspection
```

For explicit dataflow -- when the exact tasklet and memlet shape matters more
than what the parser infers -- write it directly:

```python
for i in dace.map[0:N]:
    with dace.tasklet:
        inp << a[i]
        res >> out[i]
        res = inp * 3.0
```

A schedule hint on a map uses the `@` operator, the only binary operation
allowed in a `for` iterator: `for i in dace.map[0:N] @ dace.ScheduleType.Sequential`.
A storage hint on a local uses an annotated assignment whose annotation is a
data descriptor: `buf: dace.data.Array(dace.float64, (N,), storage=dace.StorageType.CPU_ThreadLocal) = np.zeros((N,))`.
