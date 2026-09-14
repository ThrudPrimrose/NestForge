---
name: python-to-numpy
description: >-
  Rules for writing a NumPy kernel a numpy-to-native translator can lower --
  buffer-out signature, explicit shapes, and the index and rank rules a
  translator needs to size and schedule an array op. Use when writing or
  cleaning up a NumPy kernel meant to compile, not just run.
---

# python-to-numpy

A NumPy kernel meant for translation to native code follows a narrower
surface than NumPy in general. The goal is a kernel whose shapes, buffers,
and index expressions a translator can resolve at compile time.

## Two kinds of values: an ndarray and a scalar

Nothing else. No Python `list`, `dict`, or `set` holding data, no
`.append`/`.extend`, no `dataclass` or `namedtuple`, no tuple used as a
container of values, and no array whose size is known only after a loop has
run. There is nothing to lower a general Python data structure into, and
nothing to lower a data structure that grows into. Pre-declare the
worst-case buffer and fill it by index, tracking a count separately if one is
needed -- never build a result by appending and converting at the end.

A tuple that spells a *shape* or an `axis=` argument is not data: `np.zeros((n,
c, d))`, `np.sum(x, axis=(2, 3, 4))` are static literals a translator reads at
compile time, not a container carrying runtime values.

## Buffer-out signature

The kernel's entry point writes its result into an output array argument and
returns nothing. The caller allocates the output; the kernel never allocates
or returns it. This is what lets a translator emit a plain function with no
return-value marshaling:

```python
def kernel(a, b, out):
    out[:] = a + b
```

A helper that builds and returns a fresh array per call, instead of writing
into a buffer it was given, is the pattern to remove.

## Explicit shapes, not `.shape`

Take every size the computation depends on as an explicit parameter (or a
symbol the caller binds), and derive further shapes from those parameters --
do not re-derive a size from `.shape` when the caller already knows it and
could have passed it in:

```python
# avoid -- size re-derived from the buffer
def pool(x, out):
    pad = x.shape[2] + 2 * padding

# prefer -- the size is a parameter
def pool(x, depth, out):
    pad = depth + 2 * padding
```

`.shape` stays fine for reading a rank, or a shape that was never a declared
parameter to begin with. An extent read back off a value that came from a
computed expression, a conditional, or a `newaxis` view is not a real extent
either -- name the count explicitly (`count = 2 * span + 1`) instead of
reading it off a shape a translator cannot trace.

## One name, one shape

Rebinding a name to an array of a different shape invalidates everything a
translator tracked about it. Give a differently-shaped result a new name, or,
when a buffer of the right shape already exists, write into it in place:

```python
# avoid -- one name, two shapes
if tail:
    padded = np.pad(padded, pad_width)

# prefer -- one name per shape
padded_full = np.pad(padded, pad_width)
```

```python
# prefer, when the target buffer already has the right shape
out[:] = compute(...)          # not out = compute(...)
```

## A view that is later written is not a value

`b = out[:s, :s]` followed by a write through `b` aliases the live array
`out`; index the parent directly at the write site instead of holding a slice
alias across a write. The same aliasing trap applies to `a = b = c =
np.zeros(...)`: this binds three names to one buffer, so a write through any
one of them changes all three. Allocate one buffer per name.

## Index and rank rules

- An integer index **drops** an axis; a slice **keeps** it, even at length
  one: `a[0:N, 0]` is `(N,)`, `a[0:N, 0:1]` is `(N, 1)`. Never treat a size-1
  slice and an integer index as interchangeable -- they broadcast
  differently.
- A strided slice as a write **target** needs a compile-time-constant,
  positive step (`out[0::2] = a`, `out[1:10:3] += a`). A negative step on a
  target is not resolvable without knowing the array's length at compile
  time; reverse the source instead of the target, or index explicitly.
- Spell a strided read's stop as `start + count * stride`, not
  `start + (count - 1) * stride + 1`. Both select the same elements, but the
  first folds to exactly `count` regardless of what `stride` turns out to
  be; the second leaves a `ceiling` term a translator cannot always resolve
  when `stride` is a runtime value.
- **Scatter with a repeated index needs an unbuffered accumulate, not a
  buffered `+=`.** `Lx[idx] += flux` on an index array with repeats loses all
  but one contribution per repeated index (fancy-index augmented assignment
  buffers, so a repeated target lands once); `np.add.at(Lx, idx, flux)` (or
  `.subtract.at`, `.multiply.at`, `.maximum.at`) accumulates every one.
- A **gather** -- reading through an index array (`q[neigh[:, j]]`) -- is
  already a plain array operation and needs none of the scatter caution
  above.
- Reverse with an explicit index array (`rev = np.arange(n - 1, -1, -1)`,
  then `a[rev]`) rather than a negative-step slice behind an ellipsis, where
  the rank or axis is not immediately visible at the call site.
- Pick between two arrays with `np.where(cond, a, b)`, not a Python
  conditional expression (`a if cond else b`): the conditional expression
  makes the result's shape undecidable at compile time, while `np.where`
  keeps it fixed.

## Loop the window, not the output

A common anti-pattern is a Python loop over every output element, calling
NumPy once per element on a small window. Invert it: loop over the window's
taps (a handful of iterations, one per weight in a stencil or kernel), and
let each iteration's NumPy call cover the whole batch and output volume at
once:

```python
# avoid -- one np.max call per output element
for oz in range(out_d):
    for oy in range(out_h):
        out[oz, oy] = np.max(padded[oz:oz + k, oy:oy + k])

# prefer -- one np.maximum call per tap, covering every output element
out[:, :] = -np.inf
for kz in range(k):
    for ky in range(k):
        out[:, :] = np.maximum(out[:, :], padded[kz:kz + out_d, ky:ky + out_h])
```

## A sequential loop is not automatically the end of the discussion

- **Sequential in one axis, independent in the rest** -- the common case.
  Keep the dependent axis as a loop and make every other axis a full-width
  slice.
- **The recurrence is a scan or a reduction** (`np.cumsum`, `np.cumprod`,
  `np.sum`, an accumulating maximum). Reassociating the additions is allowed
  for these, so the vectorized form may differ from the scalar form in its
  last few bits; check it with a tolerance, not exact equality.
- **A genuine element-to-element recurrence with no closed form** stays a
  loop.

## Exactness

A rewrite from a scalar loop to array ops should produce a bit-identical
result unless it is a scan or reduction (above), in which case reassociation
may move the last few bits. A bigger discrepancy -- a wrong axis, a dropped
guard, a flipped constant -- is a wrong port, not a tolerance to raise.
