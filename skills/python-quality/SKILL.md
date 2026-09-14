---
name: python-quality
description: >-
  Quality-check workflow for a Python file (3.10+) and the modern-Python
  writing rules that matter for kernels and small tools -- type hints,
  explicit conversions, numpy rank rules, and typed caching. Use whenever
  checking, reviewing, or writing Python: "check this Python", "lint this",
  "type-check this", "is this clean", or "write modern Python".
---

# python-quality

Two jobs: (A) run a Python file through the check gates; (B) write new Python
that follows the rules that actually matter for a kernel or a small tool.

## Golden rule

All gates run. Warnings are errors. Type errors are errors. Fix a finding at
the source -- never silence it with `# type: ignore` or `# noqa` to get a
gate to pass (a narrow `# noqa: CODE` with a stated reason is acceptable only
for a genuine third-party false positive).

## A. The gates

### 1. Format -- yapf or the repo's own formatter
Use the formatter the repository already configures (a `.style.yapf`,
`pyproject.toml [tool.yapf]`, or an equivalent section for another
formatter); do not introduce a different one into a project that already
picked one. To check without editing:
```bash
yapf --diff --style='{based_on_style: pep8, column_limit: 120}' <file>.py
```
A non-empty diff is a failing gate.

### 2. Lint -- ruff
```bash
ruff check --line-length 120 <file>.py
```
Pass the project's own line length if it configures one; ruff defaults to 88,
which disagrees with a 120-column formatter and produces a wall of false
`E501` hits.

### 3. Type check
```bash
pyright <file>.py        # or: mypy --strict <file>.py
```
Run whichever the project configures (`pyrightconfig.json`,
`[tool.pyright]`, or a mypy CI step). Treat every type error as a failure.

### 4. Warnings-as-errors import/compile smoke
```bash
python -W error -m py_compile <file>.py     # syntax warnings, no execution
python -W error -c "import package.module"  # import-time warnings, fatal
```
Catches a `DeprecationWarning`, `SyntaxWarning`, or `ResourceWarning` that a
normal import would let through silently.

### 5. Tests -- run the file's pytest consumers
```bash
pytest --maxfail=10 path/to/test_<thing>.py
```
Run from the repository root so package-qualified imports resolve. A new
warning during the run is also a failure.

Report each gate's status; only call the file clean once every gate above
passes with zero output.

## B. Writing rules that matter for kernels and small tools

- **Type hints, always.** Every function signature -- every parameter and the
  return -- plus every non-trivial local. Modern 3.10+ syntax: `X | None`
  (not `Optional[X]`), `list[int]` / `dict[str, int]` / `tuple[int, ...]`
  (not `typing.List`/`Dict`/`Tuple`). Reach into `typing` only for what has no
  builtin spelling (`Callable`, `Protocol`, `TypeVar`, `Iterable`, `Self`,
  `Literal`).

- **Convert explicitly, never rely on an implicit coercion.** Wrap with
  `int()` / `float()` / `str()` / `bool()` at the point a value's type
  changes. Use `//` for integer division instead of `int(a / b)`. Do not use
  `bool` and `int` interchangeably, and prefer an explicit comparison
  (`if n != 0:`, `if x is not None:`) over bare truthiness when the intent is
  a specific check rather than "is this falsy".

- **numpy indexing changes rank -- three distinct rules.**
  - `a[0]` -- integer index -- **drops** the axis: `a[0:N, 0]` is `(N,)`.
  - `a[0:1]` -- slice -- **keeps** the axis, at length 1: `a[0:N, 0:1]` is
    `(N, 1)`.
  - `a[None]` / `a[np.newaxis]` -- **inserts** a length-1 axis, equivalent to
    a `0:1` slice on an axis the source did not have.

  These are not interchangeable: a length-1 axis broadcasts (every position
  reads the same element), while a dropped axis lines up against a different
  axis of the other operand. `a[:, 0:1] + b` and `a[:, 0] + b` compute
  different things. Never collapse a size-1 dimension as a shortcut --
  `squeeze()` is a deliberate operation on a known-safe axis, not a general
  shape simplification.

- **No `getattr` / `hasattr` for control flow.** An attribute's presence
  should be known statically. Use whichever of these actually fits:
  - a fixed, declared schema -> direct attribute access, with a sentinel
    value to mark "unset" rather than the attribute being absent;
  - a type or capability check -> `isinstance(x, np.ndarray)` then use its
    interface directly, not `hasattr(x, "shape")`;
  - a genuinely dynamic object with a real `__dict__` (for example an AST
    node with optional fields) -> `vars(obj).get("name", default)`.
  `vars()` only sees the instance `__dict__`, so it is unsafe for class
  attributes, properties, or `__slots__`-based objects -- it is the answer
  for a genuinely dynamic namespace, not a general substitute for `hasattr`.

- **`functools.lru_cache(maxsize=..., typed=True)` -- always `typed=True`.**
  Never bare `@lru_cache`, and never `@functools.cache` (it is
  `lru_cache(maxsize=None)` with no way to ask for `typed`). Untyped caching
  collapses `1`, `1.0`, and `True` onto one key -- along with a numpy scalar
  next to a Python number of the same value -- which silently returns a
  value computed for the wrong type wherever a cache is keyed on type-
  sensitive input. Cache only a pure function: no side effects, and no
  result that changes on its own between calls with the same arguments.
