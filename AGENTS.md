# AGENTS.md

Guide for LLM agents that drive NestForge and for coding agents that change it. Phase details live in
[docs/phases](docs/phases/).

## Agent roles

| Agent | Phases | Reads | Requests |
|---|---|---|---|
| scheduling | [1](docs/phases/1-shape-kernels.md), [2](docs/phases/2-define-scopes.md), [3](docs/phases/3-offload.md) | structure tree, kernel bodies, work/depth, OI | fusion/fission moves, kernel-to-device schedule |
| kernel | [4](docs/phases/4-optimize-kernels.md) | one kernel as NumPy, C++ or Fortran, its boundary | a source file or `lib<kernel>.a` with the given C entry |
| analysis | [feedback](docs/phases/feedback.md) | runtimes, placements, copy volume, OI | phase 1 moves, new phase 2 scopes, or stop |

Every phase has a deterministic default, so a run works with any subset of agents.

## Rules for driving agents

- Never edit SDFG nodes or memlets. Request moves through the Session API; each move is checked for
  legality before it applies.
- Name nests by the labels `Session.describe()` prints. They are unique across the whole program,
  nested SDFGs included. Pass them with the epoch from the tree's first line or from `list_moves()`:
  `apply_move(kind, labels, epoch)`.
- Labels and ids regenerate after every mutation. A move carrying an old epoch returns `stale`;
  describe or list again before the next move. Only an `applied` result changed the program.
- A result counts only after it matches the kernel's NumPy oracle. Wrong and fast loses.
- A kernel library keeps the C entry and argument order it was given; a different order corrupts the
  call silently.

## Skills

Skills load from HPCAgent-Bench plus this repository's `skills/` (pruned `dace`, `python-quality`,
`python-to-numpy`):

```python
from hpcagent_bench.harness.prompts import load_skills

skills = load_skills(["path/to/nest-forge"])
```

Kernel agents also use the bench skills `lang-cpp`, `lang-cuda`, `lang-fortran`, `lang-python`,
`canonical-parallel-form`, `profiling` and `opt-reports`.

## Rules for coding agents

- Setup: `pip install -e ".[dev]" && pre-commit install`. Unit tests:
  `pytest -m "not integration and not gpu and not vendor"`.
- `pre-commit run --all-files` runs ruff check and ruff format (120 columns), as CI does.
- Write Python as if statically typed: annotate every function, one name keeps one type, no
  `getattr`/`hasattr`, no leading-underscore names, absolute imports.
- Keep each function's cyclomatic complexity at 20 or below (`radon cc -s`).
- Keep comments short (about one line per five lines of code); docstrings with `:param:` only on
  public functions.
- Iterate `OrderedSet` or `dict`, never a plain `set`, where order can reach generated code.
- Tests assert structure as well as values; a failing test gets fixed, never weakened or deleted.
- Docs stay brief, one page per concept, linked from the README.
