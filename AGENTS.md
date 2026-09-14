# AGENTS.md

Guide for LLM agents that drive NestForge and for coding agents that change it. Phase details live in
[docs/phases](docs/phases/); this page says who does what and which rules hold everywhere.

## Agent roles

| Agent | Phases | Reads | Requests |
|---|---|---|---|
| scheduling | [1](docs/phases/1-shape-kernels.md), [2](docs/phases/2-define-scopes.md), [3](docs/phases/3-offload.md) | structure tree, kernel bodies, work/depth, OI | fusion/fission moves, kernel-to-device schedule |
| kernel | [4](docs/phases/4-optimize-kernels.md) | one kernel as NumPy, C++ or Fortran, its boundary | a source file or `lib<kernel>.a` with the given C entry |
| analysis | [feedback](docs/phases/feedback.md) | placement, copy volume, per-kernel times, OI | a phase-1 move, or stop |

Every phase has a deterministic default, so a run works with any subset of agents.

## Rules for driving agents

- Never edit SDFG nodes or memlets. Request moves through the NestForge API; each move is checked
  for legality before it applies.
- Ids go stale after any mutation. List again before the next move.
- A result counts only after it matches the kernel's NumPy oracle. Wrong and fast loses.
- A kernel library keeps the C entry and argument order it was given; a mismatched order corrupts
  the call silently.

## Skills

Skills load from HPCAgent-Bench (a dependency) plus this repository's `skills/`:

```python
from pathlib import Path
from hpcagent_bench.harness.prompts import load_skills

skills = load_skills([str(Path("path/to/nest-forge"))])
```

This repository adds pruned `dace`, `python-quality` and `python-to-numpy` skills. Kernel agents
also use the bench language skills (`lang-cpp`, `lang-cuda`, `lang-fortran`, `lang-python`),
`canonical-parallel-form`, `profiling` and `opt-reports`.

## Rules for coding agents

- Setup: `uv sync --extra dev`. Unit tests: `uv run pytest -m "not integration and not gpu"`.
- Format with yapf (120 columns), lint with ruff. Format only the files you touched.
- Python is written as if statically typed: annotate every function, one name keeps one type, no
  `getattr`/`hasattr`, no leading-underscore names, absolute imports.
- Keep cyclomatic complexity per function at 20 or below (`radon cc -s`).
- Comments and docstrings stay short: at most about one comment line per five code lines, and
  docstrings with `:param:` only on public functions.
- Iterate `OrderedSet` or `dict`, not a plain `set`, where order can reach generated code.
- Tests assert structure as well as values; a failing test gets fixed, not weakened or deleted.
- Docs stay brief, one page per concept, linked from the README.
