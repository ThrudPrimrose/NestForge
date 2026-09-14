# NestForge

NestForge optimizes a whole DaCe program in phases. Each phase makes one decision, ships a
deterministic default, and exposes the same API to a scripted optimizer, a human, and an LLM agent.

[![NestForge phases](docs/figures/pipeline.png)](docs/figures/pipeline.svg)

| Phase | Decides | Deterministic default | Agent |
|---|---|---|---|
| [0 Normalize](docs/phases/0-normalize.md) | canonical parallel form for the enabled targets | canonicalize up to fusion | none |
| [1 Shape Kernels](docs/phases/1-shape-kernels.md) | fusion and fission granularity | full fusion | scheduling |
| [2 Define Scopes](docs/phases/2-define-scopes.md) | which scopes become external kernels | top-level compute nests | scheduling |
| [3 Offload](docs/phases/3-offload.md) | device per kernel, host/device copies | DaCe GPU offloading | scheduling |
| [4 Optimize Kernels](docs/phases/4-optimize-kernels.md) | each kernel's implementation, one `lib<kernel>.a` | DaCe vectorizer | kernel |
| [5 Sweep Configurations](docs/phases/5-sweep-configurations.md) | compiler, flags, FP mode, ISA per kernel | brute-force sweep | none |

An analysis agent reads offload placements and measured runtimes and [requests changes](docs/phases/feedback.md)
from phase 1. Agents follow [AGENTS.md](AGENTS.md).

## Install and test

```bash
uv sync --extra dev                                # dace @ extended, hpcagent-bench @ main
uv run pytest -m "not integration and not gpu"     # unit set
uv run pytest -m integration                       # compiles and runs kernels
```

Benchmark kernels come from [HPCAgent-Bench](https://github.com/spcl/HPCAgent-Bench); its
loop-level-reasoning track contains TSVC-2.

## Layout

```
nestforge/
  session.py   the one API over all phases
  phases/      normalize, schedule, scopes, offload, kernel, variants, feedback
  ir/          extraction, numpy emission, the ExternalCall library node, structure views
  build/       owned DaCe build, toolchains, flags, the compile-validate-time arena
  corpus/      HPCAgent-Bench kernels and the numpy translator
```

## More docs

- [Emitter contract](docs/emitter.md): how kernels become NumPy oracles and translator input.
- [Build and linking](docs/build.md): owned DaCe build, static archives, one OpenMP runtime.
- [FP modes and vectorization](docs/fp-and-vectorization.md): the FP ladder and vectorizer knobs.

## References

- Phase 0 builds on *The Canonical Parallel Form as a Substrate for Parallelizing Compilers and
  Agentic Optimizers*, which defines the canonical parallel form (CPF).
- Phase 4 agents and the kernel corpus come from *HPCAgent-Bench*.
- Phase 5 is the variant search of *The Data Must Flow (To Vector Processors): Searching Program
  Variants to Improve Compiler Auto-Vectorization Capabilities* (ICS'26).
