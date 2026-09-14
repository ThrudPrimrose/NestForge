# NestForge

NestForge optimizes a whole DaCe program in phases. Each phase makes one decision, ships a
deterministic default, and exposes the same API to a scripted optimizer, a human, and an LLM agent.

[![NestForge phases](docs/figures/pipeline.png)](docs/figures/pipeline.svg)

| Phase | Decides | Default optimizer | Agent |
|---|---|---|---|
| [0 Normalize](docs/phases/0-normalize.md) | canonical parallel form for the enabled targets | canonicalize up to fusion | none |
| [1 Shape Kernels](docs/phases/1-shape-kernels.md) | fusion and fission granularity | fuse all legal loops | scheduling |
| [2 Define Scopes](docs/phases/2-define-scopes.md) | which nests become external kernels | one scope per parallel top-level map | scheduling |
| [3 Offload](docs/phases/3-offload.md) | device per kernel, host/device copies | all scopes on the GPU with a GPU target | scheduling |
| [4 Optimize Kernels](docs/phases/4-optimize-kernels.md) | each kernel's code, one `lib<kernel>.a` | standalone CPF kernel: C++ on CPU, CUDA on GPU | kernel |
| [5 Sweep Configurations](docs/phases/5-sweep-configurations.md) | compiler, FP mode, vectorizer cost model | keep the fastest correct variant | none |

Two analysis agents [request changes](docs/phases/feedback.md): one reads runtimes and sends phase 1
back to reshape kernels, the other reads placements and sends phase 2 back to redefine scopes. Agents
follow [AGENTS.md](AGENTS.md).

## Quick start

```bash
python examples/quickstart.py --device cpu --out quickstart_out
python examples/quickstart.py --device gpu --out quickstart_out
```

The script runs the default optimizer on HPCAgent-Bench's `fuse_diamond` at preset S (`LEN_1D=512`):
four loops where `t = a*a` feeds `u = t + 1` and `v = t - 1`, then `out = u*v`. It prints the structure
tree before phase 0:

```
SDFG 'hpcagent_bench_benchmarks_loop_level_reasoning_fuse_diamond_fuse_diamond_dace_fuse_diamond'
|- for_19  i=0:LEN_1D
|  `- slice_a_20
|- for_21  i=0:LEN_1D
|  `- slice_t_22
|- for_23  i=0:LEN_1D
|  `- slice_t_24
`- for_25  i=0:LEN_1D
   `- slice_u_26
```

and after phase 0, in canonical parallel form:

```
SDFG 'hpcagent_bench_benchmarks_loop_level_reasoning_fuse_diamond_fuse_diamond_dace_fuse_diamond'
`- single_state_body
   `- single_state_body_0_map  [_loop_it_0=0:LEN_1D]  reads=['a'] writes=['out']
```

0. Normalize fuses the four sequential loops into one parallel map that reads `a` and writes `out`.
1. Shape Kernels finds nothing left to fuse.
2. Define Scopes turns the map into one kernel, `extcall_0`.
3. Offload keeps the kernel on the host for CPU; for GPU it runs on the device, with `a` copied in and
   `out` copied back.
4. Optimize Kernels renders `extcall_0` as one CPF C++ or CUDA file and builds `libextcall_0.a`.
5. Sweep Configurations times 15 CPU variants (winner `g++`, `strict-ieee`, `no-vec`, 31.2 us per call)
   or 4 GPU variants (winner `nvcc-13.3`, `strict-ieee`, 6.9 us).

```
quickstart_out/
  0-normalize.sdfg ... 4-optimize-kernels.sdfg   program SDFG per phase (3 only with --device gpu)
  5-sweep-configurations.json                    per nest: compiler, FP mode, cost model, flags, time
  trees/                                         structure before phase 0, after phase 0, after phase 1
  kernels/extcall_0/                             CPF unit (.cpp or .cu) and libextcall_0.a
  program/                                       the program's generated code
  work/                                          build tree
```

## Install and test

```bash
pip install -e ".[dev]"                                   # dace @ extended, hpcagent-bench @ main
pre-commit install                                        # ruff check and ruff format on every commit
pytest -m "not integration and not gpu and not vendor"    # unit set
pytest -m integration                                     # compiles and runs kernels
```

NestForge assumes Linux. Benchmark kernels and the NumPy to C, C++ and Fortran translator come from
[HPCAgent-Bench](https://github.com/spcl/HPCAgent-Bench). HPCAgent-Bench imports NestForge, so
`import nestforge` never loads HPCAgent-Bench; only the functions that need it do.

## Layout

```
nestforge/
  session.py   the one API over all phases
  phases/      normalize, schedule, scopes, offload, kernel, variants, feedback
  ir/          extraction, NumPy emission, the ExternalCall library node, structure views
  build/       compile and link, compilers on PATH, FP flags, the validate-and-time arena
  corpus/      HPCAgent-Bench kernels and the translator bridge
```

More: [emitter contract](docs/emitter.md), [build and runtime linking](docs/build.md),
[FP modes and cost models](docs/fp-and-vectorization.md).

## References

- Phase 0 builds on *The Canonical Parallel Form as a Substrate for Parallelizing Compilers and
  Agentic Optimizers*, which defines the canonical parallel form (CPF).
- Phase 4 agents and the kernel corpus come from *HPCAgent-Bench*.
- Phase 5 is the variant search of *The Data Must Flow (To Vector Processors): Searching Program
  Variants to Improve Compiler Auto-Vectorization Capabilities* (ICS'26).
