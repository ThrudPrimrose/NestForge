# FP modes and vectorization

[Overview](../README.md) · related: [4 Optimize Kernels](phases/4-optimize-kernels.md),
[5 Sweep Configurations](phases/5-sweep-configurations.md)

Phase 4 leaves vectorization to the compiler. Phase 5 sweeps compiler, FP mode and the compiler's
vectorizer cost model over each kernel's CPF unit; `nestforge/build/flags.py` defines both axes.

## FP modes

`FP_LEVELS` lists three modes, strictest first. Each mode has a tolerance against the NumPy float64
oracle (`FP_ATOL`). The oracle is not bit-reproducible itself (pairwise `np.sum`, BLAS dot, libm), so
even `strict-ieee` allows a small relative error.

| Mode | atol | GNU / LLVM | oneAPI |
|---|---|---|---|
| `strict-ieee` | 1e-15 | `-ffp-contract=off` | `-fp-model=strict` |
| `contract-fma` | 1e-13 | `-ffp-contract=fast` | `-fp-model=precise` |
| `fast-math` | 1e-5 | `-ffast-math -mrecip` | `-fp-model=fast=2 -ftz` |

oneAPI compilers default to `-fp-model=fast`, so every mode sets an explicit model.
`fortran_fp_flags` adds `-fno-frontend-optimize` for gfortran below `fast-math`, because its front end
reassociates at `-O` even with `-ffp-contract=off`. `DTYPE_ATOL` adds a per-dtype floor of about one
ULP, combined as `max(mode, dtype)`. GPU variants use only `strict-ieee` and `contract-fma`.

## Cost models

`COST_MODELS` holds three settings. `default` keeps the compiler's own model. `cheap` vectorizes less;
only gcc has the knob (`-fvect-cost-model=cheap`), so the other compilers dedup it onto `default`.
`no-vec` turns the vectorizer off as a scalar baseline.

DaCe's tile-op vectorizer is not part of the pipeline; `BuildOptions.vectorize` still applies it to an
owned DaCe program build.
