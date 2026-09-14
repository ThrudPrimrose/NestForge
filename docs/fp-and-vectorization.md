# FP precision and vectorization

[../README.md](../README.md) · related: [3 Kernel optimization](phases/3-kernel-opt.md) ·
[4 Codegen variants](phases/4-codegen-variants.md)

Phase 3 vectorizes each kernel with one fixed DaCe configuration; phase 4 then sweeps compiler,
FP mode and vectorizer cost model over the result. This page covers the two axes phase 4 sweeps
and the vectorizer knobs a kernel's emitted code can vary on.

## The FP ladder

`nestforge/build/flags.py` defines four FP-precision rungs (`FP_LEVELS`), strictest first, with a
validation tolerance against the NumPy float64 oracle (`FP_ATOL`). The oracle itself is not
bit-reproducible (pairwise `np.sum`, BLAS dot, non-correctly-rounded libm), so even the strictest
rung tolerates a small relative error.

| Rung | atol | gnu / llvm | nvidia | intel |
|---|---|---|---|---|
| `strict-ieee` | 1e-15 | `-ffp-contract=off` | `-Kieee -Mnofma` | `-fp-model=strict` |
| `contract-fma` | 1e-13 | `-ffp-contract=fast` | `-Kieee -Mfma` | `-fp-model=precise` |
| `assume-finite` | 1e-13 | adds `-ffinite-math-only -fno-signed-zeros -fno-trapping-math` | same as `contract-fma` (no per-assumption flag) | adds `-ffinite-math-only -fno-math-errno` |
| `fast-math` | 1e-5 | `-ffast-math -mrecip` | `-fast -Mfma -Mfprelaxed=...` | `-fp-model=fast=2 -ftz` |

Each rung's flags are a superset of the one before it. `intel` defaults to `-fp-model=fast`, so
every rung sets an explicit model rather than relying on a bare `-ffp-contract=off`. `nvidia` has
no per-assumption flags, so `assume-finite` and `contract-fma` compile to the same flags and
`flag_matrix` dedups the pair into one cell. `fortran_fp_flags` applies the Fortran-frontend deltas
(`-fno-frontend-optimize` for gfortran below `fast-math`, since its front end reassociates at `-O`
even under `-ffp-contract=off`).

`DTYPE_ATOL` adds a floor per output dtype (about one ULP of that storage format), composed as
`max(rung, dtype)`, so a rung's tolerance never asks more of fp16 output than fp16 can represent.

## Vectorizer knobs

DaCe's multi-dimensional tile-op vectorizer (`dace.transformation.passes.vectorization`) takes a
`VectorizeConfig` with the knobs that change the emitted C++:

| Knob | Effect on emitted code |
|---|---|
| `widths` | Per-dimension tile width; changes the vector width in every tiled loop. |
| `remainder_strategy` | How a non-divisible map extent splits into a main tile and a remainder: a masked interior plus masked tail, one fully masked map, or a divisible interior plus a scalar/tile tail. |
| `branch_mode` | How a same-write-set `if`/`else` lowers: a per-lane blend (`merge`) or `c*x + (1-c)*y` arithmetic (`fp_factor`, integer-gated). |
| `target_isa` | Target instruction set for the tile-op backend; `AUTO` resolves to the host's best ISA. |
| `fuse_multiply_add` | Fuses `a*b + c` into one FMA. Off by default: a fused FMA rounds once instead of twice, so it changes the result by up to one ULP. |
| `assume_even` | Skips the remainder split outright by assuming every tiled extent divides evenly. |

`fuse_multiply_add` composes with the FP ladder above: a kernel built at `contract-fma` or higher
already lets the compiler fuse multiply-adds, so this knob only matters for the vectorizer's own
tile-op lowering, independent of the compiler flag.
