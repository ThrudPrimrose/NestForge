# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Shared compile-flag matrix for the variant sweep: FP-precision level axis crossed with a vectorizer
cost-model axis, per compiler family, for C/C++ and Fortran. ``intel`` is split from ``llvm`` because
icx/icpx/ifx default to ``-fp-model=fast``."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from nestforge.build.toolchain import CXX_STD

#: FP-precision levels, strictest first; the index is the ladder rung.
FP_LEVELS: Tuple[str, ...] = ("strict-ieee", "contract-fma", "fast-math")

#: Validation tolerance vs the numpy fp64 oracle, which isn't bit-reproducible itself (pairwise np.sum, BLAS
#: dot, non-correctly-rounded libm), so even ``strict-ieee`` isn't atol 0.
FP_ATOL: Dict[str, float] = {
    "strict-ieee": 1e-15,
    "contract-fma": 1e-13,
    "fast-math": 1e-5,
}

#: Relative tolerance floor per output dtype (about one ULP of that storage format), composed as
#: ``max(rung, dtype)`` so a gate never demands more precision than the format can represent.
DTYPE_ATOL: Dict[str, float] = {
    "float64": 2.3e-16,
    "float32": 1.2e-7,
    "float16": 9.8e-4,
}

#: FP-mode flags per (family, level) -- C spellings; Fortran deltas applied by :func:`fortran_fp_flags`.
_FP: Dict[str, Dict[str, List[str]]] = {
    "gnu": {
        "strict-ieee": ["-ffp-contract=off", "-fexcess-precision=standard"],
        "contract-fma": ["-ffp-contract=fast", "-fexcess-precision=standard"],
        "fast-math": ["-ffast-math", "-mrecip"],
    },
    "llvm": {
        "strict-ieee": ["-ffp-contract=off"],
        "contract-fma": ["-ffp-contract=fast"],
        "fast-math": ["-ffast-math", "-mrecip"],
    },
    "intel": {
        # icx/ifx default to -fp-model=fast, so each rung sets an explicit model to reset the baseline.
        "strict-ieee": ["-fp-model=strict"],
        "contract-fma": ["-fp-model=precise"],
        "fast-math": ["-fp-model=fast=2", "-ftz"],
    },
}

#: Native-tuning flag per family.
_ARCH: Dict[str, str] = {
    "gnu": "-march=native",
    "llvm": "-march=native",
    "intel": "-march=native",
}

#: Vectorizer cost-model axis: "default" = compiler's own model, "no-vec" = scalar floor, "cheap" = fewer
#: vectorizations (only gcc has a direct knob).
COST_MODELS: Tuple[str, ...] = ("default", "cheap", "no-vec")


def base_flags(family: str) -> List[str]:
    """``-O3`` + native tuning + PIC/shared -- the common prefix every cell shares."""
    return ["-O3", _ARCH.get(family, "-march=native"), "-fPIC", "-shared"]


def fortran_fp_flags(family: str, level: str) -> List[str]:
    """FP-mode flags for a family's Fortran frontend; gfortran needs ``-fno-frontend-optimize``, since it
    reassociates at ``-O`` even under ``-ffp-contract=off``."""
    drop = {"-fno-math-errno", "-fexcess-precision=standard"}  # C-family flags the Fortran frontends reject
    flags = [f for f in _FP[family][level] if f not in drop]
    if family == "gnu":
        if level != "fast-math":
            flags.append("-fno-frontend-optimize")
        else:
            flags.append("-fno-protect-parens")
    return flags


def fp_flags(family: str, level: str, lang: str = "c") -> List[str]:
    """FP-mode flags for a (family, level), adjusted for ``lang`` ("c" or "fortran")."""
    return fortran_fp_flags(family, level) if lang == "fortran" else list(_FP[family][level])


def cost_flags(family: str, model: str) -> List[str]:
    """Vectorizer cost-model flags for a family. Empty where the family has no equivalent knob."""
    if model == "no-vec":
        return {
            "gnu": ["-fno-tree-vectorize"],
            "llvm": ["-fno-vectorize", "-fno-slp-vectorize"],
            "intel": ["-fno-vectorize", "-fno-slp-vectorize"],
        }.get(family, [])
    if model == "cheap":
        return {"gnu": ["-fvect-cost-model=cheap"]}.get(family, [])
    return []


def flag_matrix(family: str, lang: str = "c") -> List[Tuple[str, str, List[str]]]:
    """``[(fp_level, cost_model, full_flags), ...]`` for a family/language, deduped by flag set."""
    matrix: List[Tuple[str, str, List[str]]] = []
    seen = set()
    base = base_flags(family)
    for level in FP_LEVELS:
        for model in COST_MODELS:
            flags = base + fp_flags(family, level, lang) + cost_flags(family, model)
            key = tuple(flags)
            if key in seen:
                continue
            seen.add(key)
            matrix.append((level, model, flags))
    return matrix


def cxx_source_flags(family: str, cxx_std: str = CXX_STD) -> List[str]:
    """Flags to compile the numpyto-emitted C source as C++; ``restrict`` and (on gnu) ``__builtin_complex``
    are shimmed since C++ lacks both."""
    flags = ["-x", "c++", "-std=" + cxx_std, "-Drestrict=__restrict__"]
    if family == "gnu":
        flags.append("-D__builtin_complex(re,im)=((__complex__ double){re,im})")
    return flags


#: nvcc's FP rungs. The device has no fast-math switch, so a GPU kernel sweeps the two rungs nvcc expresses.
CUDA_FP_LEVELS: Tuple[str, ...] = ("strict-ieee", "contract-fma")

#: Device flag per rung (``--fmad`` fuses multiply-adds on the device), plus the host rung for the unit's host code.
CUDA_FP: Dict[str, List[str]] = {
    "strict-ieee": ["--fmad=false", "-Xcompiler=-ffp-contract=off"],
    "contract-fma": ["--fmad=true", "-Xcompiler=-ffp-contract=fast"],
}

#: The cost-model value of a GPU cell: nvcc has no vectorizer cost model to sweep.
NO_COST_MODEL = "none"


def cuda_base_flags(build_flags: Sequence[str]) -> List[str]:
    """``build_flags`` (what CPF's CUDA unit needs) + ``-O3`` + native device arch + PIC/shared: the one place GPU
    flags are composed, and the counterpart of :func:`base_flags`."""
    return [*build_flags, "-O3", "-arch=native", "-Xcompiler=-fPIC", "-shared"]


def cuda_flag_matrix(build_flags: Sequence[str]) -> List[Tuple[str, List[str]]]:
    """``[(fp_level, full_flags), ...]`` for nvcc."""
    base = cuda_base_flags(build_flags)
    return [(level, base + CUDA_FP[level]) for level in CUDA_FP_LEVELS]
