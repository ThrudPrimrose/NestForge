# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Shared compile-flag matrix for the variant sweep: **FP-precision level** axis crossed with a **vectorizer
cost-model** axis and a **vector math library** axis, per compiler family, for C/C++ and Fortran (see
``docs/fp-and-vectorization.md``). ``intel`` is split from ``llvm`` because icx/icpx/ifx default to
``-fp-model=fast``, so a bare ``-ffp-contract=off`` would leave reassociation/FTZ on.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

# support_rpath_flags is re-exported on purpose: it moved down to toolchain when build.py needed it too,
# and callers here (and their tests) already spell it flags.support_rpath_flags.
from nestforge.build.toolchain import CXX_STD, VECTOR_LIBS, support_rpath_flags

#: FP-precision levels, strictest first; the index is the ladder rung.
FP_LEVELS: Tuple[str, ...] = ("strict-ieee", "contract-fma", "assume-finite", "fast-math")

#: Validation tolerance vs the numpy fp64 oracle, which isn't bit-reproducible itself (pairwise np.sum,
#: BLAS dot, non-correctly-rounded libm), so even ``strict-ieee`` isn't atol 0. A kernel compared against
#: its own NumPy oracle in the SAME op order overrides the strict rung to 0.0
#: (:data:`nestforge.build.arena.ARENA_ATOL`); these values are for the whole-program oracle only.
FP_ATOL: Dict[str, float] = {
    "strict-ieee": 1e-15,
    "contract-fma": 1e-13,
    "assume-finite": 1e-13,
    "fast-math": 1e-5,
}

#: Relative tolerance FLOOR per output dtype: about one ULP of that storage format. A gate tighter than
#: the format's own resolution can never be met -- fp16 carries 11 significand bits, so one ULP is ~1e-3
#: and any fp64-shaped gate rejects every correct fp16 kernel. Composed as ``max(rung, dtype)``: the rung
#: says how much reassociation is allowed, the dtype says how little the format can even represent.
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
        "assume-finite": [
            "-ffp-contract=fast", "-fexcess-precision=standard", "-fno-math-errno", "-fno-trapping-math",
            "-ffinite-math-only", "-fno-signed-zeros"
        ],
        "fast-math": ["-ffast-math", "-mrecip"],
    },
    "llvm": {
        "strict-ieee": ["-ffp-contract=off"],
        "contract-fma": ["-ffp-contract=fast"],
        "assume-finite":
        ["-ffp-contract=fast", "-fno-math-errno", "-fno-trapping-math", "-ffinite-math-only", "-fno-signed-zeros"],
        "fast-math": ["-ffast-math", "-mrecip"],
    },
    "nvidia": {
        # nvc has only whole-model FP knobs: assume-finite collapses to contract-fma (deduped below)
        "strict-ieee": ["-Kieee", "-Mnofma"],
        "contract-fma": ["-Kieee", "-Mfma"],
        "assume-finite": ["-Kieee", "-Mfma"],
        "fast-math": ["-fast", "-Mfma", "-Mfprelaxed=div,sqrt,rsqrt,recip"],
    },
    "intel": {
        # icx/ifx default to -fp-model=fast, so each rung sets an explicit model to reset the baseline.
        "strict-ieee": ["-fp-model=strict"],
        "contract-fma": ["-fp-model=precise"],
        "assume-finite": ["-fp-model=precise", "-ffinite-math-only", "-fno-math-errno"],
        "fast-math": ["-fp-model=fast=2", "-ftz"],
    },
}

#: Native-tuning flag per family (nvc uses -tp=native, not -march=native).
_ARCH: Dict[str, str] = {
    "gnu": "-march=native",
    "llvm": "-march=native",
    "intel": "-march=native",
    "nvidia": "-tp=native"
}

#: Vectorizer cost-model axis. "default" = compiler's own model; "no-vec" = scalar floor; "cheap" =
#: fewer/safer vectorizations (only gcc has a direct knob).
COST_MODELS: Tuple[str, ...] = ("default", "cheap", "no-vec")

#: Vector-math-library axis DOMAIN. Per-family spelling lives in ``toolchain.VectorMathLib``.
VECLIBS: Tuple[str, ...] = ("none", "sleef", "libmvec", "svml")


def base_flags(family: str) -> List[str]:
    """``-O3`` + native tuning + PIC/shared -- the common prefix every cell shares."""
    return ["-O3", _ARCH.get(family, "-march=native"), "-fPIC", "-shared"]


def fortran_fp_flags(family: str, level: str) -> List[str]:
    """FP-mode flags for a family's **Fortran** frontend. gfortran needs ``-fno-frontend-optimize`` (it
    reassociates at ``-O`` even under ``-ffp-contract=off``); drops flags gfortran/ifx reject
    (``f951: sorry, unimplemented``)."""
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
            "nvidia": ["-Mnovect"],
        }.get(family, [])
    if model == "cheap":
        return {"gnu": ["-fvect-cost-model=cheap"]}.get(family, [])
    return []


def flag_matrix(family: str, lang: str = "c") -> List[Tuple[str, str, List[str]]]:
    """``[(fp_level, cost_model, full_flags), ...]`` for a family/language, deduped so a collapse
    (nvidia assume-finite==contract-fma) produces one compile, not two."""
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


# --- the REDUCED FP axis for the full-matrix (tsvc_full) job -----------------------------------------
#: The two FP rungs the full-matrix job TIMES (``strict-ieee`` still runs as its bit-exact gate):
#:  * ``default-fp``    -- compiler's own default at ``-O3``, no flag (vendor-dependent).
#:  * ``no-fast-errno`` -- FMA contraction + ``-fno-math-errno``, no reassociation: "fast but ordered".
REDUCED_FP_MODES: Tuple[str, ...] = ("default-fp", "no-fast-errno")

#: FP-mode flags per (family, reduced-mode) -- C spellings; Fortran deltas via :func:`reduced_fp_flags`.
_REDUCED_FP: Dict[str, Dict[str, List[str]]] = {
    "gnu": {
        "default-fp": [],
        "no-fast-errno": ["-ffp-contract=fast", "-fno-math-errno"],
    },
    "llvm": {
        "default-fp": [],
        "no-fast-errno": ["-ffp-contract=fast", "-fno-math-errno"],
    },
    "nvidia": {
        "default-fp": [],
        "no-fast-errno": ["-Kieee", "-Mfma"],  # nvc has no -fno-math-errno; -Kieee -Mfma == contract-fma
    },
    "intel": {
        "default-fp": [],  # icx default is -fp-model=fast (already non-reproducible)
        "no-fast-errno": ["-fp-model=precise", "-fno-math-errno"],
    },
}

#: Validation tolerance per reduced rung; ``default-fp`` is loose (intel defaults to fast-math) but still
#: catches an O(1) wrong answer, ``no-fast-errno`` is near-bit-exact (only FMA differs).
REDUCED_FP_ATOL: Dict[str, float] = {"default-fp": 1e-6, "no-fast-errno": 1e-12}


def reduced_fp_flags(family: str, mode: str, lang: str = "c") -> List[str]:
    """FP-mode flags for a REDUCED rung (:data:`REDUCED_FP_MODES`), adjusted for ``lang``: Fortran drops
    ``-fno-math-errno``, and gfortran's ``no-fast-errno`` gains ``-fno-frontend-optimize`` (it
    reassociates at ``-O`` otherwise)."""
    flags = [f for f in _REDUCED_FP[family][mode] if not (lang == "fortran" and f == "-fno-math-errno")]
    if lang == "fortran" and family == "gnu" and mode == "no-fast-errno":
        flags.append("-fno-frontend-optimize")
    return flags


def cxx_source_flags(family: str, cxx_std: str = CXX_STD) -> List[str]:
    """Flags to compile the numpyto-emitted **C** source as C++ (there is no distinct C++ target). C++
    lacks ``restrict``, and gnu also lacks ``__builtin_complex`` in C++ mode, so both are shimmed."""
    flags = ["-x", "c++", "-std=" + cxx_std, "-Drestrict=__restrict__"]
    if family == "gnu":
        flags.append("-D__builtin_complex(re,im)=((__complex__ double){re,im})")
    return flags


def resolve_veclib(compiler: Optional[str], veclib: Optional[str]) -> Tuple[Optional[object], Optional[str]]:
    """``(library, reason)``. ``(None, None)`` is the scalar baseline -- no library and no problem -- so
    callers branch on ``reason``, not on the library."""
    if not veclib or veclib == "none":
        return None, None
    if not compiler:
        return None, f"veclib {veclib} requested without a compiler to resolve its family"
    vl = VECTOR_LIBS.get(veclib)
    if vl is None:
        return None, f"unknown veclib {veclib!r} (expected one of {tuple(VECTOR_LIBS)})"
    if not vl.compatible(compiler):
        return None, f"veclib {veclib} incompatible with {Path(compiler).name}"
    return vl, None


def veclib_compile_flags(compiler: Optional[str], veclib: Optional[str]) -> Tuple[Optional[List[str]], Optional[str]]:
    """COMPILE half: ``-fveclib=`` on llvm, nothing on gnu (``-ffast-math`` alone emits ``_ZGV*`` there)."""
    vl, reason = resolve_veclib(compiler, veclib)
    if reason is not None:
        return None, reason
    return ([] if vl is None else vl.compile_flags(compiler)), None


def veclib_link_flags(compiler: Optional[str], veclib: Optional[str]) -> Tuple[Optional[List[str]], Optional[str]]:
    """LINK half: the ``-l``/``-L``/``-rpath`` that make the packed calls resolve.

    Kept separate from the compile half because a shared object may carry undefined symbols: an object
    compiled with a veclib and linked without one links CLEANLY and fails at ``dlopen``. Any path that
    compiles and links in two steps must ask for this half explicitly.
    """
    vl, reason = resolve_veclib(compiler, veclib)
    if reason is not None:
        return None, reason
    # icx auto-links libsvml/libimf, so even the scalar baseline needs the support rpath
    support = list(support_rpath_flags(compiler)) if compiler else []
    return (support if vl is None else vl.link_flags(compiler) + support), None


def veclib_flags(compiler: Optional[str], veclib: Optional[str]) -> Tuple[Optional[List[str]], Optional[str]]:
    """Both halves, for a cell that compiles and links in ONE command (see :func:`veclib_link_flags`)."""
    compile_half, reason = veclib_compile_flags(compiler, veclib)
    if compile_half is None:
        return None, reason
    link_half, reason = veclib_link_flags(compiler, veclib)
    if link_half is None:
        return None, reason
    return compile_half + link_half, None


def lane_flags(family: str,
               fp_mode: str,
               cost_model: str,
               lang: str,
               cxx_std: str = CXX_STD,
               compiler: Optional[str] = None,
               veclib: Optional[str] = None) -> Tuple[Optional[List[str]], Optional[str]]:
    """Compose the full compile flags for ONE sweep cell, or ``(None, reason)`` when the axis combination is
    unsupported.

    :param fp_mode: a :data:`FP_LEVELS` rung or a :data:`REDUCED_FP_MODES` rung.
    :param lang: ``"c"``, ``"c++"`` (translator C compiled as C++) or ``"fortran"``.
    :param compiler: resolves veclib compatibility and the driver's support rpath; ``None`` composes only."""
    # compiler_family also yields 'intel-classic', which has no rung here (classic icc != icx): decline, not KeyError
    if family not in _FP:
        return None, f"no fp flag matrix for compiler family {family!r} (known: {tuple(_FP)})"
    fp_lang = "fortran" if lang == "fortran" else "c"
    out = base_flags(family)
    if lang == "c++":
        out = out + cxx_source_flags(family, cxx_std)
    # BOTH axes are accepted: routing every non-strict-ieee value to reduced_fp_flags made the other
    # FP_LEVELS rungs raise KeyError deep in _REDUCED_FP instead of composing.
    if fp_mode in FP_LEVELS:
        out = out + fp_flags(family, fp_mode, fp_lang)
    elif fp_mode in REDUCED_FP_MODES:
        out = out + reduced_fp_flags(family, fp_mode, fp_lang)
    else:
        return None, f"unknown fp_mode {fp_mode!r} (known: {FP_LEVELS + REDUCED_FP_MODES})"
    out = out + cost_flags(family, cost_model)
    vec, vreason = veclib_flags(compiler, veclib)
    if vec is None:
        return None, vreason
    return out + vec, None
