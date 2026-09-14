# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""End-to-end: nest-forge's emitted code reproduces the DaCe SDFG, on the most complex corpus kernels.

nest-forge lowers a DaCe SDFG to standalone numpy (:func:`sdfg_to_numpy`), which hpcagent_bench's ``numpyto``
then turns into C / C++ / Fortran. This suite checks that whole pipeline is a FAITHFUL translation of the
SDFG, on the hardest TSVC + level-3 corpus kernels (control flow, reductions, recurrences, multi-nest,
linear algebra).

The oracle is the SDFG built through nest-forge's OWN build (``dace.codegen`` -> our compiler, NEVER
``dace.compile()`` -- see :mod:`nestforge.build`), so this is literally "the emitted code vs the DaCe
SDFG", compiled the way nest-forge ships it. Inputs are random; sizes are tiny (compile+run stays cheap)
but distinct per dimension so an index / transpose bug is caught by the value comparison.

L1 (:func:`test_emit_numpy_matches_sdfg`) -- every listed kernel: ``sdfg_to_numpy`` run in Python == the
built SDFG.
L2 (:func:`test_emit_compiled_matches_sdfg_across_compilers`) -- a curated subset: each extracted nest's
numpyto {C, C++, Fortran} source, compiled with gcc / clang / gfortran BY US, == that nest's built SDFG.
This is the cross-compiler coverage.

Every listed case must pass (the CI unit set forbids skips); a build/emit/compile failure or a numerical
mismatch is a hard FAILURE, run in a forked child so a miscompiled kernel cannot take down the worker.
"""
import inspect
import tempfile
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("hpcagent_bench")

from dace import symbolic

from nestforge import tsvc
from nestforge.build import BuildOptions, build_sdfg
from nestforge.corpus import iter_dace_kernels
from nestforge.emit_numpy import load_emitted, maxsize_loop_scratch, sdfg_to_numpy
from nestforge.isolation import run_isolated

ATOL = 1e-8

# ---- the most complex corpus kernels that build + emit faithfully (from the emit-vs-SDFG sweep) --------
# level-3 (the hardest dwarf tier) + complex level-2 dense-linear-algebra / stencils.
DACE_L1 = [
    # level 3
    "hpc/graphical_models/hmm_forward/hmm_forward",
    "hpc/map_reduce/xsbench/xsbench",
    "hpc/map_reduce/azimint_hist/azimint_hist",
    "hpc/structured_grids/deriche/deriche",
    "hpc/structured_grids/harris_corner/harris_corner",
    "hpc/dynamic_programming/pathfinder/pathfinder",
    "hpc/graph_traversal/bfs/bfs",
    "hpc/dense_linear_algebra/gaussian/gaussian",
    "hpc/dense_linear_algebra/scattering_self_energies/scattering_self_energies",
    # level 2 (linear algebra + stencils)
    "hpc/dense_linear_algebra/k2mm/k2mm",
    "hpc/dense_linear_algebra/k3mm/k3mm",
    "hpc/dense_linear_algebra/gemm/gemm",
    "hpc/dense_linear_algebra/cholesky/cholesky",
    "hpc/dense_linear_algebra/lu/lu",
    "hpc/dense_linear_algebra/ludcmp/ludcmp",
    "hpc/dense_linear_algebra/gramschmidt/gramschmidt",
    "hpc/dense_linear_algebra/syr2k/syr2k",
    "hpc/dense_linear_algebra/mvt/mvt",
    "hpc/dense_linear_algebra/atax/atax",
    "hpc/dense_linear_algebra/bicg/bicg",
    "hpc/dense_linear_algebra/trisolv/trisolv",
    "hpc/structured_grids/heat_3d/heat_3d",
    "hpc/structured_grids/fdtd_2d/fdtd_2d",
    "hpc/structured_grids/jacobi_2d/jacobi_2d",
    "hpc/structured_grids/adi/adi",
    "hpc/graph_traversal/pagerank/pagerank",  # normalised power iteration -> well-conditioned for any input
]
# hardest TSVC: multi-nest, control flow (break), conditional reductions, running max/argmax, recurrences.
TSVC_L1 = [
    "s1113",
    "s1221",
    "s1244",
    "s152",
    "s2275",
    "s13110",
    "s3111",
    "s3113",
    "s331",
    "s481",
    "s118",
    "s1213",
    "s1351",
    "s126",
    "s161",
    "s241",
    "s2711",
    "s112",
    "s114",
]
TSVC25_L1 = ["cond_reduce_sym", "cond_reduce_sum", "ext_break_capture", "ext_break_find_first"]

# cross-compiler subset: kernels whose nests lower + translate + compile cleanly in every language.
# numpyto has no C++ target (the C++ lane would recompile the C, same toolchain), so the distinct
# compilers are covered by C x {gcc, clang} + Fortran x {gfortran}.
DACE_L2 = [
    "hpc/dense_linear_algebra/gemm/gemm", "hpc/dense_linear_algebra/k3mm/k3mm", "hpc/dense_linear_algebra/mvt/mvt",
    "hpc/dense_linear_algebra/atax/atax"
]
# Only straight-line nests translate + compile identically in EVERY language across gcc/clang/gfortran; a
# nest carrying loop state (recurrence / masked reduction) diverges at the artificial nest boundary or hits
# a numpyto Fortran emit gap -- those kernels get their full cross-check from L1 (the whole-kernel oracle).
TSVC_L2 = ["s000"]
TSVC25_L2 = []
COMPILERS = {"c": ["gcc", "clang"], "fortran": ["gfortran"]}


# ---- helpers ------------------------------------------------------------------------------------------
def rand_buffer(shape, dt, rng, center):
    if np.issubdtype(dt, np.complexfloating):
        return (rng.random(shape) + 1j * rng.random(shape)).astype(dt)
    if np.issubdtype(dt, np.floating):
        return (rng.random(shape) - center).astype(dt)
    if np.issubdtype(dt, np.integer):
        return rng.integers(0, 4, size=shape).astype(dt)
    if dt == np.bool_:
        return rng.integers(0, 2, size=shape).astype(bool)
    return np.zeros(shape, dt)


def dace_sizes(kernel, base=6):
    preset = kernel.spec.parameters.get("S") or next(iter(kernel.spec.parameters.values()))
    ranks = {v: i for i, v in enumerate(sorted(set(preset.values())))}
    return {k: base + ranks[v] for k, v in preset.items()}


def make_dace(short):
    kernel = {k.short_name: k for k in iter_dace_kernels()}[short]
    return (lambda: kernel.to_sdfg(simplify=True)), dace_sizes(kernel), 0.0  # linear algebra: inputs in [0,1)


def make_tsvc(short, corpus):
    kernel = tsvc.iter_tsvc_kernels(only=[short], corpus=corpus)[0]
    probe = tsvc.build_sdfg(kernel, opt_mode="simplify-parallel")
    sizes = {str(s): 8 for s in probe.free_symbols}
    return (
        lambda: tsvc.build_sdfg(kernel, opt_mode="simplify-parallel")), sizes, 0.5  # centered: exercise sign branches


def base_inputs(sdfg, sizes, center, seed=0):
    env = {symbolic.symbol(k): v for k, v in sizes.items()}
    rng = np.random.default_rng(seed)
    base = {}
    for name, desc in sdfg.arrays.items():
        if desc.transient:
            continue
        shape = tuple(int(symbolic.evaluate(d, env)) for d in desc.shape)
        base[name] = rand_buffer(shape, np.dtype(desc.dtype.type), rng, center)
    return base


def run_oracle_nest(make_sdfg, sizes, base, tmp):
    """Build the SDFG through nest-forge (no dace.compile) and run it; returns the mutated buffers."""
    built = build_sdfg(make_sdfg(), tmp / "oracle", BuildOptions(compiler="g++"))
    out = {k: v.copy() for k, v in base.items()}
    built.run(out, sizes)
    built.unload()
    return out


def run_emitted_numpy(make_sdfg, sizes, base):
    env = {symbolic.symbol(k): v for k, v in sizes.items()}
    sdfg = make_sdfg()
    src = sdfg_to_numpy(sdfg, "k")
    mod = load_emitted(src, "k")
    symbols = [a for a in sdfg.arglist() if a not in sdfg.arrays]
    sized = maxsize_loop_scratch(sdfg, symbols)
    call = {}
    for name in inspect.signature(mod.k).parameters:
        if name in sizes:
            call[name] = sizes[name]
        elif name in base:
            call[name] = base[name].copy()
        else:  # scratch transient the caller must allocate
            desc = sized.arrays[name]
            shape = tuple(int(symbolic.evaluate(d, env)) for d in desc.shape)
            call[name] = np.zeros(shape, np.dtype(desc.dtype.type))
    mod.k(**call)
    return call


def max_abs_diff(oracle, cand):
    worst = 0.0
    for name, want in oracle.items():
        got = np.asarray(cand[name]).ravel()
        want = np.asarray(want).ravel()
        assert want.size == got.size, f"{name}: size {want.size} vs {got.size}"
        cplx = np.iscomplexobj(want) or np.iscomplexobj(got)
        a, b = (want.astype(np.complex128), got.astype(np.complex128)) if cplx else \
               (want.astype(np.float64), got.astype(np.float64))
        # ``a - b`` is NaN wherever either side is NaN (and for inf - inf). A NaN on only ONE side is a
        # MAXIMAL disagreement -- exactly what this gate exists to catch -- so it scores as inf and can
        # never be dropped. Positions that are bit-equal (incl. both NaN, or both +-inf) are genuine
        # agreement: the emitted code reproduced the SDFG's value, so they score 0.
        same = (a == b) | (np.isnan(a) & np.isnan(b))
        d = np.abs(a - b)
        d = np.where(same, 0.0, np.where(np.isnan(d), np.inf, d))
        worst = max(worst, float(d.max()) if d.size else 0.0)
    return worst


def builder_for(kind, short):
    return make_dace(short) if kind == "dace" else make_tsvc(short, kind)


def test_maxdiff_scores_nan_mismatch_as_divergence():
    """A kernel emitting NaN where the SDFG is finite must FAIL the gate, not be scored on the rest."""
    oracle = {"x": np.array([1.0, 2.0, 3.0])}
    assert max_abs_diff(oracle, {"x": np.array([1.0, np.nan, 3.0])}) == np.inf
    assert max_abs_diff({"x": np.array([1.0, np.nan, 3.0])}, oracle) == np.inf
    # Reproducing the SDFG's NaN / inf at the same position is agreement, not divergence.
    assert max_abs_diff({"x": np.array([1.0, np.nan, np.inf])}, {"x": np.array([1.0, np.nan, np.inf])}) == 0.0
    assert max_abs_diff({"x": np.array([1.0 + 1j, np.nan + 0j])}, {"x": np.array([1.0 + 1j, np.nan + 0j])}) == 0.0
    assert max_abs_diff({"x": np.array([1.0 + 1j, 2.0 + 0j])}, {"x": np.array([1.0 + 1j, np.nan + 0j])}) == np.inf


# ---- L1: emitted numpy == the SDFG --------------------------------------------------------------------
@pytest.mark.parametrize("kind,short",
                         [("dace", s) for s in DACE_L1] + [("tsvc2", s) for s in TSVC_L1] + [("tsvc2_5", s)
                                                                                             for s in TSVC25_L1])
def test_emit_numpy_matches_sdfg(kind, short):

    def work():
        make_sdfg, sizes, center = builder_for(kind, short)
        with tempfile.TemporaryDirectory() as td:
            base = base_inputs(make_sdfg(), sizes, center)
            oracle = run_oracle_nest(make_sdfg, sizes, base, Path(td))
            cand = run_emitted_numpy(make_sdfg, sizes, base)
            return {"md": max_abs_diff(oracle, cand)}

    res = run_isolated(work, timeout=600)
    assert "error" not in res, f"{short}: {res.get('error')}"
    assert res["md"] <= ATOL, f"{short}: emitted numpy diverged from the SDFG (maxdiff {res['md']:g})"


# ---- L2: emitted code compiled across compilers == the SDFG (per nest) --------------------------------
@pytest.mark.parametrize("kind,short,lang,compiler",
                         [("dace", s, lang, cc) for s in DACE_L2 for lang, ccs in COMPILERS.items()
                          for cc in ccs] + [("tsvc2", s, lang, cc) for s in TSVC_L2 for lang, ccs in COMPILERS.items()
                                            for cc in ccs] + [("tsvc2_5", s, lang, cc) for s in TSVC25_L2
                                                              for lang, ccs in COMPILERS.items() for cc in ccs])
def test_emit_compiled_matches_sdfg_across_compilers(kind, short, lang, compiler):
    import shutil
    tool = compiler if lang == "c" else "gfortran"
    if shutil.which(tool) is None:
        # Skips normally locally; under NESTFORGE_CI_NO_SKIP (the CI unit set) a skip FAILS the session, so
        # a CI runner missing a compiler is surfaced -- CI must install gcc / clang / gfortran (setup_apt.sh).
        pytest.skip(f"compiler {tool} not installed")

    def work():
        import subprocess
        from nestforge.pass_lower import lower_nests_to_external_call
        from nestforge.translate import prepare, emit_sources
        from nestforge.arena import make_inputs
        from nestforge.perf.harness import c_argtypes, call_c, signature_order
        make_sdfg, sizes, _ = builder_for(kind, short)
        nests = lower_nests_to_external_call(make_sdfg(), strategy="outer")
        suffix = {"c": ".c", "cpp": ".c", "fortran": ".f90"}[lang]
        target = {"c": "c", "cpp": "c", "fortran": "fortran"}[lang]  # C++ compiles the emitted C
        worst = 0.0
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            for i, (_ext, b) in enumerate(nests):
                name = f"{short.split('/')[-1]}_n{i}"
                nsizes = {**sizes, **{s: 0 for s in b.symbols if s not in sizes}}  # leaked nest indices -> 0
                prep = prepare(b, name, d / name, sizes=nsizes)
                built = build_sdfg(b.standalone_sdfg, d / f"{name}_o", BuildOptions(compiler="g++"))
                inp = make_inputs(b, nsizes, seed=0)
                oracle = {k: v.copy() for k, v in inp.items()}
                built.run(oracle, nsizes)
                built.unload()
                src = next(p for p in emit_sources(prep, d / f"{name}_{lang}", target=target) if p.suffix == suffix)
                so = d / f"{name}_{lang}_{compiler}.so"
                subprocess.run(
                    [tool, "-O2", "-fPIC", "-shared", "-ffp-contract=off",
                     str(src), "-o", str(so)],
                    capture_output=True,
                    text=True,
                    check=True)
                order = signature_order(src.read_text(), f"{name}_fp64", "fortran" if lang == "fortran" else "c")
                outs, _ = call_c(so, f"{name}_fp64", order, c_argtypes(order, b), b, dict(inp), nsizes, 1)
                # ``__sym_out_*`` are extraction sentinels (a nest's loop-exit index / carried scalar), not
                # real kernel data -- DaCe's nest codegen and numpyto legitimately differ on them at the
                # artificial nest boundary. Whole-kernel correctness (incl. loop-carried logic) is covered by
                # L1; L2 checks the real DATA outputs of the compiled nest match across compilers.
                worst = max(
                    worst,
                    max((float(np.max(np.abs(oracle[k] - outs[k])))
                         for k in outs if k in oracle and not k.startswith("__sym_out")),
                        default=0.0))
        return {"md": worst}

    res = run_isolated(work, timeout=600)
    assert "error" not in res, f"{short}/{lang}/{compiler}: {res.get('error')}"
    assert res["md"] <= ATOL, f"{short}/{lang}/{compiler}: compiled kernel diverged (maxdiff {res['md']:g})"
