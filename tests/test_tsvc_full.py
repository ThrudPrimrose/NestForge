# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit + end-to-end tests for the TSVC full-matrix driver (nestforge.perf.tsvc_full).

Pure-logic tests (timing, flag composition, sizing, winner selection, table math) run without a compiler;
end-to-end tests compile and run real kernels through all three lanes, skipping when no toolchain is on PATH.
"""
import ctypes
import json
import shutil
import subprocess
from typing import List, Optional

import pytest

from nestforge import tsvc
from nestforge.perf import flags, tsvc_full
from nestforge.perf.harness import lang_compilers
from nestforge.toolchain import discover_toolchains


# --- median-of-N timing summary (pure) ----------------------------------------------------------------
def test_summarize_times_median_min_percentiles():
    s = tsvc_full.summarize_times([10.0, 2.0, 6.0, 4.0, 8.0])  # sorted: 2,4,6,8,10
    assert s["median_us"] == 6.0
    assert s["min_us"] == 2.0
    assert s["mean_us"] == 6.0
    assert s["p25_us"] == 4.0 and s["p75_us"] == 8.0  # linear-interpolated quartiles on the 5-sample set


def test_summarize_times_single_and_empty():
    one = tsvc_full.summarize_times([3.5])
    assert one["median_us"] == one["min_us"] == one["p25_us"] == one["p75_us"] == 3.5
    empty = tsvc_full.summarize_times([])
    assert all(v == float("inf") for v in empty.values())


# --- reduced FP axis ----------------------------------------------------------------------------------
def test_reduced_fp_modes_and_atol():
    assert flags.REDUCED_FP_MODES == ("default-fp", "no-fast-errno")
    assert set(flags.REDUCED_FP_ATOL) == set(flags.REDUCED_FP_MODES)
    assert flags.reduced_fp_flags("gnu", "default-fp") == []  # vendor default: no FP flag
    assert flags.reduced_fp_flags("gnu", "no-fast-errno", "c") == ["-ffp-contract=fast", "-fno-math-errno"]
    # Fortran drops -fno-math-errno (no errno) and gains gfortran's anti-reassociation guard.
    ftn = flags.reduced_fp_flags("gnu", "no-fast-errno", "fortran")
    assert "-fno-math-errno" not in ftn and "-fno-frontend-optimize" in ftn
    # nvc has no -fno-math-errno, so its no-fast rung is exactly contract-fma.
    assert flags.reduced_fp_flags("nvidia", "no-fast-errno") == ["-Kieee", "-Mfma"]


# --- auto-parallelization axis ------------------------------------------------------------------------
def test_autopar_flags_per_family(monkeypatch):
    # gcc default auto-par is Graphite: parloops + isl-loop-nest-optimize + -fgraphite-identity (forces SCoP
    # construction). Never forces the back end past its own cost model (no -process-unprofitable).
    gnu, r = flags.autopar_flags("gnu", 72)
    assert gnu == [
        "-ftree-parallelize-loops=72", "-floop-parallelize-all", "-floop-nest-optimize", "-fgraphite-identity",
        "-fopenmp"
    ]
    assert r is None
    # llvm default auto-par is Polly (mirrors optarena POLLY_PAR).
    llvm, rl = flags.autopar_flags("llvm", 8)
    assert llvm == ["-mllvm", "-polly", "-mllvm", "-polly-parallel", "-fopenmp"] and rl is None
    assert flags.autopar_flags("nvidia", 8)[0] == ["-Mconcur"]
    assert flags.autopar_flags("intel", 8)[0] == ["-qopenmp", "-parallel"]
    # Both polyhedral back ends are OPTIONAL, gated by TWO probes: flag accepted (compiler_accepts) AND
    # actually parallel (autopar_fires) -- an inert-but-accepted back end (Ubuntu clang's Polly) must be a
    # recorded skip, not a false "parallel" cell. Forced here so the test doesn't depend on this box's clang.
    monkeypatch.setattr(flags, "compiler_accepts", lambda *a, **k: False)
    apl, reasonl = flags.autopar_flags("llvm", 8, compiler="clang")
    assert apl is None and "Polly" in reasonl
    apg, reasong = flags.autopar_flags("gnu", 8, compiler="gcc")
    assert apg is None and "Graphite" in reasong
    # accepted but INERT (fires False) -> still a skip, now for the emits-no-parallel-loop reason.
    monkeypatch.setattr(flags, "compiler_accepts", lambda *a, **k: True)
    monkeypatch.setattr(flags, "autopar_fires", lambda *a, **k: False)
    apn, reasonn = flags.autopar_flags("llvm", 8, compiler="clang")
    assert apn is None and "no parallel loop" in reasonn
    # accepted AND firing -> the flags come back.
    monkeypatch.setattr(flags, "autopar_fires", lambda *a, **k: True)
    ok, rok = flags.autopar_flags("llvm", 8, compiler="clang")
    assert ok[:2] == ["-mllvm", "-polly"] and rok is None


# --- C++ lane flag composition (the C source recompiled as C++) --------------------------------------
def test_cxx_source_flags_restrict_and_builtin_complex():
    for fam in ("gnu", "llvm", "nvidia", "intel"):
        f = flags.cxx_source_flags(fam)
        assert f[:3] == ["-x", "c++", "-std=" + flags.CXX_STD]
        assert "-Drestrict=__restrict__" in f  # C++ has no `restrict` keyword
    # only g++ lacks __builtin_complex in C++ mode -> it alone gets the compound-literal shim.
    assert any("__builtin_complex" in t for t in flags.cxx_source_flags("gnu"))
    assert not any("__builtin_complex" in t for t in flags.cxx_source_flags("llvm"))


def test_lane_flags_gate_reduced_cxx_and_unsupported(monkeypatch):
    # strict-ieee gate uses the ladder's strict flags.
    gate, r = flags.lane_flags("gnu", "strict-ieee", "default", "sequential", "c", 4)
    assert r is None and "-ffp-contract=off" in gate
    # a reduced timing cell in C++ carries the C++ frontend flags + the C FP spelling.
    cxx, _ = flags.lane_flags("gnu", "no-fast-errno", "default", "sequential", "c++", 4)
    assert "-x" in cxx and "-ffp-contract=fast" in cxx and "-Drestrict=__restrict__" in cxx
    # llvm auto-par is Polly now (a real lane): with no compiler to probe, the intended flags come back.
    poll, reason = flags.lane_flags("llvm", "default-fp", "default", "auto-par", "c", 4)
    assert reason is None and "-polly" in poll
    # when a compiler IS probed and its polyhedral back end is absent, the cell records (None, reason).
    monkeypatch.setattr(flags, "compiler_accepts", lambda *a, **k: False)
    none, why = flags.lane_flags("llvm", "default-fp", "default", "auto-par", "c", 4, compiler="clang")
    assert none is None and why
    # gcc auto-par threads the requested count into the flag.
    par, _ = flags.lane_flags("gnu", "default-fp", "default", "auto-par", "c", 16)
    assert "-ftree-parallelize-loops=16" in par


def test_lane_flags_declines_a_family_the_matrix_has_no_rung_for():
    """``toolchain.compiler_family`` also returns ``intel-classic`` (icc/icpc/ifort), which this matrix has no
    rung for -- classic icc is not icx. Every other unsupported axis degrades to ``(None, reason)``, but
    ``_FP`` was indexed directly, so an icc backend raised KeyError and killed the sweep instead of
    recording a skip."""
    none, why = flags.lane_flags("intel-classic", "strict-ieee", "default", "sequential", "c", 1)
    assert none is None and "intel-classic" in why
    # and the families it DOES know still compose at every rung -- a blanket decline would be no better
    for family in ("gnu", "llvm", "nvidia", "intel"):
        for level in flags.FP_LEVELS:
            composed, reason = flags.lane_flags(family, level, "default", "sequential", "c", 1)
            assert composed and reason is None, (family, level)


def test_lane_flags_omp_emit_enables_openmp_every_family():
    # omp-emit is supported for EVERY family (pragmas are in OUR source, a bare -fopenmp/-mp, not an
    # auto-parallelizer) -- so clang (which auto-par cannot reach) still gets an omp-emit lane.
    assert "omp-emit" in flags.PARALLEL_MODES
    for fam, sw in (("gnu", "-fopenmp"), ("llvm", "-fopenmp"), ("intel", "-fopenmp"), ("nvidia", "-mp")):
        omp, reason = flags.lane_flags(fam, "default-fp", "default", "omp-emit", "c", 8)
        assert reason is None and sw in omp
        assert "-ftree-parallelize-loops" not in " ".join(omp)  # NOT the auto-parallelizer path


# --- PROF sizing (must exceed the GH200 Grace L3) -----------------------------------------------------
def test_prof_preset_exceeds_l3():
    l3_bytes = 114 * 2**20  # ~114 MB Grace L3/socket
    assert tsvc._PRESET["LEN_1D"]["PROF"] * 8 > l3_bytes  # one fp64 array out of L3
    assert tsvc._PRESET["LEN_2D"]["PROF"]**2 * 8 > l3_bytes
    assert tsvc._PRESET["LEN_3D"]["PROF"]**3 * 8 > l3_bytes
    # PROF sits between L and XL for the 1D axis (bigger than L, the sweep's timing size).
    assert tsvc._PRESET["LEN_1D"]["L"] < tsvc._PRESET["LEN_1D"]["PROF"]


def test_validate_cap_never_runs_big_oracle():
    assert tsvc_full.validate_cap("S") == "S"
    assert tsvc_full.validate_cap("M") == "M"
    for big in ("L", "PROF", "XL"):
        assert tsvc_full.validate_cap(big) == "M"  # the O(N) oracle never runs at L/PROF/XL


# --- winner selection + table math --------------------------------------------------------------------
def make_cell(opt, lang, comp, par, cost, fp, role, ok, median, maxdiff=0.0):
    return {
        "opt_mode": opt,
        "language": lang,
        "compiler": comp,
        "parallel": par,
        "cost_model": cost,
        "fp_mode": fp,
        "role": role,
        "ok": ok,
        "maxdiff": maxdiff,
        "median_us": median,
        "min_us": median,
        "p25_us": median,
        "p75_us": median,
        "mean_us": median,
        "compile_us": 1.0,
        "error": None
    }


def test_kernel_winner_picks_fastest_validated_timing_cell():
    cells = [
        make_cell("simplify-parallel", "c", "gcc", "sequential", "default", "default-fp", "timing", True, 5.0),
        make_cell("simplify-parallel", "c", "gcc", "auto-par", "no-vec", "no-fast-errno", "timing", True, 2.0),
        make_cell("simplify-parallel", "c", "gcc", "sequential", "cheap", "default-fp", "timing", False, 1.0),  # not ok
        make_cell("simplify-parallel", "c", "gcc", "sequential", "default", "strict-ieee", "gate", True,
                  9.0),  # gate, not timing
    ]
    win = tsvc_full.kernel_winner(cells, "simplify-parallel", "c", "gcc")
    assert win["median_us"] == 2.0 and win["parallel"] == "auto-par"
    assert tsvc_full.kernel_winner(cells, "simplify-parallel", "c", "clang") is None  # no such group


def test_render_tables_gate_speedup_and_unsupported(tmp_path):
    cells = [
        # a bit-exact gate + a validated timing winner
        make_cell("simplify-parallel",
                  "c",
                  "gcc",
                  "sequential",
                  "default",
                  "strict-ieee",
                  "gate",
                  True,
                  0.0,
                  maxdiff=0.0),
        make_cell("simplify-parallel", "c", "gcc", "sequential", "default", "no-fast-errno", "timing", True, 2.0),
        # an unsupported auto-par cell (recorded)
        {
            **make_cell("simplify-parallel", "c", "clang", "auto-par", "default", "default-fp", "timing", False,
                        float("inf")), "error":
            "unsupported: clang/flang has no plain-loop auto-parallelizer"
        },
    ]
    (tmp_path / "tsvc2_sA.json").write_text(
        json.dumps({
            "key": "sA",
            "corpus": "tsvc2",
            "regime": "1d",
            "profile_preset": "PROF",
            "native": {
                "ok": True,
                "median_us": 8.0
            },
            "dace_cpp": {
                "ok": True,
                "median_us": 10.0
            },
            "cells": cells
        }))
    (tmp_path / "tsvc2_sB.json").write_text(json.dumps({"key": "sB", "corpus": "tsvc2", "skipped": "no compute nest"}))
    rep = tsvc_full.render_tables(tmp_path)
    assert "1 kernels measured, 1 skipped" in rep
    assert "strict-ieee gate:** PASS" in rep  # gate verdict keys off cell.ok, not maxdiff == 0
    assert "5.00x" in rep  # DaCe-cpp 10 / winner 2
    assert "unsupported" in rep and "auto-parallelizer" in rep
    assert "`sB` (tsvc2) — no compute nest" in rep
    assert (tmp_path / "tables.md").exists()


def test_render_tables_reports_vectorized_dace_geomean(tmp_path):
    """When a kernel carries a vectorized DaCe lane, the reporter emits the plain/vectorized geomean."""
    cells = [make_cell("simplify-parallel", "c", "gcc", "sequential", "default", "no-fast-errno", "timing", True, 2.0)]
    (tmp_path / "tsvc2_sV.json").write_text(
        json.dumps({
            "key":
            "sV",
            "corpus":
            "tsvc2",
            "regime":
            "1d",
            "profile_preset":
            "PROF",
            "native": {
                "ok": True,
                "median_us": 8.0
            },
            "dace_cpp": [{
                "ok": True,
                "median_us": 10.0,
                "codegen_impl": "experimental",
                "nest": 0
            }],
            "dace_cpp_vec": [{
                "ok": True,
                "median_us": 4.0,
                "vectorized": True,
                "vec_variant": "cpu-avx512-w16",
                "nest": 0
            }],
            "cells":
            cells
        }))
    rep = tsvc_full.render_tables(tmp_path)
    assert "plain / tile-op-vectorized speedup:** 2.500x" in rep  # 10 / 4


def kernel_json(key: str, corpus: str, dace_us: float, win_us: float) -> dict:
    """One measured kernel whose winner beats its DaCe-cpp baseline by ``dace_us / win_us``."""
    return {
        "key":
        key,
        "corpus":
        corpus,
        "regime":
        "1d",
        "profile_preset":
        "PROF",
        "native": {
            "ok": True,
            "median_us": 8.0
        },
        "dace_cpp": [{
            "ok": True,
            "median_us": dace_us,
            "codegen_impl": "experimental",
            "nest": 0
        }],
        "cells":
        [make_cell("simplify-parallel", "c", "gcc", "sequential", "default", "no-fast-errno", "timing", True, win_us)],
    }


def test_render_tables_splits_the_geomean_per_corpus(tmp_path):
    """tsvc2 and tsvc2_5 share NO kernel key, so a pooled geomean over a two-corpus sweep estimates
    neither population. The per-corpus table is what a corpus comparison reads."""
    (tmp_path / "tsvc2_sA.json").write_text(json.dumps(kernel_json("sA", "tsvc2", 10.0, 2.0)))  # 5.00x
    (tmp_path / "tsvc2_5_xB.json").write_text(json.dumps(kernel_json("xB", "tsvc2_5", 10.0, 5.0)))  # 2.00x
    rep = tsvc_full.render_tables(tmp_path)
    assert "### per corpus" in rep
    assert "| tsvc2 | best-nest/DaCe | 5.000x | 1 |" in rep
    assert "| tsvc2_5 | best-nest/DaCe | 2.000x | 1 |" in rep
    # the pooled headline stays, and is neither corpus' number: sqrt(5*2) = 3.162
    assert "**Geomean best-nest / DaCe-cpp speedup:** 3.162x" in rep


def test_render_tables_omits_the_per_corpus_table_for_one_corpus(tmp_path):
    """A single-corpus sweep has nothing to compare, and an extra table restating the headline geomean
    reads as if a second population had been measured."""
    (tmp_path / "tsvc2_sA.json").write_text(json.dumps(kernel_json("sA", "tsvc2", 10.0, 2.0)))
    assert "### per corpus" not in tsvc_full.render_tables(tmp_path)


def test_render_tables_reports_gate_failure(tmp_path):
    bad = make_cell("simplify-parallel",
                    "c",
                    "gcc",
                    "sequential",
                    "default",
                    "strict-ieee",
                    "gate",
                    False,
                    float("inf"),
                    maxdiff=3.0)
    (tmp_path / "tsvc2_sX.json").write_text(
        json.dumps({
            "key": "sX",
            "corpus": "tsvc2",
            "regime": "1d",
            "cells": [bad],
            "dace_cpp": {},
            "native": {}
        }))
    rep = tsvc_full.render_tables(tmp_path)
    assert "1 FAILURES" in rep and "| sX |" in rep  # a non-bit-exact strict cell is flagged loudly


def test_render_tables_gate_ok_with_tiny_nonzero_maxdiff_is_not_a_failure(tmp_path):
    """REGRESSION: strict-ieee is NOT atol-0 (transcendentals drift ~1e-15 from numpy), so a gate cell that
    VALIDATED (``ok=True``) with a tiny non-zero ``maxdiff`` must NOT report as a failure -- keyed off
    ``cell.ok``, never ``maxdiff == 0.0``."""
    passing = make_cell("simplify-parallel",
                        "c",
                        "gcc",
                        "sequential",
                        "default",
                        "strict-ieee",
                        "gate",
                        True,
                        4.0,
                        maxdiff=1e-16)
    (tmp_path / "tsvc2_sO.json").write_text(
        json.dumps({
            "key": "sO",
            "corpus": "tsvc2",
            "regime": "1d",
            "cells": [passing],
            "dace_cpp": {},
            "native": {}
        }))
    rep = tsvc_full.render_tables(tmp_path)
    assert "strict-ieee gate:** PASS" in rep  # ok=True with maxdiff=1e-16 is a PASS, not a FAILURE
    assert "FAILURES" not in rep
    assert "| sO |" not in rep  # the passing kernel is not listed in a (nonexistent) failure table


# --- multi-rank partition (mirror test_my_slice_disjoint_and_covers) ----------------------------------
def test_my_slice_disjoint_and_covers():
    items = list(range(29))
    for n in (1, 2, 3, 4):
        slices = [tsvc_full.my_slice(items, r, n) for r in range(n)]
        flat = [x for s in slices for x in s]
        assert sorted(flat) == items  # union == all kernels, each exactly once
        for i in range(n):
            for j in range(i + 1, n):
                assert not (set(slices[i]) & set(slices[j]))  # disjoint across ranks


# --- end-to-end (all three lanes) ---------------------------------------------------------------------
def small_axes():
    # One FP mode, not both (test_reduced_fp_modes_and_atol covers that axis): halves the compile matrix so
    # this test stays light when the 3-language matrix runs concurrently under -n auto.
    return {
        "opt_modes": ["simplify-parallel"],
        "languages": ["c", "c++", "fortran"],
        "parallelism": ["sequential", "auto-par"],
        "cost_models": ["default"],
        "fp_modes": [flags.REDUCED_FP_MODES[0]],
        "gate": True
    }


def test_run_kernel_all_lanes_s000(tmp_path):
    tcs = discover_toolchains("gcc")
    if not tcs:
        pytest.skip("no gcc")
    ftn = lang_compilers(["fortran"], tcs).get("fortran", {})
    k = tsvc.iter_tsvc_kernels(only=["s000"])[0]
    res = tsvc_full.run_kernel(k,
                               tcs,
                               ftn,
                               "skip-taskloops", {
                                   **small_axes(), "opt_mode": None
                               },
                               reps=1,
                               profile_preset="S",
                               nthreads=2,
                               cxx_std=flags.CXX_STD,
                               compile_jobs=4,
                               workdir=tmp_path)
    assert "skipped" not in res, res.get("skipped")
    # lane 1 native + lane 2 DaCe-cpp both validate bit-exact.
    assert res["native"]["ok"] and res["native"]["maxdiff"] == 0.0
    # dace_cpp is now a per-nest list (one cell per offloaded nest); every nest must bit-match + time.
    assert res["dace_cpp"] and all(c["ok"] and c["maxdiff"] == 0.0 for c in res["dace_cpp"])
    assert all(c["median_us"] != float("inf") for c in res["dace_cpp"])
    # the strict-ieee gate is bit-exact for EVERY language (incl. C++, whose entry keeps C linkage).
    gates = [c for c in res["cells"] if c["role"] == "gate" and c["error"] is None]
    langs = {c["language"] for c in gates}
    assert {"c", "c++", "fortran"} <= langs
    assert all(c["ok"] and c["maxdiff"] == 0.0 for c in gates), [g for g in gates if not g["ok"]]
    # median timing populated for sequential AND auto-par (gcc has a real auto-parallelizer).
    seq = [c for c in res["cells"] if c["role"] == "timing" and c["parallel"] == "sequential" and c["ok"]]
    par = [
        c for c in res["cells"]
        if c["role"] == "timing" and c["parallel"] == "auto-par" and c["ok"] and c["compiler"] == "gcc"
    ]
    assert seq and all(c["median_us"] != float("inf") for c in seq)
    assert par and all(c["median_us"] != float("inf") for c in par)


def test_omp_emit_lane_runs_for_parallel_nest_s000(tmp_path):
    """s000 (elementwise ``a[i]=b[i]+1``) is DaCe-parallel, so it MUST get the omp-emit lane: OUR ``#pragma
    omp parallel for`` source, validated bit-exact and timed across C/C++/Fortran, incl. clang."""
    tcs = discover_toolchains("auto")
    if not tcs:
        pytest.skip("no toolchain")
    ftn = lang_compilers(["fortran"], tcs).get("fortran", {})
    k = tsvc.iter_tsvc_kernels(only=["s000"])[0]
    axes = {
        "opt_modes": ["simplify-parallel"],
        "languages": ["c", "c++", "fortran"],
        "parallelism": ["sequential", "omp-emit"],
        "cost_models": ["default"],
        "fp_modes": ["default-fp"],
        "gate": False,
        "opt_mode": None,
    }
    res = tsvc_full.run_kernel(k,
                               tcs,
                               ftn,
                               "skip-taskloops",
                               axes,
                               reps=1,
                               profile_preset="S",
                               nthreads=2,
                               cxx_std=flags.CXX_STD,
                               compile_jobs=4,
                               workdir=tmp_path)
    assert "skipped" not in res, res.get("skipped")
    omp = [c for c in res["cells"] if c["parallel"] == "omp-emit" and c["role"] == "timing"]
    assert omp, "s000 is parallel -> the omp-emit lane must produce cells"
    ok = [c for c in omp if c["ok"]]
    assert ok and all(c["maxdiff"] == 0.0 for c in ok)  # OpenMP source is bit-exact vs the oracle
    assert all(c["median_us"] != float("inf") for c in ok)  # and timed
    assert {"c", "c++", "fortran"} <= {c["language"] for c in ok}  # every language got a working omp lane


def test_omp_emit_sources_carry_the_pragma(tmp_path):
    """The omp-emit source numpyto produces for a parallel nest contains the OpenMP pragma under the SAME
    symbol as the sequential emit (a drop-in); a sequential nest produces no omp source."""
    tcs = discover_toolchains("gcc")
    if not tcs:
        pytest.skip("no gcc")
    k = tsvc.iter_tsvc_kernels(only=["s000"])[0]
    ctxs = tsvc_full.build_opt_context(k, "simplify-parallel", "skip-taskloops", "S", ["c", "fortran"], tmp_path)
    nc = ctxs[0]
    assert nc["parallel"], "s000 nest should be classified parallel"
    assert nc["omp_src"], "a parallel nest must have omp sources"
    csrc = nc["omp_src"]["c"][0].read_text()
    assert "#pragma omp parallel for" in csrc and nc["symbol"] in csrc
    fsrc = nc["omp_src"]["fortran"][0].read_text()
    assert "!$omp parallel do" in fsrc


def test_cxx_lane_symbol_is_c_abi_unmangled(tmp_path):
    """The C++ lane recompiles the C source, so the entry MUST stay C-ABI (unmangled) or ctypes -- and any
    whole-program link -- cannot resolve it."""
    tcs = discover_toolchains("gcc")
    if not tcs or shutil.which("nm") is None:
        pytest.skip("no gcc / nm")
    k = tsvc.iter_tsvc_kernels(only=["s000"])[0]
    # build_opt_context returns a per-nest list; s000 is single-nest -> nest 0, symbol suffixed ``_n0_fp64``.
    ctxs = tsvc_full.build_opt_context(k, "simplify-parallel", "skip-taskloops", "S", ["c", "c++"], tmp_path)
    wrapper, order, argtypes = ctxs[0]["lang_src"]["c++"]
    symbol = ctxs[0]["symbol"]  # s000_n0_fp64
    cflags, _ = flags.lane_flags("gnu", "default-fp", "default", "sequential", "c++", 2)
    so = tmp_path / "s000_cxx.so"
    subprocess.run(["g++", *cflags, str(wrapper), "-o", str(so)], check=True, capture_output=True)
    nm = subprocess.run(["nm", "-D", str(so)], capture_output=True, text=True).stdout
    sym_lines = [ln for ln in nm.splitlines() if symbol in ln]
    assert any(ln.strip().endswith(f" {symbol}") and " T " in ln for ln in sym_lines), sym_lines  # unmangled, defined
    assert not any("_Z" in ln and symbol in ln for ln in sym_lines)  # NOT C++-mangled
    assert ctypes.CDLL(str(so))[symbol]  # ctypes binds it


def test_cxx_only_run_keeps_the_veclib_axis(tmp_path):
    """A ``--languages c++`` run (no ``c``) must still gate the veclib axis ON for a nest whose emitted C
    source calls libm (the C++ lane compiles that source via its ``extern "C"`` wrapper): the gate must not
    be read off a ``c`` entry a c++-only run never produces, nor silently collapse to ``none``."""
    k = tsvc.iter_tsvc_kernels(only=["s451"])[0]
    ctxs = tsvc_full.build_opt_context(k, "simplify-parallel", "skip-taskloops", "S", ["c++"], tmp_path)
    nc = ctxs[0]
    assert "c" not in nc["lang_src"], "c++-only run must not carry a c lane to inherit the gate from"
    assert nc["has_math"]["c++"], "sin() in the emitted C source must gate the veclib axis ON for c++"
    assert tsvc_full.veclibs_for(nc["has_math"]["c++"], ("none", "svml"), "icx") == ("none", "svml")


def one_lang_axes():
    return {
        "opt_modes": ["simplify-parallel"],
        "languages": ["c"],
        "parallelism": ["sequential"],
        "cost_models": ["default"],
        "fp_modes": ["default-fp"],
        "gate": False,
        "opt_mode": None
    }


def test_dace_baseline_timing_always_produced_recurrence(tmp_path):
    """s111 is a linear recurrence: nest extraction promotes its loop-carried state to boundary I/O, so
    DaCe's raw codegen of the standalone nest diverges from numpyto's semantics. Same iteration space
    though, so its MEDIAN TIME must still be produced even without a bit-match (the gate is the real check)."""
    tcs = discover_toolchains("gcc")
    if not tcs:
        pytest.skip("no gcc")
    ftn = lang_compilers(["fortran"], tcs).get("fortran", {})
    k = tsvc.iter_tsvc_kernels(only=["s111"])[0]
    res = tsvc_full.run_kernel(k,
                               tcs,
                               ftn,
                               "skip-taskloops",
                               one_lang_axes(),
                               reps=2,
                               profile_preset="S",
                               nthreads=2,
                               cxx_std=flags.CXX_STD,
                               compile_jobs=2,
                               workdir=tmp_path)
    assert "skipped" not in res, res.get("skipped")
    # per-nest list: every nest's DaCe-cpp timing baseline is produced regardless of bit-match.
    assert res["dace_cpp"] and all(c["median_us"] != float("inf") for c in res["dace_cpp"])
    assert all(c.get("error") is None for c in res["dace_cpp"])  # ran; ok may be False (recorded, not a crash)


def test_dace_baseline_validates_for_2d_inner_nest(tmp_path):
    """s1115 is peeled to an INNER nest (outer index fixed): the standalone DaCe lane does the same one-row
    work as the nest-forge lanes, so it bit-matches and its time stays apples-to-apples (a whole-kernel SDFG
    would do ~LEN more work and inflate the speedup)."""
    tcs = discover_toolchains("gcc")
    if not tcs:
        pytest.skip("no gcc")
    ftn = lang_compilers(["fortran"], tcs).get("fortran", {})
    k = tsvc.iter_tsvc_kernels(only=["s1115"])[0]
    res = tsvc_full.run_kernel(k,
                               tcs,
                               ftn,
                               "skip-taskloops",
                               one_lang_axes(),
                               reps=2,
                               profile_preset="S",
                               nthreads=2,
                               cxx_std=flags.CXX_STD,
                               compile_jobs=2,
                               workdir=tmp_path)
    assert "skipped" not in res, res.get("skipped")
    assert res["dace_cpp"] and all(c["ok"] and c["maxdiff"] == 0.0 for c in res["dace_cpp"])


class FakeGenerated:
    """Stands in for build.GeneratedProgram: only ``source`` is read before the compile screen."""

    def __init__(self, source: str):
        self.source = source
        self.name = "fake"
        self.codegen_seconds = 0.0


def test_the_vectorize_lane_times_one_artifact_once(monkeypatch, tmp_path):
    """The descent compiles a cell per config, and a nest the vectorizer declines to touch emits the SAME
    object for every one of them -- `resolved_key` cannot see that, because the configs genuinely differ.
    Without the artifact key the lane re-times one build a dozen times and reports the spread as a
    measured width/remainder preference.
    """
    import dace

    from nestforge.toolchain import Toolchain

    N = dace.symbol("N")

    @dace.program
    def vadd(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
        c[:] = a + b

    class FakeBoundary:
        standalone_sdfg = vadd.to_sdfg(simplify=True)

    class FakeBuilt:
        name = "vadd"
        so_path = tmp_path / "inert.so"

        def unload(self):
            pass

    runs, compiles = {"n": 0}, {"n": 0}
    keyed: List[Optional[str]] = []

    def fake_run(*a, **k):
        runs["n"] += 1
        return {
            "ok": True,
            "maxdiff": 0.0,
            "median_us": 4.0,
            "min_us": 4.0,
            "p25_us": 4.0,
            "p75_us": 4.0,
            "mean_us": 4.0
        }

    def fake_compile(gen, opts=None):
        compiles["n"] += 1
        return FakeBuilt()

    monkeypatch.setattr(tsvc_full, "generate_program", lambda *a, **k: FakeGenerated("void __program_vadd() {}"))
    monkeypatch.setattr(tsvc_full, "compile_program", fake_compile)
    monkeypatch.setattr(tsvc_full, "run_isolated", fake_run)

    def fake_key(so, symbol=None):
        keyed.append(symbol)
        return "one-and-only-artifact"

    monkeypatch.setattr(tsvc_full, "variant_key", fake_key)

    tc = Toolchain(name="gcc", cc="gcc", cxx="g++", version=(13, 0), source="path")
    cell = tsvc_full.measure_dace_vectorized_lane(tc,
                                                  FakeBoundary(), {"N": 8}, {}, {"N": 8}, {},
                                                  reps=1,
                                                  cxx_std="c++20",
                                                  workdir=tmp_path)
    assert cell["ok"] and cell["median_us"] == 4.0
    assert runs["n"] == 1, f"one artifact, {runs['n']} timed runs -- the artifact dedup is not wired in"
    # the SOURCE screen must fire FIRST, so an inert nest costs ONE compile and not one per config -- that
    # is the whole point of screening before the compile rather than after it
    assert compiles["n"] == 1, f"identical source, {compiles['n']} compiles -- the codegen screen runs too late"
    # the WHOLE object, never `__program_<name>`: that entry point is a trampoline that disassembles the
    # same for every config, so keying on it collapsed every cell onto the first and the descent measured
    # nothing. init/exit belong in the key -- init allocates the persistent storage a config can change.
    assert set(keyed) == {None}, keyed


def test_the_vectorize_lane_still_times_distinct_artifacts(monkeypatch, tmp_path):
    """The other direction: a key that differs per build must NOT collapse, or the lane would report one
    config's time for every cell and the search would be measuring nothing."""
    import dace

    from nestforge.toolchain import Toolchain

    N = dace.symbol("N")

    @dace.program
    def vadd2(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
        c[:] = a + b

    class FakeBoundary:
        standalone_sdfg = vadd2.to_sdfg(simplify=True)

    class FakeBuilt:
        name = "vadd2"
        so_path = tmp_path / "distinct.so"

        def unload(self):
            pass

    runs = {"n": 0}

    def fake_run(*a, **k):
        runs["n"] += 1
        return {
            "ok": True,
            "maxdiff": 0.0,
            "median_us": float(10 - runs["n"]),
            "min_us": 1.0,
            "p25_us": 1.0,
            "p75_us": 1.0,
            "mean_us": 1.0
        }

    sources = {"n": 0}

    def fake_generate(*a, **k):
        sources["n"] += 1
        return FakeGenerated(f"void __program_vadd2() {{ int x = {sources['n']}; }}")

    monkeypatch.setattr(tsvc_full, "generate_program", fake_generate)
    monkeypatch.setattr(tsvc_full, "compile_program", lambda gen, opts=None: FakeBuilt())
    monkeypatch.setattr(tsvc_full, "run_isolated", fake_run)
    monkeypatch.setattr(tsvc_full, "variant_key", lambda so, symbol=None: f"artifact-{runs['n']}")

    tc = Toolchain(name="gcc", cc="gcc", cxx="g++", version=(13, 0), source="path")
    tsvc_full.measure_dace_vectorized_lane(tc,
                                           FakeBoundary(), {"N": 8}, {}, {"N": 8}, {},
                                           reps=1,
                                           cxx_std="c++20",
                                           workdir=tmp_path)
    assert runs["n"] > 1, "every artifact differed, so every cell had to be timed"
