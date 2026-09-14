# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The configuration sweep: what it enumerates, how a build shared by several cells is measured once and gated per
cell, and that the winning ``lib<kernel>.a`` links into the parent program and computes the right answer."""
import subprocess

import numpy as np
import pytest

import dace

from nestforge.build import flags
from nestforge.build.toolchain import Toolchain
from nestforge.corpus.translate import prepare
from nestforge.ir.libnode import ExternLibEnv
from nestforge.phases.kernel import KernelVerdict, at_rung, schedule_kernel, use_kernel_library
from nestforge.phases.normalize import Targets, normalize
from nestforge.phases.schedule import full_fusion
from nestforge.phases.scopes import lower_nests_to_external_call
from nestforge.phases.variants import enumerate_variants, select_variant

N = dace.symbol("N")


@dace.program
def vadd(b: dace.float64[N], c: dace.float64[N], a: dace.float64[N]):
    for i in dace.map[0:N]:
        a[i] = b[i] + c[i]


GCC = Toolchain(name="gcc", cc="gcc", cxx="g++", source="path")
CLANG = Toolchain(name="clang", cc="clang", cxx="clang++", source="path")


def lowered_vadd():
    sdfg = vadd.to_sdfg(simplify=True)
    normalize(sdfg, Targets())
    full_fusion(sdfg, Targets())
    (ext, boundary), = lower_nests_to_external_call(sdfg)
    return sdfg, ext, boundary


def gcc_variants(**axes):
    """The g++ cells whose axes equal ``axes``, in sweep order."""
    return [v for v in enumerate_variants([GCC]) if all(vars_of(v)[k] == value for k, value in axes.items())]


def vars_of(variant):
    return {"fp_mode": variant.fp_mode, "cost_model": variant.cost_model, "veclib": variant.veclib}


def test_gcc_variants_cross_every_fp_rung_with_every_cost_model():
    variants = enumerate_variants([GCC])

    unvectorized = [v for v in variants if v.veclib == "none"]
    assert {(v.fp_mode, v.cost_model)
            for v in unvectorized} == {(f, c)
                                       for f in flags.FP_LEVELS
                                       for c in flags.COST_MODELS}
    assert len(unvectorized) == len(flags.FP_LEVELS) * len(flags.COST_MODELS)
    assert all(v.compiler == "g++" for v in variants)
    assert {v.veclib for v in variants} <= {"none", "sleef", "libmvec"}, "gcc never emits __svml_* calls"
    assert len({v.label for v in variants}) == len(variants)
    for v in variants:
        assert ("-fno-tree-vectorize" in v.flags) == (v.cost_model == "no-vec"), v.label
        assert ("-ffast-math" in v.flags) == (v.fp_mode == "fast-math"), v.label
        assert ("-ffp-contract=off" in v.flags) == (v.fp_mode == "strict-ieee"), v.label


def test_a_cost_model_the_family_has_no_knob_for_is_not_a_second_build():
    """clang has no ``cheap`` cost model, so ``cheap`` composes ``default``'s flags and must not be built twice."""
    variants = enumerate_variants([CLANG])

    assert {v.cost_model for v in variants} == {"default", "no-vec"}
    assert len([v for v in variants if v.veclib == "none"]) == len(flags.FP_LEVELS) * 2
    assert all("-fveclib" not in " ".join(v.flags) for v in variants), "the veclib reaches the build, not the flags"


def test_a_toolchain_without_a_cxx_compiler_contributes_no_variant():
    assert enumerate_variants([Toolchain(name="gcc", cc="gcc", cxx=None, source="path")]) == []


def test_a_shared_measurement_is_gated_at_each_cells_own_rung():
    """A build shared by a strict and a looser cell is timed once, but ``ok`` must be read at each cell's
    rung: inheriting the strict verdict would fail a looser cell whose build is correct at its tolerance."""
    strict = KernelVerdict("strict-ieee", maxdiff=1e-14, md_rel=1e-14, dtype_floor=2.3e-16, time_us=5.0)

    loose = at_rung(strict, "contract-fma")

    assert not strict.ok and loose.ok
    assert (loose.maxdiff, loose.time_us) == (strict.maxdiff, strict.time_us)
    assert not at_rung(KernelVerdict("fast-math", 0.0, 0.0, 0.0, 1.0, error="crashed"), "fast-math").ok


@pytest.mark.e2e
def test_the_sweep_measures_identical_builds_once_and_the_fastest_correct_build_wins(tmp_path):
    """A pure add reaches no FP rung below fast-math, so strict and contract-fma compile to one artifact:
    every cell is still reported, the twin carries the measured numbers, and the winner is correct."""
    _, ext, boundary = lowered_vadd()
    src = schedule_kernel(ext, boundary, Targets(), tmp_path / "gen")
    prep = prepare(boundary, ext.name, tmp_path / "ref")
    variants = gcc_variants(cost_model="default", veclib="none")

    result = select_variant(src, prep, {"N": 1037}, 3, variants, tmp_path / "variants")

    assert len(result.cells) == len(variants) == len(flags.FP_LEVELS)
    assert all(cell.verdict.error == "" for cell in result.cells), [c.verdict.error for c in result.cells]
    assert result.collapsed, "strict-ieee and contract-fma build the same vadd; nothing collapsed means dedup is inert"
    ids = {f"{i}:{cell.variant.label}": cell for i, cell in enumerate(result.cells)}
    twins = [cell for cell in result.cells if cell.same_as]
    assert twins and len(result.collapsed) == len({cell.same_as for cell in twins})
    for twin in twins:
        head = ids[twin.same_as]
        assert (twin.verdict.maxdiff, twin.verdict.time_us) == (head.verdict.maxdiff, head.verdict.time_us)
    strict = next(cell for cell in result.cells if cell.variant.fp_mode == "strict-ieee")
    assert strict.verdict.ok and strict.verdict.maxdiff == 0.0
    correct = [cell for cell in result.cells if cell.verdict.ok]
    assert result.winner is not None and result.winner.verdict.time_us == min(c.verdict.time_us for c in correct)
    assert result.library == result.winner.archive and result.library.name == f"lib{ext.name}.a"
    assert (result.symbol, result.abi_order) == (src.symbol, src.abi_order)


def openmp_runtimes(so_path):
    """OpenMP runtimes the built parent library pulls in, per ``ldd``, deduped by soname."""
    out = subprocess.check_output(["ldd", so_path], text=True)
    return {line.split()[0] for line in out.splitlines() if any(t in line for t in ("libgomp", "libomp", "libiomp"))}


@pytest.mark.e2e
def test_the_winning_archive_links_statically_into_the_parent_and_matches_numpy(tmp_path):
    """The whole flow: the phase-4 winner linked into the parent through ``ExternalCall``'s extern-call
    expansion. An archive carries no runtime, so the parent ends with at most one OpenMP runtime."""
    sdfg, ext, boundary = lowered_vadd()
    src = schedule_kernel(ext, boundary, Targets(), tmp_path / "gen")
    prep = prepare(boundary, ext.name, tmp_path / "ref")
    result = select_variant(src, prep, {"N": 257}, 1,
                            gcc_variants(fp_mode="strict-ieee", cost_model="default", veclib="none"),
                            tmp_path / "variants")
    assert result.library is not None, [c.verdict.error for c in result.cells]

    use_kernel_library(ext, result.library, result.symbol, result.abi_order)
    sdfg.expand_library_nodes()
    sdfg.validate()
    sdfg.build_folder = str(tmp_path / "parent")
    compiled = sdfg.compile()
    n = 4099
    b, c = np.random.default_rng(0).random(n), np.random.default_rng(1).random(n)
    a = np.zeros(n)
    compiled(b=b, c=c, a=a, N=n)

    np.testing.assert_array_equal(a, b + c)
    assert str(result.library) in ExternLibEnv.cmake_libraries
    assert not any("-rpath" in f for f in ExternLibEnv.cmake_link_flags), "statically in, not loaded"
    assert len(openmp_runtimes(str(compiled._lib._library_filename))) <= 1
