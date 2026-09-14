# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Parallel loop-nest coverage: the seq/parallel classifier + nested-map emission.

``is_parallel_nest`` routes each extracted nest to the OpenMP vs serial emit; the
nested-map recursion in ``map_lines`` lets a map-inside-a-map (e.g. tsvc_2_s2275's 2-D
FMA) emit as nested ``for`` loops instead of raising. Both are pure/emit-level -- no compile.
"""
import numpy as np

import dace
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize import canonicalize

from nestforge.corpus.bench import iter_dace_kernels
from nestforge.ir.extract import extract_nest_to_sdfg
from nestforge.ir.emit_numpy import load_emitted, sdfg_to_numpy
from nestforge.phases.scopes import is_parallel_nest, parallel_top_level_maps


def load(key):
    for kernel in iter_dace_kernels("loop_level_reasoning"):
        if kernel.short_name.rsplit("/", 1)[-1] == key:
            return kernel
    raise AssertionError(f"{key} is not in the loop_level_reasoning track -- the corpus this test pins has changed")


def nest_refs(key):
    sdfg = load(key).to_sdfg(simplify=True)
    canonicalize(sdfg, target="cpu")
    return sdfg, parallel_top_level_maps(sdfg)


# --- is_parallel_nest ------------------------------------------------------------------------------
def test_map_schedule_drives_parallel_classification():
    sdfg = dace.SDFG("t")
    sdfg.add_array("a", [10], dace.float64)
    state = sdfg.add_state()
    entry, _ = state.add_map("m", dict(i="0:10"))
    entry.map.schedule = dace.ScheduleType.Sequential
    assert not is_parallel_nest(entry)
    entry.map.schedule = dace.ScheduleType.Default
    assert is_parallel_nest(entry)


def test_loop_region_is_sequential():
    # A LoopRegion is a loop LoopToMap refused (a recurrence) -> sequential.
    assert not is_parallel_nest(LoopRegion("l"))


def test_real_parallel_map_kernel_is_parallel():
    _, refs = nest_refs("tsvc_2_s2275")
    assert refs
    assert any(is_parallel_nest(node) for _, node in refs)


# --- nested-map emission (map_lines recursion) ----------------------------------------------------
def test_s2275_nested_map_emits_and_computes():
    # tsvc_2_s2275 baseline = an i-loop with an inner j-loop (2-D aa FMA) + an i-level 1-D statement.
    # Canonicalization legally DISTRIBUTES the two (the yaml puzzle: interchange for the matrix update
    # is legal only once the vector statement is out of the i loop), so phase 2 sees two top-level
    # parallel maps; the 2-D aa update is the one that exercises map_lines' nested-for recursion.
    _, refs = nest_refs("tsvc_2_s2275")
    assert len(refs) == 2
    boundary = extract_nest_to_sdfg(refs[0][0], refs[0][1], name="s2275_aa")
    src = sdfg_to_numpy(boundary.standalone_sdfg, "s2275_aa")
    assert src.count("for ") >= 2  # nested for-loops emitted, not the old UnsupportedNest raise

    kernel = load_emitted(src, "s2275_aa").s2275_aa

    n = 6
    rng = np.random.default_rng(0)
    aa, bb, cc = (rng.random((n, n)) for _ in range(3))
    aa_ref = aa + bb * cc  # a fresh array; the in-place kernel below must reproduce it
    kernel(aa=aa, bb=bb, cc=cc, LEN_2D=n)
    assert np.allclose(aa, aa_ref)
