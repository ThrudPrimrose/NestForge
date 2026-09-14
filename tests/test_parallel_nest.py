# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Parallel loop-nest coverage: the seq/parallel classifier + nested-map emission.

``is_parallel_nest`` routes each extracted nest to the OpenMP vs serial emit; the
nested-map recursion in ``map_lines`` lets a map-inside-a-map (e.g. s2275) emit as
nested ``for`` loops instead of raising. Both are pure/emit-level -- no compile.
"""
import numpy as np

import dace
from dace.sdfg.state import LoopRegion

from nestforge.corpus import tsvc
from nestforge.ir.extract import extract_nest_to_sdfg
from nestforge.ir.emit_numpy import load_emitted, sdfg_to_numpy
from nestforge.phases.scopes import get_strategy, is_parallel_nest


def nest_refs(key, opt_mode="simplify-parallel", strategy="skip-taskloops"):
    kernel = tsvc.iter_tsvc_kernels(only=[key], corpus="tsvc2")[0]
    sdfg = tsvc.build_sdfg(kernel, opt_mode=opt_mode)
    return sdfg, get_strategy(strategy)(sdfg)


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
    _, refs = nest_refs("s2275")
    assert refs
    assert any(is_parallel_nest(node) for _, node in refs)


# --- nested-map emission (map_lines recursion) ----------------------------------------------------
def test_s2275_nested_map_emits_and_computes():
    # s2275 baseline = outer i-loop with an inner j-loop (2-D aa FMA) + an i-level 1-D statement.
    # The 'outer' strategy offloads the whole parent nest; map_lines must recurse to nested for-loops.
    _, refs = nest_refs("s2275", strategy="outer")
    assert len(refs) == 1
    boundary = extract_nest_to_sdfg(refs[0][0], refs[0][1], name="s2275")
    src = sdfg_to_numpy(boundary.standalone_sdfg, "s2275")
    assert src.count("for ") >= 2  # nested for-loops emitted, not the old UnsupportedNest raise

    kernel = load_emitted(src, "s2275").s2275

    n = 6
    rng = np.random.default_rng(0)
    a, b, c, d = (rng.random(n) for _ in range(4))
    aa, bb, cc = (rng.random((n, n)) for _ in range(3))
    a_ref = b + c * d
    aa_ref = aa + bb * cc  # a fresh array; the in-place kernel below must reproduce it
    kernel(a=a, aa=aa, b=b, bb=bb, c=c, cc=cc, d=d, LEN_2D=n)
    assert np.allclose(a, a_ref)
    assert np.allclose(aa, aa_ref)
