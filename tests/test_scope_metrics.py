# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Scope metrics: the work, depth, bytes moved and operational intensity a scheduling agent reads per scope."""

import re

import dace
import numpy as np
import sympy

from dace.sdfg.state import LoopRegion
from nestforge.phases.schedule import ScopeMetrics, scope_metrics
from nestforge.phases.scopes import top_level_map_entries
from nestforge.session import Session

N = dace.symbol("N")
TSTEPS = dace.symbol("TSTEPS")


@dace.program
def vector_add(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N]:
        C[i] = A[i] + B[i]


@dace.program
def stencil_sweeps(A: dace.float64[N], B: dace.float64[N]):
    for t in range(TSTEPS):
        for i in dace.map[1 : N - 1]:
            B[i] = A[i - 1] + A[i] + A[i + 1]


@dace.program
def vertical_pair(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    T = np.empty_like(A)
    for i in dace.map[0:N]:
        T[i] = A[i] + B[i]
    for i in dace.map[0:N]:
        C[i] = T[i] * 2.0


def top_level_maps(sdfg: dace.SDFG):
    return [entry for state in sdfg.all_states() for entry in top_level_map_entries(state)]


def assert_symbolic(actual: sympy.Expr, expected: sympy.Expr) -> None:
    assert dace.symbolic.equal(actual, expected) is True, f"{actual} != {expected}"


def test_a_vector_add_does_one_operation_per_element_over_three_arrays():
    sdfg = vector_add.to_sdfg(simplify=True)
    (entry,) = top_level_maps(sdfg)
    metrics = scope_metrics(sdfg, entry)
    assert isinstance(metrics, ScopeMetrics)
    assert_symbolic(metrics.work, N)
    assert_symbolic(metrics.depth, 1)
    assert_symbolic(metrics.bytes, 24 * N)
    assert metrics.oi == sympy.Rational(1, 24)


def test_a_loop_around_a_stencil_map_scales_work_and_bytes_by_the_sweeps_but_not_oi():
    sdfg = stencil_sweeps.to_sdfg(simplify=True)
    (loop,) = [block for block in sdfg.nodes() if isinstance(block, LoopRegion)]
    (entry,) = top_level_maps(sdfg)
    assert entry in [node for state in loop.all_states() for node in state.nodes()]
    in_map = scope_metrics(sdfg, entry)
    in_loop = scope_metrics(sdfg, loop)
    assert_symbolic(in_map.work, 2 * N - 4)
    assert_symbolic(in_map.depth, 2)
    assert_symbolic(in_map.bytes, 16 * N - 16)
    assert_symbolic(in_loop.work, TSTEPS * (2 * N - 4))
    assert_symbolic(in_loop.depth, 2 * TSTEPS)
    assert_symbolic(in_loop.bytes, TSTEPS * (16 * N - 16))
    assert_symbolic(in_map.oi, (N - 2) / (8 * N - 8))
    assert_symbolic(in_loop.oi, in_map.oi)


def test_fusing_a_producer_into_its_consumer_raises_oi_above_both_parts():
    """The fused scope no longer moves the intermediate T, so its OI beats either unfused map."""
    session = Session(vertical_pair.to_sdfg(simplify=True))
    producer, consumer = top_level_maps(session.sdfg)
    unfused = [scope_metrics(session.sdfg, entry) for entry in (producer, consumer)]
    assert [m.oi for m in unfused] == [sympy.Rational(1, 24), sympy.Rational(1, 16)]
    (move,) = session.list_fusions()
    session.fuse(move["id"])
    (fused_entry,) = top_level_maps(session.sdfg)
    fused = scope_metrics(session.sdfg, fused_entry)
    assert_symbolic(fused.work, 2 * N)
    assert_symbolic(fused.bytes, 24 * N)
    assert fused.oi == sympy.Rational(1, 12)


def test_scope_metrics_leaves_the_program_untouched():
    sdfg = stencil_sweeps.to_sdfg(simplify=True)
    before = sdfg.to_json()
    (loop,) = [block for block in sdfg.nodes() if isinstance(block, LoopRegion)]
    scope_metrics(sdfg, loop)
    scope_metrics(sdfg, top_level_maps(sdfg)[0])
    assert sdfg.to_json() == before


def test_describe_appends_metrics_to_kernel_lines_only_when_asked():
    session = Session(vector_add.to_sdfg(simplify=True))
    plain = session.describe()
    with_metrics = session.describe(metrics=True)
    assert "work=" not in plain
    (kernel,) = [line for line in with_metrics.splitlines() if "reads=" in line]
    assert kernel.endswith("  work=N depth=1 bytes=24*N OI=0.04167"), kernel
    strip = lambda tree: re.sub(r"\[e\d+:nest:\d+\]", "[nest]", tree)
    assert strip(with_metrics.replace("  work=N depth=1 bytes=24*N OI=0.04167", "")) == strip(plain)
    assert session.epoch == 0
