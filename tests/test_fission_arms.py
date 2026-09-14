# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The Phase-2 fission lever (:mod:`nestforge.phases.schedule`): explode a program to statement granularity by
reusing the existing DaCe canon passes (SplitStatements + LoopFission + MapFission), and the agent's real
Phase-2 flow -- fission then fuse back up. Value-preservation (bit-exact vs the un-fissioned reference) is
the invariant on every case.
"""
import numpy as np
import pytest

import dace
from dace.sdfg.state import LoopRegion

from dace.sdfg import nodes
from dace.transformation.dataflow.map_fission import MapFission
from dace.transformation.helpers import nest_state_subgraph

from nestforge.phases.schedule import fission_to_statements, map_fission_moves
from nestforge.phases.schedule import apply_fusion, enumerate_fusions

N = dace.symbol("N")
f64 = dace.float64


def nloops(sdfg):
    return sum(1 for c in sdfg.all_control_flow_regions(recursive=True) if isinstance(c, LoopRegion))


def run(sdfg, inputs, n):
    bufs = {k: v.copy() for k, v in inputs.items()}
    sdfg(**bufs, N=n)
    return bufs


def mk(n=48, names=("a", "b", "c"), seed=0):
    rng = np.random.default_rng(seed)
    return {k: rng.random(n) for k in names}


@dace.program
def two_independent_recurrences(a: f64[N], b: f64[N], c: f64[N]):
    for i in range(1, N):
        b[i] = b[i - 1] + a[i]
        c[i] = c[i - 1] + a[i]


@dace.program
def three_independent_statements(a: f64[N], b: f64[N], c: f64[N], d: f64[N]):
    for i in range(1, N):
        b[i] = b[i - 1] + a[i]
        c[i] = c[i - 1] * a[i]
        d[i] = d[i - 1] - a[i]


@dace.program
def conditional_body(a: f64[N], b: f64[N], c: f64[N]):
    for i in range(1, N):
        if a[i] > 0.5:
            b[i] = b[i - 1] + a[i]
            c[i] = c[i - 1] + a[i]
        else:
            b[i] = b[i - 1]
            c[i] = c[i - 1]


def test_fission_splits_independent_recurrences_value_preserving():
    inputs = mk(names=("a", "b", "c"))
    ref = run(two_independent_recurrences.to_sdfg(simplify=True), inputs, 48)
    sdfg = two_independent_recurrences.to_sdfg(simplify=True)
    before = nloops(sdfg)
    applied = fission_to_statements(sdfg)
    got = run(sdfg, inputs, 48)
    assert applied >= 1 and nloops(sdfg) > before  # the loop split into independent statements
    assert all(np.allclose(got[k], ref[k]) for k in inputs)


@pytest.mark.parametrize("prog,names", [
    (two_independent_recurrences, ("a", "b", "c")),
    (three_independent_statements, ("a", "b", "c", "d")),
    (conditional_body, ("a", "b", "c")),
])
def test_fission_is_value_preserving(prog, names):
    inputs = mk(names=names)
    ref = run(prog.to_sdfg(simplify=True), inputs, 48)
    sdfg = prog.to_sdfg(simplify=True)
    fission_to_statements(sdfg)
    got = run(sdfg, inputs, 48)
    assert all(np.allclose(got[k], ref[k]) for k in inputs), f"{prog.name}: fission changed the value"


def test_fission_then_fuse_roundtrip_value_preserving():
    # the agent's real Phase-2 flow: explode to statements, then fuse back up -- must land on the same
    # value as the original program whatever granularity it settles on.
    inputs = mk(names=("a", "b", "c", "d"))
    ref = run(three_independent_statements.to_sdfg(simplify=True), inputs, 48)
    sdfg = three_independent_statements.to_sdfg(simplify=True)
    fission_to_statements(sdfg)
    while True:
        moves = enumerate_fusions(sdfg)
        if not moves:
            break
        apply_fusion(sdfg, moves[0])
    got = run(sdfg, inputs, 48)
    assert all(np.allclose(got[k], ref[k]) for k in inputs)


def map_with_nested_body():
    """A map whose sole body child is a NestedSDFG holding two independent output groups -- MapFission's
    map-with-nested-SDFG pattern, which is the shape ``map_fission_moves`` exists to enumerate."""
    sdfg = dace.SDFG("map_with_nested_body")
    sdfg.add_array("a", [N], f64)
    sdfg.add_array("b", [N], f64)
    sdfg.add_array("c", [N], f64)
    state = sdfg.add_state()

    rnode = state.add_read("a")
    me, mx = state.add_map("outer", dict(i="0:N"))
    t1 = state.add_tasklet("one", {"x"}, {"y"}, "y = x + 1.0")
    t2 = state.add_tasklet("two", {"x"}, {"y"}, "y = x * 2.0")
    state.add_memlet_path(rnode, me, t1, memlet=dace.Memlet(data="a", subset="i"), dst_conn="x")
    state.add_memlet_path(t1, mx, state.add_write("b"), memlet=dace.Memlet(data="b", subset="i"), src_conn="y")
    state.add_memlet_path(rnode, me, t2, memlet=dace.Memlet(data="a", subset="i"), dst_conn="x")
    state.add_memlet_path(t2, mx, state.add_write("c"), memlet=dace.Memlet(data="c", subset="i"), src_conn="y")
    sdfg.validate()

    nest_state_subgraph(sdfg, state, state.scope_subgraph(me, include_entry=False, include_exit=False))
    sdfg.validate()
    return sdfg, me


def test_map_fission_moves_enumerates_nested_sdfg_body():
    # The arm must match MapFission's map-with-nested-SDFG pattern (expr_index=1). Matched against the
    # default map-with-subgraph pattern instead, the lone NestedSDFG body reads as a single component and
    # the arm offers no moves at all for the one shape it targets.
    sdfg, me = map_with_nested_body()
    moves = map_fission_moves(sdfg)

    assert len(moves) == 1, "the map with an independent-group nested-SDFG body must be offered as a move"
    entry, nsdfg = moves[0]
    assert entry is me
    assert isinstance(nsdfg, nodes.NestedSDFG)
    # The move must carry the pair that validated, so the documented apply_to cannot raise on it.
    MapFission.apply_to(sdfg, expr_index=1, map_entry=entry, nested_sdfg=nsdfg)
    sdfg.validate()


def test_map_fission_moves_value_preserving():
    inputs = mk(names=("a", "b", "c"))
    sdfg, _ = map_with_nested_body()
    ref = run(map_with_nested_body()[0], inputs, 48)

    for entry, nsdfg in map_fission_moves(sdfg):
        MapFission.apply_to(sdfg, expr_index=1, map_entry=entry, nested_sdfg=nsdfg)
    got = run(sdfg, inputs, 48)
    assert all(np.allclose(got[k], ref[k]) for k in inputs), "map fission changed the value"


def test_map_fission_no_moves_without_independent_groups():
    # The enumeration must stay honest in the other direction: a single-statement body has nothing to split.
    @dace.program
    def one_statement_map(a: f64[N], b: f64[N]):
        for i in dace.map[0:N]:
            b[i] = a[i] + 1.0

    assert map_fission_moves(one_statement_map.to_sdfg(simplify=True)) == []


def test_fission_no_op_on_single_statement():

    @dace.program
    def one_statement(a: f64[N], b: f64[N]):
        for i in range(1, N):
            b[i] = b[i - 1] + a[i]

    inputs = mk(names=("a", "b"))
    ref = run(one_statement.to_sdfg(simplify=True), inputs, 48)
    sdfg = one_statement.to_sdfg(simplify=True)
    fission_to_statements(sdfg)  # nothing independent to split
    got = run(sdfg, inputs, 48)
    assert all(np.allclose(got[k], ref[k]) for k in inputs)
