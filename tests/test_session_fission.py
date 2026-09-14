# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Session's per-nest fission API (:meth:`Session.list_fissions`, :meth:`Session.fission`): the same
id/epoch safety layer :mod:`test_session` proves for fusion, applied to the single-pair
:func:`nestforge.phases.schedule.map_fission_moves` arm.
"""
from typing import Dict, List, Optional, Tuple

import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from dace.sdfg.state import SDFGState
from dace.transformation.helpers import nest_state_subgraph

from nestforge.session import Session, StaleHandle

N = dace.symbol("N")
f64 = dace.float64


def two_statement_map() -> Tuple[dace.SDFG, SDFGState, nodes.MapEntry]:
    """One flat map computing ``b = a + 1`` and ``c = a * 2`` -- both statements under ONE map."""
    sdfg = dace.SDFG("multi_statement_map")
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
    return sdfg, state, me


def multi_statement_map_sdfg() -> dace.SDFG:
    """``two_statement_map`` with its body wrapped into a NestedSDFG -- MapFission's map-with-nested-SDFG
    pattern, the shape :func:`map_fission_moves` enumerates."""
    sdfg, state, me = two_statement_map()
    nest_state_subgraph(sdfg, state, state.scope_subgraph(me, include_entry=False, include_exit=False))
    sdfg.validate()
    return sdfg


def all_map_entries(sdfg: dace.SDFG) -> List[nodes.MapEntry]:
    return [
        n for sd in sdfg.all_sdfgs_recursive() for state in sd.all_states() for n in state.nodes()
        if isinstance(n, nodes.MapEntry)
    ]


def tasklet_scopes(sdfg: dace.SDFG) -> Dict[str, Optional[nodes.Node]]:
    """Tasklet label -> its immediate enclosing MapEntry (``None`` if unmapped at that state)."""
    scopes: Dict[str, Optional[nodes.Node]] = {}
    for sd in sdfg.all_sdfgs_recursive():
        for state in sd.all_states():
            scope = state.scope_dict()
            for node in state.nodes():
                if isinstance(node, nodes.Tasklet):
                    scopes[node.label] = scope[node]
    return scopes


def test_two_statement_map_starts_as_one_map_two_statements():
    sdfg, state, me = two_statement_map()
    assert len(all_map_entries(sdfg)) == 1
    scope = state.scope_dict()
    tasklets = {n.label: n for n in state.nodes() if isinstance(n, nodes.Tasklet)}
    assert scope[tasklets["one"]] is me and scope[tasklets["two"]] is me


def test_list_fissions_finds_the_multi_output_map():
    session = Session(multi_statement_map_sdfg())
    moves = session.list_fissions()
    assert len(moves) == 1
    assert moves[0]["id"].startswith("e0:fission:")
    assert "fission-map" in moves[0]["label"]


def test_fission_splits_the_map_by_statement_and_stales_prior_ids_and_matches_numpy():
    session = Session(multi_statement_map_sdfg())
    assert len(all_map_entries(session.sdfg)) == 1

    moves = session.list_fissions()
    session.fission(moves[0]["id"])

    assert session.epoch == 1
    assert len(all_map_entries(session.sdfg)) == 2  # the outer map split, one map per statement
    with pytest.raises(StaleHandle):  # the move id from before the split is gone
        session.fission(moves[0]["id"])

    scopes = tasklet_scopes(session.sdfg)
    assert scopes["one"] is not None and scopes["two"] is not None
    assert scopes["one"] is not scopes["two"]  # b = a+1 and c = a*2 now sit in separate maps

    n = 16
    a = np.random.default_rng(0).random(n)
    b = np.zeros(n)
    c = np.zeros(n)
    session.sdfg(a=a.copy(), b=b, c=c, N=n)
    assert np.allclose(b, a + 1.0)
    assert np.allclose(c, a * 2.0)


def test_fission_resolve_rejects_wrong_kind():
    session = Session(multi_statement_map_sdfg())
    nest_id = session.list_nests()[0]["id"]
    with pytest.raises(KeyError):  # a nest id is not a fission-move id
        session.fission(nest_id)
