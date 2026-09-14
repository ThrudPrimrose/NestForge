# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Phase-2 offload-granularity API (:mod:`nestforge.phases.scopes`): the named/registered granularity that
selects which nests leave the SDFG as external calls, the non-mutating candidate inspector the agent
reads before committing, and the externalize-before-offload commit.
"""
import numpy as np
import pytest

import dace
from dace.sdfg import nodes

from nestforge.ir.libnode import ExternalCall
from nestforge.phases.scopes import (DEFAULT_GRANULARITY, OffloadCandidate, get_strategy, lower_nests_to_external_call,
                                     offload_candidates, strategy_names, whole_program_boundary)

N = dace.symbol("N")
f64 = dace.float64


@dace.program
def two_nests(a: f64[N], c: f64[N]):
    tmp = np.empty_like(a)
    for i in dace.map[0:N]:
        tmp[i] = a[i] * 2.0
    for i in dace.map[0:N]:
        c[i] = tmp[i] + 1.0


def test_registry_has_default_granularity():
    assert DEFAULT_GRANULARITY in strategy_names()
    assert get_strategy(DEFAULT_GRANULARITY)  # resolvable


def test_get_unknown_granularity_raises():
    with pytest.raises(KeyError):
        get_strategy("no-such-granularity")


def test_offload_candidates_are_non_mutating():
    sdfg = two_nests.to_sdfg(simplify=True)
    before = sum(isinstance(n, nodes.MapEntry) for st in sdfg.states() for n in st.nodes())
    cands = offload_candidates(sdfg)  # default granularity
    assert all(isinstance(c, OffloadCandidate) for c in cands)
    assert len(cands) == 2  # two top-level parallel map-nests
    assert all(c.parallel for c in cands)  # LoopToMap-ed maps are parallel
    assert all(c.label.startswith("map[") for c in cands)
    after = sum(isinstance(n, nodes.MapEntry) for st in sdfg.states() for n in st.nodes())
    assert before == after  # inspection did NOT extract


def test_lower_externalizes_each_candidate():
    sdfg = two_nests.to_sdfg(simplify=True)
    cands = offload_candidates(sdfg)
    lowered = lower_nests_to_external_call(sdfg)
    assert len(lowered) == len(cands)
    ext_nodes = [n for st in sdfg.states() for n in st.nodes() if isinstance(n, ExternalCall)]
    assert len(ext_nodes) == len(cands)  # every selected nest became an ExternalCall


def test_lowered_sdfg_stays_value_preserving():
    rng = np.random.default_rng(0)
    inputs = {k: rng.random(48) for k in ("a", "c")}
    ref = {k: v.copy() for k, v in inputs.items()}
    two_nests.to_sdfg(simplify=True)(**ref, N=48)

    sdfg = two_nests.to_sdfg(simplify=True)
    lower_nests_to_external_call(sdfg)  # DaceReference expansion keeps it runnable
    got = {k: v.copy() for k, v in inputs.items()}
    sdfg(**got, N=48)
    assert all(np.allclose(got[k], ref[k]) for k in inputs)


def test_whole_program_is_coarsest_granularity():
    sdfg = two_nests.to_sdfg(simplify=True)
    b = whole_program_boundary(sdfg)
    assert b.nsdfg_node is None  # no extraction -- one unit
    assert set(b.inputs) == {"a"}
    assert set(b.outputs) == {"c"}  # b is transient scratch, not a caller output
