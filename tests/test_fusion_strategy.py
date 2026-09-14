# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Phase-1 deterministic default (:mod:`nestforge.phases.schedule`): ``full_fusion`` on a ``normalize``d SDFG
reaches the same fixed point as draining the per-move arm surface -- so the deterministic default and the
agent's move-by-move policy agree.
"""
import numpy as np

import dace
from dace.sdfg import nodes

from nestforge.phases.normalize import Targets, normalize
from nestforge.phases.schedule import enumerate_fusions, fission_to_statements, full_fusion

N = dace.symbol("N")
f64 = dace.float64


@dace.program
def producer_consumer_maps(a: f64[N], b: f64[N]):
    tmp = np.empty_like(a)
    for i in dace.map[0:N]:
        tmp[i] = a[i] * 2.0
    for i in dace.map[0:N]:
        b[i] = tmp[i] + 1.0


@dace.program
def sibling_maps(a: f64[N], b: f64[N], c: f64[N]):
    for i in dace.map[0:N]:
        b[i] = a[i] * 2.0
    for i in dace.map[0:N]:
        c[i] = a[i] + 1.0


def map_count(sdfg):
    return sum(
        isinstance(n, nodes.MapEntry) for sd in sdfg.all_sdfgs_recursive() for st in sd.all_states()
        for n in st.nodes())


def test_full_fusion_reaches_a_state_with_no_legal_fusions_left():
    """A vertical producer/consumer pair ends as one map, with no legal fuse move remaining."""
    raw = producer_consumer_maps.to_sdfg(simplify=False)
    assert map_count(raw) == 2, "fixture must start with two separate maps"

    sdfg = producer_consumer_maps.to_sdfg(simplify=False)
    targets = Targets()
    normalize(sdfg, targets)
    # normalize's own loop-fusion sub-stage already folds this pair to one map, before the
    # map-level 'fuse' stage full_fusion runs -- so full_fusion is confirmed idempotent here.
    full_fusion(sdfg, targets)
    assert map_count(sdfg) == 1
    assert enumerate_fusions(sdfg) == []


def test_full_fusion_redrains_legal_fusions_after_fission():
    """Fissioning a fused horizontal pair back apart reopens a legal move that full_fusion re-drains."""
    sdfg = sibling_maps.to_sdfg(simplify=False)
    targets = Targets()
    normalize(sdfg, targets)
    full_fusion(sdfg, targets)
    assert map_count(sdfg) == 1, "normalize+full_fusion must reach one map before fission reopens anything"

    assert fission_to_statements(sdfg) >= 1
    assert map_count(sdfg) == 2
    reopened = enumerate_fusions(sdfg)
    assert [move.kind for move in reopened] == ["fuse-map-horizontal"]

    full_fusion(sdfg, targets)
    assert map_count(sdfg) == 1
    assert enumerate_fusions(sdfg) == []


def test_full_fusion_is_value_preserving():
    """Fusing must not change the vertical pair's numeric result."""
    rng = np.random.default_rng(0)
    inputs = {k: rng.random(48) for k in ("a", "b")}
    ref = {k: v.copy() for k, v in inputs.items()}
    producer_consumer_maps.to_sdfg(simplify=True)(**ref, N=48)

    sdfg = producer_consumer_maps.to_sdfg(simplify=False)
    targets = Targets()
    normalize(sdfg, targets)
    full_fusion(sdfg, targets)
    got = {k: v.copy() for k, v in inputs.items()}
    sdfg(**got, N=48)
    assert all(np.allclose(got[k], ref[k]) for k in inputs)
