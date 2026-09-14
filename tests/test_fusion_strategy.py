# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Phase-1 fusion-strategy API (:mod:`nestforge.phases.schedule`): the named/registered granularity strategy that
Phase 1 applies, and the guarantee that ``maximal-fusion`` reaches the same fixed point as draining the
per-move arm surface -- so the deterministic default and the agent's move-by-move policy agree.
"""
import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from dace.transformation.interstate.state_fusion import StateFusion

from nestforge.phases import schedule as fusion
from nestforge.phases.schedule import (enumerate_fusions, fusion_strategy_names, get_fusion_strategy, maximal_fusion,
                              register_fusion_strategy)

N = dace.symbol("N")
f64 = dace.float64


@dace.program
def producer_consumer_maps(a: f64[N], b: f64[N]):
    tmp = np.empty_like(a)
    for i in dace.map[0:N]:
        tmp[i] = a[i] * 2.0
    for i in dace.map[0:N]:
        b[i] = tmp[i] + 1.0


def test_registry_has_maximal_fusion():
    assert "maximal-fusion" in fusion_strategy_names()
    assert get_fusion_strategy("maximal-fusion") is maximal_fusion


def test_get_unknown_strategy_raises():
    with pytest.raises(KeyError):
        get_fusion_strategy("no-such-strategy")


def test_register_roundtrip(monkeypatch):
    # Register into a COPY of the registry: it is process-global, so a leaked entry would show up in
    # every later test that lists the strategies (tests/test_phase_api_contract.py does).
    monkeypatch.setattr(fusion, "_REGISTRY", dict(fusion._REGISTRY))
    marker = lambda sdfg: 0
    register_fusion_strategy("test-noop-strategy", marker)
    assert get_fusion_strategy("test-noop-strategy") is marker


def map_count(sdfg):
    return sum(isinstance(n, nodes.MapEntry) for st in sdfg.states() for n in st.nodes())


def test_maximal_fusion_drains_all_legal_fusions():
    # After max-fuse the producer/consumer pair is ONE map, and no legal fuse move remains -- the same
    # fixed point the agent reaches applying moves one at a time.
    sdfg = producer_consumer_maps.to_sdfg(simplify=True)
    sdfg.apply_transformations_repeated(StateFusion)
    steps = maximal_fusion(sdfg)
    assert steps >= 1
    assert map_count(sdfg) == 1
    assert enumerate_fusions(sdfg) == []


def test_maximal_fusion_is_value_preserving():
    rng = np.random.default_rng(0)
    inputs = {k: rng.random(48) for k in ("a", "b")}
    ref = {k: v.copy() for k, v in inputs.items()}
    producer_consumer_maps.to_sdfg(simplify=True)(**ref, N=48)

    sdfg = producer_consumer_maps.to_sdfg(simplify=True)
    sdfg.apply_transformations_repeated(StateFusion)
    maximal_fusion(sdfg)
    got = {k: v.copy() for k, v in inputs.items()}
    sdfg(**got, N=48)
    assert all(np.allclose(got[k], ref[k]) for k in inputs)
