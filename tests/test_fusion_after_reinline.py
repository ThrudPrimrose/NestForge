# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Phase-IV re-inlining: an externalized nest must come back FUSABLE, via ``full_fusion``.

A NestedSDFG hides its maps from ``MapFusion``, so a re-inlined program left nested fuses nothing and
reports success while doing it. ``normalize`` + ``full_fusion`` already fold ``two_maps`` to one map, so
the fixture fissions it back apart first to get two nests worth externalizing.
"""
import numpy as np

import dace

from nestforge.phases.normalize import Targets, normalize
from nestforge.phases.schedule import fission_to_statements, full_fusion
from nestforge.phases.scopes import lower_nests_to_external_call

N = dace.symbol("N")
f64 = dace.float64


@dace.program
def two_maps(a: f64[N], b: f64[N], c: f64[N]):
    """Two elementwise maps over the same range: a horizontal pair once both are visible."""
    b[:] = a[:] * 2.0
    c[:] = a[:] + 1.0


def maps_in(sdfg):
    """Top-level MapEntry nodes only -- exactly what MapFusion can see."""
    return [n for state in sdfg.states() for n in state.nodes() if isinstance(n, dace.nodes.MapEntry)]


def nested_in(sdfg):
    return [n for state in sdfg.states() for n in state.nodes() if isinstance(n, dace.nodes.NestedSDFG)]


def externalized():
    """Two map nests, each lowered to its own ``ExternalCall`` -- the state the agent measures in."""
    sdfg = two_maps.to_sdfg(simplify=False)
    targets = Targets()
    normalize(sdfg, targets)
    full_fusion(sdfg, targets)
    assert fission_to_statements(sdfg) >= 1, "fixture fused to one map but fission split nothing back apart"
    calls = lower_nests_to_external_call(sdfg)
    assert len(calls) == 2, "fixture must externalize both statements separately"
    return sdfg, calls


def test_externalized_nest_keeps_its_own_sdfg_for_reinlining():
    """(a) The material to re-inline with. Without it phase IV cannot start."""
    _, calls = externalized()
    for ext, _boundary in calls:
        assert ext._standalone_sdfg is not None, f"{ext.name} cannot be re-inlined: no standalone SDFG"


def test_reinlined_nests_are_inlined_so_map_fusion_can_see_them():
    """(b) The round trip. After expanding back to NestedSDFGs, ``full_fusion`` must reach ONE map."""
    sdfg, _ = externalized()
    sdfg.expand_library_nodes()  # DaceReference: each nest returns as a NestedSDFG
    assert nested_in(sdfg), "fixture did not produce NestedSDFGs; the hazard under test is absent"

    full_fusion(sdfg, Targets())
    assert not nested_in(sdfg), "re-inlined nests were left nested, so MapFusion could not see their maps"
    assert len(maps_in(sdfg)) == 1, f"expected the two maps to fuse into one, got {len(maps_in(sdfg))}"


def test_reinlined_and_fused_program_still_computes_the_same_values():
    """Fusing after a re-inline must be value-preserving -- the whole point of measuring the fused rung."""
    n = 64
    rng = np.random.default_rng(0)
    a = rng.random(n)

    ref_b, ref_c = np.empty(n), np.empty(n)
    two_maps.to_sdfg(simplify=True)(a=a.copy(), b=ref_b, c=ref_c, N=n)

    sdfg, _ = externalized()
    sdfg.expand_library_nodes()
    full_fusion(sdfg, Targets())
    got_b, got_c = np.empty(n), np.empty(n)
    sdfg(a=a.copy(), b=got_b, c=got_c, N=n)

    assert np.array_equal(got_b, ref_b)
    assert np.array_equal(got_c, ref_c)
