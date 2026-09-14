# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Phase-IV re-inlining: an externalized nest must come back FUSABLE.

The agent externalizes nests to measure them, then asks for two of them to be fused again. That round trip
only works if (a) the ``ExternalCall`` still carries the nest's own SDFG and (b) expanding it back leaves
maps the fusion arms can actually see. A NestedSDFG hides its maps from ``MapFusion``, so a re-inlined
program that is never inlined fuses nothing -- and reports success while doing it.
"""
import numpy as np

import dace

from nestforge.phases.schedule import maximal_fusion
from nestforge.phases.scopes import lower_nests_to_external_call

N = dace.symbol("N")
f64 = dace.float64


@dace.program
def two_maps(a: f64[N], b: f64[N], c: f64[N]):
    """Two elementwise maps over the same range: a horizontal pair once both are visible."""
    b[:] = a[:] * 2.0
    c[:] = a[:] + 1.0


def maps_in(sdfg):
    """Every MapEntry reachable from the top level, NOT descending into NestedSDFGs -- which is exactly
    what MapFusion can see."""
    return [n for state in sdfg.states() for n in state.nodes() if isinstance(n, dace.nodes.MapEntry)]


def nested_in(sdfg):
    return [n for state in sdfg.states() for n in state.nodes() if isinstance(n, dace.nodes.NestedSDFG)]


def externalized():
    """The program with every map nest lowered to an ``ExternalCall`` -- the state the agent measures in."""
    sdfg = two_maps.to_sdfg(simplify=True)
    calls = lower_nests_to_external_call(sdfg, "map")
    assert calls, "fixture lowered no nest; the test would prove nothing"
    return sdfg, calls


def test_externalized_nest_keeps_its_own_sdfg_for_reinlining():
    """(a) The material to re-inline with. Without it phase IV cannot start."""
    _, calls = externalized()
    for ext, _boundary in calls:
        assert ext._standalone_sdfg is not None, f"{ext.name} cannot be re-inlined: no standalone SDFG"


def test_reinlined_nests_are_inlined_so_map_fusion_can_see_them():
    """(b) The round trip. After expanding back to NestedSDFGs, ``maximal_fusion`` must reach ONE map.

    This is the failure this test exists for: MapFusion never descends into a NestedSDFG, so if the
    re-inlined nests are left nested, fusion finds no pair, returns without error, and the granularity
    ladder silently reports the rung as unfusable.
    """
    sdfg, _ = externalized()
    sdfg.expand_library_nodes()  # DaceReference: each nest returns as a NestedSDFG
    assert nested_in(sdfg), "fixture did not produce NestedSDFGs; the hazard under test is absent"

    maximal_fusion(sdfg)
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
    maximal_fusion(sdfg)
    got_b, got_c = np.empty(n), np.empty(n)
    sdfg(a=a.copy(), b=got_b, c=got_c, N=n)

    assert np.array_equal(got_b, ref_b)
    assert np.array_equal(got_c, ref_c)
