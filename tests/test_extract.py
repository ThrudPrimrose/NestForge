# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np
import dace

from nestforge.phases.scopes import parallel_top_level_maps
from nestforge.ir.extract import extract_nest_to_sdfg, trip_count_symbols

N = dace.symbol('N')
B_SYM = dace.symbol('B')
OUTER = dace.symbol('OUTER')
OFF = dace.symbol('OFF')


@dace.program
def vadd(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N]:
        C[i] = A[i] + B[i]


@dace.program
def shifted(A: dace.float64[N], C: dace.float64[N]):
    """``OFF`` only ever lands in a SUBSCRIPT; ``N`` bounds the map. The two must classify differently."""
    for i in dace.map[0:N - 64]:
        C[i + OFF] = A[i + OFF] * 2.0


def test_outer_finds_the_map():
    sdfg = vadd.to_sdfg(simplify=True)
    refs = parallel_top_level_maps(sdfg)
    assert len(refs) == 1
    _, node = refs[0]
    assert isinstance(node, dace.sdfg.nodes.MapEntry)


def test_extract_map_nest_boundary_and_correctness():
    sdfg = vadd.to_sdfg(simplify=True)
    psdfg, node = parallel_top_level_maps(sdfg)[0]
    b = extract_nest_to_sdfg(psdfg, node, name="vadd_nest")

    assert set(b.inputs) == {"A", "B"}
    assert set(b.outputs) == {"C"}
    assert "N" in b.symbols

    standalone = b.standalone_sdfg
    for name in ("A", "B", "C"):
        assert name in standalone.arrays

    # The parent SDFG (now holding the NestedSDFG) still computes vadd.
    A = np.random.default_rng(0).random(16)
    B = np.random.default_rng(1).random(16)
    C = np.zeros(16)
    sdfg(A=A, B=B, C=C, N=16)
    np.testing.assert_allclose(C, A + B)

    # The standalone SDFG computes vadd on its own.
    A2 = np.random.default_rng(2).random(16)
    B2 = np.random.default_rng(3).random(16)
    C2 = np.zeros(16)
    standalone(A=A2, B=B2, C=C2, N=16)
    np.testing.assert_allclose(C2, A2 + B2)


def nested_bound_sdfg(symbol_mapping):
    """An outer SDFG whose only loop bound lives INSIDE a NestedSDFG, reached under ``symbol_mapping``."""
    sdfg = dace.SDFG("outer")
    sdfg.add_array("a", [N], dace.float64)
    state = sdfg.add_state(is_start_block=True)
    inner = dace.SDFG("inner")
    inner.add_array("a", [N], dace.float64)
    inner.add_state(is_start_block=True).add_mapped_tasklet("t", {"i": "0:B"}, {},
                                                            "out = 1.0", {"out": dace.Memlet("a[i]")},
                                                            external_edges=True)
    nsdfg = state.add_nested_sdfg(inner, {}, {"a"}, symbol_mapping=symbol_mapping)
    state.add_edge(nsdfg, "a", state.add_access("a"), None, dace.Memlet("a[0:N]"))
    sdfg.validate()
    return sdfg


def test_trip_count_symbols_separates_a_bound_from_a_subscript():
    # The whole contract in one nest: N bounds the map, OFF only indexes. Built by the FRONTEND, so the
    # shape is the one real kernels have rather than one hand-assembled to agree with the analysis.
    syms = trip_count_symbols(shifted.to_sdfg(simplify=True))
    assert "N" in syms
    assert "OFF" not in syms


def test_trip_count_symbols_recurses_into_a_nested_sdfg():
    # REGRESSION: a bound one level down is still a bound. Reporting {} here would let sample_sizes bind
    # it to 0, make the inner map zero-trip, and have oracle and candidate agree on untouched memory.
    assert trip_count_symbols(nested_bound_sdfg({"N": N, "B": B_SYM})) == {"B"}


def test_trip_count_symbols_translates_an_inner_bound_through_symbol_mapping():
    # The inner map is bounded by `B`, but the parent binds B := OUTER - 1 and has no `B` of its own.
    # Only OUTER is nameable by a caller, so only OUTER is a useful answer.
    syms = trip_count_symbols(nested_bound_sdfg({"N": N, "B": OUTER - 1}))
    assert syms == {"OUTER"}


def test_trip_count_symbols_takes_a_condition_but_not_an_assignment():
    """The `j = j + 1` shape: an inter-state ASSIGNMENT moves a value along, a CONDITION gates the
    iteration. Reading an assignment as a trip-count use would raise on every leaked induction start."""
    sdfg = dace.SDFG("branchy")
    sdfg.add_array("a", [N], dace.float64)
    sdfg.add_symbol("gate", dace.int64)
    sdfg.add_symbol("carried", dace.int64)
    s1 = sdfg.add_state(is_start_block=True)
    s2 = sdfg.add_state()
    sdfg.add_edge(s1, s2, dace.InterstateEdge(condition="gate > 0", assignments={"carried": "carried + 1"}))
    syms = trip_count_symbols(sdfg)
    assert "gate" in syms  # decides whether s2 runs at all
    assert "carried" not in syms  # only carried along


if __name__ == "__main__":
    test_outer_finds_the_map()
    test_extract_map_nest_boundary_and_correctness()
    print("extract OK")
