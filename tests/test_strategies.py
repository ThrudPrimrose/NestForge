# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Detection strategies: skip-taskloops (default), innermost maps, innermost loops."""
import pytest

pytest.importorskip("hpcagent_bench")

from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion

import dace

from nestforge.corpus.bench import iter_dace_kernels
from nestforge.phases.scopes import empty_strategy_reason, get_strategy


def sdfg_for(short):
    return {k.short_name: k for k in iter_dace_kernels()}[short].to_sdfg(simplify=True)


def test_empty_strategy_reason_distinguishes_libnode_only_from_empty():
    # A kernel whose only compute is a library node is NOT "no compute nest": DaCe offloads the library
    # node to its fastest implementation, so there is legitimately no loop-nest to externalise.
    sdfg = dace.SDFG("lib_only")
    st = sdfg.add_state()
    from dace.libraries.standard.nodes.reduce import Reduce
    st.add_node(Reduce("Reduce", wcr="lambda a, b: a + b", axes=None))  # a library node with no map/loop
    assert "library-node" in empty_strategy_reason(sdfg)
    # An honestly empty SDFG (no compute at all) keeps the plain message.
    assert "no compute nest" in empty_strategy_reason(dace.SDFG("empty"))


def test_skip_taskloops_descends_through_a_map_only_time_loop():
    # jacobi's TSTEPS loop body is only maps -> a taskloop; skip it and offload the spatial maps.
    sdfg = sdfg_for("hpc/structured_grids/jacobi_1d/jacobi_1d")
    refs = get_strategy("skip-taskloops")(sdfg)
    assert refs, "expected the inner compute maps"
    assert all(isinstance(n, nodes.MapEntry) for _, n in refs)
    # outer would instead offload the loop wrapper itself.
    outer = get_strategy("outer")(sdfg)
    assert len(outer) == 1 and isinstance(outer[0][1], LoopRegion)


def test_innermost_yields_leaf_maps_when_present():
    # jacobi's loop holds maps, so the innermost units are those maps (not the loop wrapper).
    sdfg = sdfg_for("hpc/structured_grids/jacobi_1d/jacobi_1d")
    refs = get_strategy("innermost")(sdfg)
    assert refs and all(isinstance(n, nodes.MapEntry) for _, n in refs)


def test_innermost_yields_leaf_loops_for_loop_only_kernels():
    # lu has no maps and an outer loop wrapping two inner loops; the innermost units are those two.
    sdfg = sdfg_for("hpc/dense_linear_algebra/lu/lu")
    refs = get_strategy("innermost")(sdfg)
    assert len(refs) == 2 and all(isinstance(n, LoopRegion) for _, n in refs)
