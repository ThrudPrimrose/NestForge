# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Structure inspection + fuse diagnosis: describe_graph (CFG tree + per-nest read/write sets) and
can_fuse ("yes" or a reason). can_fuse shares the exact gates of enumerate_fusions, so a "yes" here is a
move apply_fusion would accept and a reason marks a pair that never appears in enumerate_fusions.
"""
import numpy as np
import dace

from nestforge.ir.introspect import describe_graph, nest_reads_writes
from nestforge.phases.schedule import can_fuse, enumerate_fusions
from nestforge.phases.scopes import top_level_map_entries

N = dace.symbol('N')


@dace.program
def vertical_pair(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    T = np.empty_like(A)  # transient intermediate -> vertical fusion is legal
    for i in dace.map[0:N]:
        T[i] = A[i] + B[i]
    for i in dace.map[0:N]:
        C[i] = T[i] * 2.0


@dace.program
def live_output_pair(A: dace.float64[N], B: dace.float64[N], T: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N]:  # T is a parameter -> a live (non-transient) output
        T[i] = A[i] + B[i]
    for i in dace.map[0:N]:
        C[i] = T[i] * 2.0


@dace.program
def independent_pair(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N], D: dace.float64[N]):
    for i in dace.map[0:N]:  # no shared data -> horizontal fusion
        C[i] = A[i] * 2.0
    for i in dace.map[0:N]:
        D[i] = B[i] * 3.0


def map_entries(sdfg):
    return [(st, me) for st in sdfg.states() for me in top_level_map_entries(st)]


def test_describe_graph_lists_nests_with_read_write_sets():
    sdfg = vertical_pair.to_sdfg(simplify=True)
    text = describe_graph(sdfg)
    # This SDFG is NOT normalized, so the blocks keep their frontend labels -- describe_graph renders
    # whatever it is given, and normalize_for_tree is what makes those labels canonical.
    assert "`- MapState" in text  # the state the two nests share, as a tree row
    assert "[i=0:N]" in text  # the iteration domain. There is no parallel/sequential column: a Map is
    #                           data-parallel by definition, so that column said the same thing always
    assert "reads=['A', 'B'] writes=['T']" in text  # producer nest
    assert "reads=['T'] writes=['C']" in text  # consumer nest


def test_nest_reads_writes_matches_the_tree():
    sdfg = vertical_pair.to_sdfg(simplify=True)
    entries = map_entries(sdfg)
    reads_writes = [nest_reads_writes(st, me) for st, me in entries]
    assert (['A', 'B'], ['T']) in reads_writes
    assert (['T'], ['C']) in reads_writes


def test_can_fuse_yes_for_vertical_transient():
    sdfg = vertical_pair.to_sdfg(simplify=True)
    (_, m1), (_, m2) = map_entries(sdfg)
    assert can_fuse(sdfg, m1, m2) == "yes"
    assert can_fuse(sdfg, m2, m1) == "yes"  # direction-independent
    assert enumerate_fusions(sdfg), "a legal move must exist when can_fuse says yes"


def test_can_fuse_yes_for_independent_horizontal():
    sdfg = independent_pair.to_sdfg(simplify=True)
    entries = map_entries(sdfg)
    (s1, m1), (s2, m2) = entries[0], entries[1]
    assert s1 is s2, "independent maps land in one state after simplify"
    assert can_fuse(sdfg, m1, m2) == "yes"
    assert enumerate_fusions(sdfg)


def test_can_fuse_refuses_live_output_with_reason():
    sdfg = live_output_pair.to_sdfg(simplify=True)
    (_, m1), (_, m2) = map_entries(sdfg)
    reason = can_fuse(sdfg, m1, m2)
    assert reason != "yes"
    assert "live output" in reason and "non-transient" in reason
    assert not enumerate_fusions(sdfg), "a refused pair must not be an enumerated move"


def test_can_fuse_reports_state_barrier_across_states():
    sdfg = independent_pair.to_sdfg(simplify=False)  # each map keeps its own state
    entries = map_entries(sdfg)
    (s1, m1), (s2, m2) = entries[0], entries[1]
    assert s1 is not s2
    reason = can_fuse(sdfg, m1, m2)
    assert "different states" in reason
    assert "control-flow dependency" in reason


def test_can_fuse_rejects_map_loop_mix():
    sdfg = vertical_pair.to_sdfg(simplify=True)
    (_, m1), _ = map_entries(sdfg)
    reason = can_fuse(sdfg, m1, "not-a-nest")
    assert "map-nest with a loop-nest" in reason
