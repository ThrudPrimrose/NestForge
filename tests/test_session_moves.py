# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Scheduling moves addressed by tree labels: :meth:`Session.list_moves` and :meth:`Session.apply_move`."""

import copy
import hashlib
import json
import re
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import pytest

import dace
from dace.libraries.blas import Dot
from dace.sdfg import nodes
from dace.sdfg.state import ControlFlowBlock, LoopRegion

from nestforge.ir.names import normalize_for_tree
from nestforge.session import MoveResult, Session

N = dace.symbol("N", dtype=dace.int64)
T = dace.symbol("T", dtype=dace.int64)
SIZE = 7
STEPS = 5


@dace.program
def recur(x: dace.float64[N], out: dace.float64[N]):
    for i in range(1, N):
        out[i] = out[i - 1] + x[i]


@dace.program
def calls_then_loops(a: dace.float64[N], b: dace.float64[N], oa: dace.float64[N], ob: dace.float64[N]):
    recur(a, oa)
    recur(b, ob)
    for i in range(1, N):
        a[i] = a[i - 1] + oa[i]
    for i in range(1, N):
        b[i] = b[i - 1] + ob[i]


@dace.program
def two_loops(a: dace.float64[N], b: dace.float64[N]):
    for i in range(1, N):
        a[i] = a[i - 1] + 1.0
    for i in range(1, N):
        b[i] = b[i - 1] + a[i]


@dace.program
def read_ahead(a: dace.float64[N], b: dace.float64[N]):
    for i in range(1, N - 1):
        a[i] = a[i - 1] + 1.0
    for i in range(1, N - 1):
        b[i] = b[i - 1] + a[i + 1]


@dace.program
def vertical_pair(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    tmp = np.empty_like(A)
    for i in dace.map[0:N]:
        tmp[i] = A[i] + B[i]
    for i in dace.map[0:N]:
        C[i] = tmp[i] * 2.0


@dace.program
def two_indep(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N], D: dace.float64[N]):
    for i in dace.map[0:N]:
        C[i] = A[i] * 2.0
    for i in dace.map[0:N]:
        D[i] = B[i] * 3.0


@dace.program
def two_statements(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
    for i in dace.map[0:N]:
        b[i] = a[i] + 1.0
        c[i] = a[i] * 2.0


@dace.program
def nested(A: dace.float64[N, N], B: dace.float64[N, N]):
    for i in dace.map[0:N]:
        for j in dace.map[0:N]:
            B[i, j] = A[i, j] * 2.0


@dace.program
def triangle(A: dace.float64[N, N], B: dace.float64[N, N]):
    for i in dace.map[0:N]:
        for j in dace.map[i:N]:
            B[i, j] = A[i, j] * 2.0


@dace.program
def sweep(A: dace.float64[T, N]):
    for t in range(1, T):
        for i in dace.map[0:N]:
            A[t, i] = A[t - 1, i] + 1.0


@dace.program
def lane_cross(A: dace.float64[T, N]):
    for t in range(1, T):
        for i in dace.map[1:N]:
            A[t, i] = A[t - 1, i - 1] + 1.0


@dace.program
def sweep_then_scale(A: dace.float64[T, N], B: dace.float64[N]):
    for t in range(1, T):
        for i in dace.map[0:N]:
            A[t, i] = A[t - 1, i] + 1.0
    for i in dace.map[0:N]:
        B[i] = A[1, i] * 2.0


def tree_labels(sdfg: dace.SDFG) -> List[str]:
    return [
        node.label
        for node, _ in sdfg.all_nodes_recursive()
        if isinstance(node, (ControlFlowBlock, nodes.MapEntry, nodes.LibraryNode))
    ]


def duplicate_labels(sdfg: dace.SDFG) -> Dict[str, int]:
    return {label: count for label, count in Counter(tree_labels(sdfg)).items() if count > 1}


def digest(sdfg: dace.SDFG) -> str:
    return hashlib.sha256(json.dumps(sdfg.to_json(), sort_keys=True, default=str).encode()).hexdigest()


def loops(sdfg: dace.SDFG) -> List[LoopRegion]:
    return [
        b for cfg in sdfg.all_control_flow_regions(recursive=True) for b in cfg.nodes() if isinstance(b, LoopRegion)
    ]


def top_level_maps(sdfg: dace.SDFG) -> List[nodes.MapEntry]:
    return [n for state in sdfg.all_states() for n in state.scope_children()[None] if isinstance(n, nodes.MapEntry)]


def map_writes(sdfg: dace.SDFG) -> List[List[str]]:
    """The arrays each map writes, over every map in the SDFG hierarchy."""
    return sorted(
        sorted({e.data.data for e in state.out_edges(state.exit_node(node))})
        for node, state in sdfg.all_nodes_recursive()
        if isinstance(node, nodes.MapEntry)
    )


def session_and_reference(program, simplify: bool = True) -> Tuple[Session, dace.SDFG]:
    sdfg = program.to_sdfg(simplify=simplify)
    reference = copy.deepcopy(sdfg)
    reference.name = f"{sdfg.name}_reference"
    return Session(sdfg), reference


def apply_listed(session: Session, kind: str) -> MoveResult:
    (move,) = session.list_moves(kind)
    return session.apply_move(move["kind"], move["labels"], move["epoch"])


def random_arrays(**shapes: Tuple[int, ...]) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    return {name: rng.random(shape) for name, shape in shapes.items()}


def assert_same_values(reference: dace.SDFG, moved: dace.SDFG, arrays: Dict[str, np.ndarray], **symbols: int) -> None:
    expected = {name: value.copy() for name, value in arrays.items()}
    actual = {name: value.copy() for name, value in arrays.items()}
    reference(**expected, **symbols)
    moved(**actual, **symbols)
    for name in arrays:
        np.testing.assert_allclose(actual[name], expected[name], rtol=1e-13, err_msg=name)


# labels


def test_a_helper_called_twice_repeats_labels_that_the_session_makes_unique():
    sdfg = calls_then_loops.to_sdfg(simplify=False)
    repeated = duplicate_labels(sdfg)
    assert "init" in repeated and any(label.startswith("for_") for label in repeated), repeated

    Session(sdfg)

    assert duplicate_labels(sdfg) == {}


def test_labels_stay_unique_after_a_fusion_move():
    sut, reference = session_and_reference(calls_then_loops, simplify=False)
    loop_count = len(loops(sut.sdfg))

    result = apply_listed(sut, "loop-fusion")

    assert result.status == "applied"
    assert len(loops(sut.sdfg)) == loop_count - 1
    assert duplicate_labels(sut.sdfg) == {}
    assert_same_values(reference, sut.sdfg, random_arrays(a=(SIZE,), b=(SIZE,), oa=(SIZE,), ob=(SIZE,)), N=SIZE)


def test_a_session_keeps_the_labels_normalize_for_tree_already_gave():
    sdfg = sweep_then_scale.to_sdfg(simplify=True)
    normalize_for_tree(sdfg)
    labels = tree_labels(sdfg)

    Session(sdfg)

    assert tree_labels(sdfg) == labels


def test_repeated_library_node_labels_become_unique_and_keep_name_equal_to_label():
    sdfg = dace.SDFG("two_dots")
    state = sdfg.add_state()
    first, second = Dot("dot"), Dot("dot")
    state.add_node(first)
    state.add_node(second)

    Session(sdfg)

    assert first.label == "dot" and second.label != "dot"
    assert (first.name, second.name) == (first.label, second.label)


def test_the_tree_names_its_epoch_on_the_first_line():
    sut = Session(vertical_pair.to_sdfg(simplify=True))
    assert sut.describe().splitlines()[0] == f"SDFG '{sut.sdfg.label}'  epoch=0"

    apply_listed(sut, "map-fusion")

    assert sut.describe().splitlines()[0] == f"SDFG '{sut.sdfg.label}'  epoch=1"


def test_listed_moves_name_rows_of_the_tree_at_the_current_epoch():
    sut = Session(sweep_then_scale.to_sdfg(simplify=True))
    tree = sut.describe()

    moves = sut.list_moves()

    assert moves, "the fixture offers no move"
    assert {move["epoch"] for move in moves} == {0}
    for label in {label for move in moves for label in move["labels"]}:
        assert re.search(rf"\] {label}\b", tree), (label, tree)


# loop fusion


def test_loop_fusion_merges_two_loops_into_one_that_writes_both_arrays():
    sut, reference = session_and_reference(two_loops)

    result = apply_listed(sut, "loop-fusion")

    assert (result.status, result.reason) == ("applied", "LoopFusion")
    (loop,) = loops(sut.sdfg)
    assert {"a", "b"} <= set(loop.read_and_write_sets()[1])
    assert_same_values(reference, sut.sdfg, random_arrays(a=(SIZE,), b=(SIZE,)), N=SIZE)


def test_loop_fusion_that_would_read_ahead_is_illegal_and_changes_nothing():
    sut = Session(read_ahead.to_sdfg(simplify=True))
    before = digest(sut.sdfg)

    result = sut.apply_move("loop-fusion", ["for0_0", "for0_1"], 0)

    assert result.status == "illegal" and "FuseLoops" in result.reason, result
    assert (digest(sut.sdfg), sut.epoch) == (before, 0)
    assert sut.list_moves("loop-fusion") == []


# map fusion


def test_map_fusion_through_a_transient_is_vertical_and_leaves_one_map():
    sut, reference = session_and_reference(vertical_pair)

    result = apply_listed(sut, "map-fusion")

    assert (result.status, result.reason) == ("applied", "MapFusionVertical")
    assert len(top_level_maps(sut.sdfg)) == 1
    assert_same_values(reference, sut.sdfg, random_arrays(A=(SIZE,), B=(SIZE,), C=(SIZE,)), N=SIZE)


def test_map_fusion_of_independent_siblings_is_horizontal_and_leaves_one_map():
    sut, reference = session_and_reference(two_indep)

    result = apply_listed(sut, "map-fusion")

    assert (result.status, result.reason) == ("applied", "MapFusionHorizontal")
    assert len(top_level_maps(sut.sdfg)) == 1
    arrays = random_arrays(A=(SIZE,), B=(SIZE,), C=(SIZE,), D=(SIZE,))
    assert_same_values(reference, sut.sdfg, arrays, N=SIZE)


def test_map_fusion_across_states_is_illegal_and_names_the_state_barrier():
    sut = Session(two_indep.to_sdfg(simplify=False))
    labels = [entry.label for entry in top_level_maps(sut.sdfg)]
    before = digest(sut.sdfg)

    result = sut.apply_move("map-fusion", labels, 0)

    assert result.status == "illegal" and "different states" in result.reason, result
    assert (digest(sut.sdfg), sut.epoch) == (before, 0)


# map fission


def test_map_fission_splits_a_two_statement_map_into_maps_that_each_write_one_array():
    sut, reference = session_and_reference(two_statements, simplify=False)
    assert map_writes(sut.sdfg) == [["b", "c"]]

    result = apply_listed(sut, "map-fission")

    assert (result.status, result.reason) == ("applied", "MapFission")
    # MapFission moves the split maps into the nested SDFG that was the map's body
    assert top_level_maps(sut.sdfg) == []
    writes = map_writes(sut.sdfg)
    assert len(writes) > 1 and all(len(arrays) == 1 for arrays in writes), writes
    assert_same_values(reference, sut.sdfg, random_arrays(a=(SIZE,), b=(SIZE,), c=(SIZE,)), N=SIZE)


def test_map_fission_of_a_map_without_a_nested_body_is_illegal():
    sut = Session(two_statements.to_sdfg(simplify=True))
    (entry,) = top_level_maps(sut.sdfg)
    before = digest(sut.sdfg)

    result = sut.apply_move("map-fission", [entry.label], 0)

    assert result.status == "illegal" and "MapFission" in result.reason, result
    assert (digest(sut.sdfg), sut.epoch) == (before, 0)


# interchange


def test_map_map_interchange_puts_the_inner_parameter_outside():
    sut, reference = session_and_reference(nested)

    result = apply_listed(sut, "interchange-map-map")

    assert (result.status, result.reason) == ("applied", "MapInterchange")
    (outer,) = top_level_maps(sut.sdfg)
    state = next(s for s in sut.sdfg.all_states() if outer in s.nodes())
    (inner,) = [n for n in state.scope_children()[outer] if isinstance(n, nodes.MapEntry)]
    assert (outer.map.params, inner.map.params) == (["j"], ["i"])
    assert_same_values(reference, sut.sdfg, random_arrays(A=(SIZE, SIZE), B=(SIZE, SIZE)), N=SIZE)


def test_map_map_interchange_of_a_triangular_nest_is_illegal():
    sut = Session(triangle.to_sdfg(simplify=True))
    before = digest(sut.sdfg)

    result = sut.apply_move("interchange-map-map", ["kernel1_0", "kernel2_0"], 0)

    assert result.status == "illegal" and "MapInterchange" in result.reason, result
    assert (digest(sut.sdfg), sut.epoch) == (before, 0)


def test_naming_the_inner_map_first_is_illegal():
    sut = Session(nested.to_sdfg(simplify=True))
    before = digest(sut.sdfg)

    result = sut.apply_move("interchange-map-map", ["kernel2_0", "kernel1_0"], 0)

    assert result.status == "illegal" and "not directly inside" in result.reason, result
    assert digest(sut.sdfg) == before


def test_loop_map_interchange_moves_the_loop_inside_the_map():
    sut, reference = session_and_reference(sweep)

    result = apply_listed(sut, "interchange-loop-map")

    assert (result.status, result.reason) == ("applied", "MoveLoopIntoMap")
    assert not [block for block in sut.sdfg.nodes() if isinstance(block, LoopRegion)]
    (entry,) = top_level_maps(sut.sdfg)
    state = next(s for s in sut.sdfg.all_states() if entry in s.nodes())
    (body,) = [n for n in state.scope_children()[entry] if isinstance(n, nodes.NestedSDFG)]
    assert [block for block in body.sdfg.nodes() if isinstance(block, LoopRegion)]
    assert_same_values(reference, sut.sdfg, random_arrays(A=(STEPS, SIZE)), N=SIZE, T=STEPS)


def test_loop_map_interchange_that_would_cross_map_iterations_is_illegal():
    sut = Session(lane_cross.to_sdfg(simplify=True))
    before = digest(sut.sdfg)

    result = sut.apply_move("interchange-loop-map", ["for0_0", "kernel2_0"], 0)

    assert result.status == "illegal" and "MoveLoopIntoMap" in result.reason, result
    assert (digest(sut.sdfg), sut.epoch) == (before, 0)


def test_loop_map_interchange_with_a_map_outside_the_loop_is_illegal():
    sut = Session(sweep_then_scale.to_sdfg(simplify=True))
    (loop,) = loops(sut.sdfg)
    (outside,) = [n for n in top_level_maps(sut.sdfg) if all(n not in s.nodes() for s in loop.all_states())]
    before = digest(sut.sdfg)

    result = sut.apply_move("interchange-loop-map", [loop.label, outside.label], 0)

    assert result.status == "illegal" and "not the one map directly inside" in result.reason, result
    assert digest(sut.sdfg) == before


# kinds without a DaCe transformation


@pytest.mark.parametrize(
    "program, kind, labels",
    [
        (two_loops, "loop-fission", ["for0_0"]),
        (two_loops, "interchange-loop-loop", ["for0_0", "for0_1"]),
        (sweep, "interchange-map-loop", ["kernel2_0", "for0_0"]),
    ],
)
def test_a_kind_without_a_dace_transformation_is_not_implemented_and_changes_nothing(program, kind, labels):
    sut = Session(program.to_sdfg(simplify=True))
    before = digest(sut.sdfg)

    result = sut.apply_move(kind, labels, 0)

    assert (result.status, result.kind, result.labels) == ("not-implemented", kind, tuple(labels))
    assert result.reason
    assert (digest(sut.sdfg), sut.epoch) == (before, 0)
    assert sut.list_moves(kind) == []


def test_per_loop_fission_points_at_the_whole_program_fission():
    result = Session(two_loops.to_sdfg(simplify=True)).apply_move("loop-fission", ["for0_0"], 0)

    assert "whole-SDFG" in result.reason and "fission_all" in result.reason, result


# refusals that never touch the graph


def test_an_unknown_label_is_not_found_and_changes_nothing():
    sut = Session(vertical_pair.to_sdfg(simplify=True))
    before = digest(sut.sdfg)

    result = sut.apply_move("map-fusion", ["kernel1_0", "kernel9_9"], 0)

    assert result.status == "not-found" and "kernel9_9" in result.reason, result
    assert (digest(sut.sdfg), sut.epoch) == (before, 0)


def test_labels_read_before_a_move_are_stale_after_it_and_change_nothing():
    sut = Session(vertical_pair.to_sdfg(simplify=True))
    (move,) = sut.list_moves("map-fusion")
    sut.apply_move(move["kind"], move["labels"], move["epoch"])
    after_move = digest(sut.sdfg)

    result = sut.apply_move(move["kind"], move["labels"], move["epoch"])

    assert result.status == "stale", result
    assert (digest(sut.sdfg), sut.epoch) == (after_move, 1)


def test_an_unknown_kind_is_a_malformed_call():
    sut = Session(vertical_pair.to_sdfg(simplify=True))
    with pytest.raises(ValueError, match="unknown move kind"):
        sut.apply_move("loop-tiling", ["for0_0"], 0)
    with pytest.raises(ValueError, match="unknown move kind"):
        sut.list_moves("loop-tiling")


def test_a_wrong_label_count_is_a_malformed_call():
    sut = Session(vertical_pair.to_sdfg(simplify=True))
    with pytest.raises(ValueError, match="takes 2 label"):
        sut.apply_move("map-fusion", ["kernel1_0"], 0)
