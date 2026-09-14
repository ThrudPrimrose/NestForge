# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Container-level kernel dependencies: which producers reach each ``ExternalCall`` argument, on programs lowered
by phase 2 and on hand-built SDFGs where the frontend cannot express the case. Structural only; nothing compiles."""

import copy
import json
from typing import Dict, List, Sequence, Tuple

import pytest

import dace
from dace.sdfg.state import LoopRegion

from nestforge.ir.depends import ArgEdge, KernelGraph, Producer, Reach, UnsupportedProgram, kernel_dependencies
from nestforge.ir.libnode import ExternalCall, in_conn, out_conn
from nestforge.phases.scopes import lower_nests_to_external_call

N = dace.symbol("N")
K = dace.symbol("K")
STEPS = dace.symbol("STEPS")


@dace.program
def chain(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    T = A + B
    C[:] = T * 2


@dace.program
def diamond(A: dace.float64[N], D: dace.float64[N]):
    T = A + 1
    U = T * 2
    V = T + 3
    D[:] = U + V


@dace.program
def gather(A: dace.float64[N], idx: dace.int64[N], B: dace.float64[N]):
    for i in dace.map[0:N]:
        B[i] = A[idx[i]]


@dace.program
def scale_by_body_symbol(A: dace.float64[N], B: dace.float64[N]):
    for i in dace.map[0:N]:
        B[i] = A[i] * K


@dace.program
def overwrite(A: dace.float64[N], B: dace.float64[N], X: dace.float64[N], C: dace.float64[N]):
    X[:] = A + 1
    X[:] = B + 2
    C[:] = X * 3


@dace.program
def if_without_else(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N], c: dace.int64):
    if c > 0:
        for i in dace.map[0:N]:
            B[i] = A[i] + 1
    for i in dace.map[0:N]:
        C[i] = B[i] * 2


@dace.program
def if_with_else(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N], c: dace.int64):
    if c > 0:
        for i in dace.map[0:N]:
            B[i] = A[i] + 1
    else:
        for i in dace.map[0:N]:
            B[i] = A[i] - 1
    for i in dace.map[0:N]:
        C[i] = B[i] * 2


@dace.program
def relax_steps(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    for t in range(STEPS):
        for i in dace.map[0:N]:
            B[i] = A[i] + 1
        for i in dace.map[0:N]:
            A[i] = B[i] * 0.5
    for i in dace.map[0:N]:
        C[i] = A[i] * 3


@dace.program
def relax_three(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    for t in range(3):
        for i in dace.map[0:N]:
            B[i] = A[i] + 1
        for i in dace.map[0:N]:
            A[i] = B[i] * 0.5
    for i in dace.map[0:N]:
        C[i] = A[i] * 3


@dace.program
def relax_until_negative(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N], c: dace.float64[N]):
    for t in range(3):
        for i in dace.map[0:N]:
            B[i] = A[i] + 1
        if c[0] < 0:
            break
        for i in dace.map[0:N]:
            A[i] = B[i] * 0.5
    for i in dace.map[0:N]:
        C[i] = A[i] * 3


@dace.program
def copy_between(A: dace.float64[N], T: dace.float64[N], U: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N]:
        T[i] = A[i] + 1
    U[:] = T
    for i in dace.map[0:N]:
        C[i] = U[i] * 2


def lowered(program) -> Tuple[dace.SDFG, Dict[str, List[str]]]:
    """``program`` through phase 2, with the arguments each kernel writes (read off its out-connectors)."""
    sdfg = program.to_sdfg(simplify=True)
    kernels = lower_nests_to_external_call(sdfg)
    writes = {ext.label: sorted(c.removeprefix(out_conn("")) for c in ext.out_connectors) for ext, _ in kernels}
    return sdfg, writes


def reaching(graph: KernelGraph) -> Dict[Tuple[str, str], Tuple[str, ...]]:
    return {(edge.consumer, edge.arg): tuple(r.text() for r in edge.producers) for edge in (*graph.edges, *graph.exits)}


def loop_label(sdfg: dace.SDFG) -> str:
    (loop,) = [region for region in sdfg.all_control_flow_regions() if isinstance(region, LoopRegion)]
    return loop.label


def hand_kernel(
    sdfg: dace.SDFG,
    state: dace.SDFGState,
    name: str,
    reads: Sequence[str],
    writes: Sequence[str],
    symbols: Sequence[str] = (),
) -> ExternalCall:
    """An ``ExternalCall`` wired to whole-array reads and writes, with the manifest phase 2 would give it."""
    manifest = {"input_args": [*reads, *writes, *symbols], "array_args": [*reads, *writes], "output_args": [*writes]}
    ext = ExternalCall(
        name,
        inputs={in_conn(r): None for r in reads},
        outputs={out_conn(w): None for w in writes},
        config=manifest,
    )
    state.add_node(ext)
    for r in reads:
        state.add_edge(state.add_read(r), None, ext, in_conn(r), dace.Memlet.from_array(r, sdfg.arrays[r]))
    for w in writes:
        state.add_edge(ext, out_conn(w), state.add_write(w), None, dace.Memlet.from_array(w, sdfg.arrays[w]))
    return ext


def test_chain_feeds_the_first_kernels_temporary_to_the_second():
    sdfg, writes = lowered(chain)
    assert writes == {"extcall_0": ["T"], "extcall_1": ["C"]}

    graph = kernel_dependencies(sdfg)

    assert graph.kernels == ("extcall_0", "extcall_1")
    assert reaching(graph) == {
        ("extcall_0", "A"): ("program",),
        ("extcall_0", "B"): ("program",),
        ("extcall_0", "N"): ("program",),
        ("extcall_1", "T"): ("extcall_0.T",),
        ("extcall_1", "N"): ("program",),
        ("exit", "C"): ("extcall_1.C",),
    }
    assert [(edge.arg, edge.role) for edge in graph.edges if edge.consumer == "extcall_1"] == [
        ("T", "input"),
        ("N", "symbol"),
    ]


def test_chain_lines_list_every_argument_per_kernel_then_the_exit():
    sdfg, _ = lowered(chain)

    lines = kernel_dependencies(sdfg).lines()

    assert lines == [
        "extcall_0: A <- program, B <- program, N <- program",
        "extcall_1: T <- extcall_0.T, N <- program",
        "exit: C <- extcall_1.C",
    ]


def test_diamond_kernel_zero_feeds_both_middle_kernels():
    sdfg, writes = lowered(diamond)
    assert writes == {"extcall_0": ["T"], "extcall_1": ["U"], "extcall_2": ["V"], "extcall_3": ["D"]}

    graph = kernel_dependencies(sdfg)

    assert reaching(graph) == {
        ("extcall_0", "A"): ("program",),
        ("extcall_0", "N"): ("program",),
        ("extcall_1", "T"): ("extcall_0.T",),
        ("extcall_1", "N"): ("program",),
        ("extcall_2", "T"): ("extcall_0.T",),
        ("extcall_2", "N"): ("program",),
        ("extcall_3", "U"): ("extcall_1.U",),
        ("extcall_3", "V"): ("extcall_2.V",),
        ("extcall_3", "N"): ("program",),
        ("exit", "D"): ("extcall_3.D",),
    }
    assert sorted(edge.consumer for edge in graph.consumers_of("extcall_0")) == ["extcall_1", "extcall_2"]


def test_an_index_array_argument_comes_from_the_program():
    sdfg, writes = lowered(gather)
    assert writes == {"extcall_0": ["B"]}

    graph = kernel_dependencies(sdfg)

    assert reaching(graph) == {
        ("extcall_0", "A"): ("program",),
        ("extcall_0", "idx"): ("program",),
        ("extcall_0", "N"): ("program",),
        ("exit", "B"): ("extcall_0.B",),
    }


def test_a_symbol_read_only_in_the_kernel_body_is_a_kernel_argument():
    sdfg, writes = lowered(scale_by_body_symbol)
    assert writes == {"extcall_0": ["B"]}

    graph = kernel_dependencies(sdfg)

    assert ArgEdge("extcall_0", "K", "symbol", (Reach(Producer("program")),)) in graph.edges


def test_a_second_write_kills_the_first_writer():
    sdfg, writes = lowered(overwrite)
    assert writes == {"extcall_0": ["X"], "extcall_1": ["X"], "extcall_2": ["C"]}

    graph = kernel_dependencies(sdfg)

    assert reaching(graph) == {
        ("extcall_0", "A"): ("program",),
        ("extcall_0", "N"): ("program",),
        ("extcall_1", "B"): ("program",),
        ("extcall_1", "N"): ("program",),
        ("extcall_2", "X"): ("extcall_1.X",),
        ("extcall_2", "N"): ("program",),
        ("exit", "C"): ("extcall_2.C",),
        ("exit", "X"): ("extcall_1.X",),
    }


def test_a_kernel_under_if_without_else_joins_the_program_value():
    sdfg, writes = lowered(if_without_else)
    assert writes == {"extcall_0": ["B"], "extcall_1": ["C"]}

    graph = kernel_dependencies(sdfg)

    assert reaching(graph) == {
        ("extcall_0", "A"): ("program",),
        ("extcall_0", "N"): ("program",),
        ("extcall_1", "B"): ("extcall_0.B", "program"),
        ("extcall_1", "N"): ("program",),
        ("exit", "B"): ("extcall_0.B", "program"),
        ("exit", "C"): ("extcall_1.C",),
    }


def test_kernels_in_both_branches_replace_the_program_value():
    sdfg, writes = lowered(if_with_else)
    assert writes == {"extcall_0": ["B"], "extcall_1": ["B"], "extcall_2": ["C"]}

    graph = kernel_dependencies(sdfg)

    assert reaching(graph) == {
        ("extcall_0", "A"): ("program",),
        ("extcall_0", "N"): ("program",),
        ("extcall_1", "A"): ("program",),
        ("extcall_1", "N"): ("program",),
        ("extcall_2", "B"): ("extcall_0.B", "extcall_1.B"),
        ("extcall_2", "N"): ("program",),
        ("exit", "B"): ("extcall_0.B", "extcall_1.B"),
        ("exit", "C"): ("extcall_2.C",),
    }


def test_a_loop_carries_the_last_kernels_output_back_to_the_first():
    sdfg, writes = lowered(relax_steps)
    assert writes == {"extcall_0": ["B"], "extcall_1": ["A"], "extcall_2": ["C"]}
    loop = loop_label(sdfg)

    graph = kernel_dependencies(sdfg)

    first_reads = next(edge for edge in graph.edges if (edge.consumer, edge.arg) == ("extcall_0", "A"))
    assert first_reads.producers == (Reach(Producer("kernel", "extcall_1", "A"), (loop,)), Reach(Producer("program")))
    assert reaching(graph) == {
        ("extcall_0", "A"): (f"extcall_1.A [carried: {loop}]", "program"),
        ("extcall_0", "N"): ("program",),
        ("extcall_1", "B"): ("extcall_0.B",),
        ("extcall_1", "N"): ("program",),
        ("extcall_2", "A"): ("extcall_1.A", "program"),
        ("extcall_2", "N"): ("program",),
        ("exit", "A"): ("extcall_1.A", "program"),
        ("exit", "B"): ("extcall_0.B", "program"),
        ("exit", "C"): ("extcall_2.C",),
    }


def test_a_loop_proven_to_run_drops_the_value_it_was_entered_with():
    sdfg, writes = lowered(relax_three)
    assert writes == {"extcall_0": ["B"], "extcall_1": ["A"], "extcall_2": ["C"]}
    loop = loop_label(sdfg)

    graph = kernel_dependencies(sdfg)

    assert reaching(graph) == {
        ("extcall_0", "A"): (f"extcall_1.A [carried: {loop}]", "program"),
        ("extcall_0", "N"): ("program",),
        ("extcall_1", "B"): ("extcall_0.B",),
        ("extcall_1", "N"): ("program",),
        ("extcall_2", "A"): ("extcall_1.A",),
        ("extcall_2", "N"): ("program",),
        ("exit", "A"): ("extcall_1.A",),
        ("exit", "B"): ("extcall_0.B",),
        ("exit", "C"): ("extcall_2.C",),
    }


def test_a_break_before_the_second_kernel_leaves_with_the_value_the_loop_had():
    sdfg, writes = lowered(relax_until_negative)
    assert writes == {"extcall_0": ["B"], "extcall_1": ["A"], "extcall_2": ["C"]}
    loop = loop_label(sdfg)

    graph = kernel_dependencies(sdfg)

    assert reaching(graph)[("extcall_2", "A")] == ("extcall_1.A", f"extcall_1.A [carried: {loop}]", "program")


def test_a_symbol_assigned_from_a_kernel_output_names_that_kernel_and_the_assignment():
    sdfg = dace.SDFG("symbol_from_output")
    for name in ("x", "y", "z"):
        sdfg.add_array(name, [8], dace.float64)
    sdfg.add_array("cnt", [3], dace.int64, transient=True)
    counting = sdfg.add_state("counting", is_start_block=True)
    hand_kernel(sdfg, counting, "extcall_0", ["x"], ["cnt"])
    sizing = sdfg.add_state("sizing")
    hand_kernel(sdfg, sizing, "extcall_1", ["y"], ["z"], symbols=["M"])
    sdfg.add_edge(counting, sizing, dace.InterstateEdge(assignments={"M": "cnt[2]"}))

    graph = kernel_dependencies(sdfg)

    assert graph.edges == (
        ArgEdge("extcall_0", "x", "input", (Reach(Producer("program")),)),
        ArgEdge("extcall_1", "y", "input", (Reach(Producer("program")),)),
        ArgEdge("extcall_1", "M", "symbol", (Reach(Producer("kernel", "extcall_0", "cnt")),), ("M = cnt[2]",)),
    )


def test_a_host_writer_is_named_by_its_state():
    sdfg = dace.SDFG("host_writer")
    sdfg.add_array("x", [8], dace.float64)
    sdfg.add_array("y", [8], dace.float64)
    state = sdfg.add_state("fill", is_start_block=True)
    kernel = hand_kernel(sdfg, state, "extcall_0", ["x"], ["y"])
    (x_node,) = [edge.src for edge in state.in_edges(kernel)]
    fill = state.add_tasklet("fill", {}, {"out": dace.float64}, "out = 1.0")
    state.add_edge(fill, "out", x_node, None, dace.Memlet("x[0]"))

    graph = kernel_dependencies(sdfg)

    assert reaching(graph)[("extcall_0", "x")] == ("host:fill",)


def test_a_host_copy_between_kernels_forwards_the_copied_producer():
    sdfg, writes = lowered(copy_between)
    assert writes == {"extcall_0": ["T"], "extcall_1": ["C"]}

    graph = kernel_dependencies(sdfg)

    assert reaching(graph) == {
        ("extcall_0", "A"): ("program",),
        ("extcall_0", "N"): ("program",),
        ("extcall_1", "U"): ("extcall_0.T",),
        ("extcall_1", "N"): ("program",),
        ("exit", "C"): ("extcall_1.C",),
        ("exit", "T"): ("extcall_0.T",),
        ("exit", "U"): ("extcall_0.T",),
    }


def test_a_reference_container_is_refused():
    sdfg = dace.SDFG("with_reference")
    sdfg.add_array("A", [8], dace.float64)
    sdfg.add_reference("R", [8], dace.float64)
    sdfg.add_state("only", is_start_block=True)

    with pytest.raises(UnsupportedProgram, match="'R' .* Reference"):
        kernel_dependencies(sdfg)


def test_a_kernel_inside_a_nested_sdfg_is_refused():
    inner = dace.SDFG("inner")
    inner.add_array("a", [8], dace.float64)
    inner.add_array("b", [8], dace.float64)
    hand_kernel(inner, inner.add_state("body", is_start_block=True), "extcall_0", ["a"], ["b"])
    outer = dace.SDFG("outer")
    outer.add_array("A", [8], dace.float64)
    outer.add_array("B", [8], dace.float64)
    state = outer.add_state("call", is_start_block=True)
    nested = state.add_nested_sdfg(inner, {"a": None}, {"b": None})
    state.add_edge(state.add_read("A"), None, nested, "a", dace.Memlet.from_array("A", outer.arrays["A"]))
    state.add_edge(nested, "b", state.add_write("B"), None, dace.Memlet.from_array("B", outer.arrays["B"]))

    with pytest.raises(UnsupportedProgram, match="'extcall_0' sits inside nested SDFG 'inner'"):
        kernel_dependencies(outer)


def test_two_runs_on_copies_give_byte_identical_json():
    sdfg, _ = lowered(relax_steps)

    first = json.dumps(kernel_dependencies(copy.deepcopy(sdfg)).to_json())
    second = json.dumps(kernel_dependencies(copy.deepcopy(sdfg)).to_json())

    assert first == second
