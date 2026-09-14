# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression tests for LATENT emit bugs found in the full-repo audit -- the classes that produce a WRONG
or uncompilable numpy oracle rather than an obvious crash. Unit set, no compile.

Each test fails without its fix:
  * sympy ``Max``/``Min``/``Abs`` reached the oracle verbatim (only ``np`` is in scope -> NameError);
  * an empty (data=None) happens-before edge was taken as a library node's operand -> ``arrays[None]``;
  * an unstructured join double-applied every predecessor's inter-state assignments;
  * an array dtype with no extern-C spelling KeyError'd mid-codegen.
"""

from dataclasses import dataclass

import numpy as np
import pytest
import sympy

import dace
from dace import symbolic

from nestforge.ir.emit_libnode import UnsupportedLibraryNode, data_edge
from nestforge.ir.emit_numpy import EMITTED_BUILTINS, UnsupportedNest, int_floor, normalize_casts, sdfg_to_numpy
from nestforge.ir.libnode import ExternalCall, proto_and_call

I = sympy.Symbol("i")
N = sympy.Symbol("N")


def evaluated(expr, **env):
    """symstr the sympy expression the way a subset/bound reaches the emitter, normalize it, and evaluate
    the emitted text in the SAME namespace the oracle runs in (python builtins + EMITTED_BUILTINS)."""
    code = normalize_casts(symbolic.symstr(expr))
    assert "Max(" not in code and "Min(" not in code and "Abs(" not in code, f"un-rewritten sympy func: {code}"
    return eval(code, dict(EMITTED_BUILTINS), dict(env))


def test_clamped_index_exprs_are_emittable_and_exact():
    # a stencil/boundary-clamped subset: A[Max(0, i-1)] / bound Min(N, i+1). Before the fix symstr rendered
    # `Max(0, i - 1)` straight into the kernel and the oracle died with NameError: name 'Max'.
    assert evaluated(sympy.Max(0, I - 1), i=5) == 4
    assert evaluated(sympy.Max(0, I - 1), i=0) == 0  # the clamp actually clamps
    assert evaluated(sympy.Min(N, I + 1), i=5, N=4) == 4
    assert evaluated(sympy.Abs(I - 3), i=1) == 2


def test_variadic_max_min_are_rewritten():
    # sympy Max/Min are n-ary (a tile extent clamped by several bounds); a binary-only rewrite would
    # mis-render or drop arguments.
    assert evaluated(sympy.Max(0, N, I - 1), i=5, N=4) == 4
    assert evaluated(sympy.Min(0, N, I - 1), i=5, N=4) == 0


def test_clamped_expr_keeps_integer_type_for_indexing():
    # the rewrite must yield a Python int, not a numpy scalar: the value indexes an array and feeds range().
    value = evaluated(sympy.Max(0, I - 1), i=5)
    assert isinstance(value, int)
    assert range(0, evaluated(sympy.Min(N, I + 1), i=5, N=4))  # usable as a bound


@dataclass
class FakeEdge:
    """Minimal stand-in for a graph edge: ``data_edge`` only inspects ``e.data``."""

    data: dace.Memlet


def test_data_edge_skips_empty_ordering_edges():
    # DaCe's StateFusion adds empty (data=None) happens-before edges to sequence nodes without merging.
    # Taking edges[0] blindly picked one as the operand -> sdfg.arrays[None] KeyError mid-emission.
    empty, real = FakeEdge(dace.Memlet()), FakeEdge(dace.Memlet(data="A", subset="0:N"))
    assert empty.data.is_empty() and not real.data.is_empty()
    assert data_edge([empty, real], None, "input") is real  # the ordering edge is not the operand
    assert data_edge([real, empty], None, "input") is real


def test_data_edge_refuses_when_only_ordering_edges_exist():
    # no data-carrying edge at all: an actionable refusal, never arrays[None].
    with pytest.raises(UnsupportedLibraryNode, match="no data-carrying"):
        data_edge([FakeEdge(dace.Memlet())], None, "input")


def join_sdfg() -> dace.SDFG:
    """An unstructured join: two unconditional predecessors both assigning the same symbol into one block."""
    sdfg = dace.SDFG("join")
    sdfg.add_array("A", [2], dace.float64)
    sdfg.add_symbol("k", dace.int64)
    start = sdfg.add_state("start", is_start_block=True)
    left, right, join = sdfg.add_state("left"), sdfg.add_state("right"), sdfg.add_state("join")
    sdfg.add_edge(start, left, dace.InterstateEdge())
    sdfg.add_edge(start, right, dace.InterstateEdge())
    # both carry an assignment; straight-line emission would apply BOTH into `join`
    sdfg.add_edge(left, join, dace.InterstateEdge(assignments={"k": "k + 1"}))
    sdfg.add_edge(right, join, dace.InterstateEdge(assignments={"k": "k + 1"}))
    return sdfg


def test_unstructured_join_is_refused_not_double_applied():
    # Before the fix interstate_lines emitted `k = k + 1` once per in-edge, double-incrementing a symbol
    # that only ONE predecessor actually assigns at runtime -- a silently wrong index.
    with pytest.raises(UnsupportedNest, match="carry assignments"):
        sdfg_to_numpy(join_sdfg(), fn_name="join")


def test_single_predecessor_assignment_still_emits():
    # the refusal must not fire on the normal case: one edge carrying assignments.
    sdfg = dace.SDFG("chain")
    sdfg.add_array("A", [2], dace.float64)
    sdfg.add_symbol("k", dace.int64)
    first = sdfg.add_state("first", is_start_block=True)
    second = sdfg.add_state("second")
    sdfg.add_edge(first, second, dace.InterstateEdge(assignments={"k": "k + 1"}))
    assert "k = k + 1" in sdfg_to_numpy(sdfg, fn_name="chain")


DTYPES = {"float64": dace.float64, "complex128": dace.complex128}


def extern_call_with_dtype(dtype_name: str, shape=(8,)):
    """An ExternalCall wired into a real state: ``proto_and_call`` reads the memlets to tell a pointer
    connector from a value one, so a node without edges cannot answer the question it is asked."""
    manifest = {
        "array_args": ["A"],
        "output_args": [],
        "init": {"arrays": {"A": {"dtype": dtype_name}}, "scalars": {}},
    }
    node = ExternalCall("k", inputs={"_in_A"}, outputs=set(), config=manifest)
    node.symbol, node.abi_order = "k_fp64", ["A"]
    sdfg = dace.SDFG("host")
    sdfg.add_array("A", list(shape), DTYPES[dtype_name])
    state = sdfg.add_state()
    state.add_edge(state.add_read("A"), None, node, "_in_A", dace.Memlet.from_array("A", sdfg.arrays["A"]))
    return node, state


def test_extern_c_prototype_builds_for_a_known_dtype():
    proto, call = proto_and_call(*extern_call_with_dtype("float64"))
    assert "const double* A" in proto and "k_fp64(_in_A);" == call


def test_a_one_element_connector_is_passed_by_address():
    """DaCe defines a one-element memlet as a VALUE, but the compiled signature always takes a pointer:
    passing it directly is "cannot convert double to double*" at compile time (E3: s114/s115/s116)."""
    _, state = extern_call_with_dtype("float64", shape=(1,))
    node = next(n for n in state.nodes() if isinstance(n, ExternalCall))
    proto, call = proto_and_call(node, state)
    assert "const double* A" in proto
    assert call == "k_fp64(&_in_A);"


def test_unspellable_array_dtype_is_refused_not_keyerror():
    # a complex/fp16/unsigned array used to raise a bare KeyError from _CPP_SCALAR mid-codegen; it must be an
    # actionable refusal naming the array and dtype so the caller can keep the DaceReference variant.
    with pytest.raises(ValueError, match="complex128"):
        proto_and_call(*extern_call_with_dtype("complex128"))


@pytest.mark.parametrize(
    "expr, want",
    [
        ("int_floor(a[i, j], 2)", "((a[i, j]) // (2))"),
        ("int_floor(aa[i, j] + b[k, l], 2)", "((aa[i, j] + b[k, l]) // (2))"),
        ("int_floor(x, 2)", "((x) // (2))"),
    ],
)
def test_a_subscript_comma_does_not_split_an_argument(expr, want):
    """``apply_call`` counted parens only, so the comma inside ``a[i, j]`` was read as an argument
    separator: the rewrite got the wrong arity and the pieces spliced back with unmatched brackets.
    That emitted C which would not parse (TSVC s1111/s1113), and where the arity raised, the call was
    left unrewritten and leaked into C as an unresolved function (s111's ``int_ceil``)."""
    from nestforge.ir.emit_numpy import apply_call

    assert apply_call(expr, "int_floor", lambda a, b: f"(({a}) // ({b}))") == want


def test_multidim_subscript_survives_the_userfunc_fixpoint():
    """End to end through rewrite_userfuncs, where the real emitter routes it. ``int_floor``/``int_ceil``
    are NOT in the rewrite table -- they stay calls for both back ends -- so a rewritten function
    (variadic ``Max``) carries the subscript-comma case here."""
    from nestforge.ir.emit_numpy import rewrite_userfuncs

    out = rewrite_userfuncs("d[Max(aa[i, j], 2)] = b[int_ceil(Min(c[k, l], 4), 4)]")
    assert out.count("[") == out.count("]"), out
    assert out.count("(") == out.count(")"), out
    assert "Max" not in out and "Min" not in out, out
    assert out == "d[max(aa[i, j], 2)] = b[int_ceil(min(c[k, l], 4), 4)]", out


# --- data-dependent scratch extents (the spmv CSR span) -----------------------------------------------
M_SYM = dace.symbol("M")
NNZ_SYM = dace.symbol("NNZ")


@dace.program
def spmv_row_scratch(
    A_indptr: dace.int64[M_SYM + 1], A_vals: dace.float64[NNZ_SYM], x: dace.float64[M_SYM], y: dace.float64[M_SYM]
):
    """A CSR row loop: the row scratch is sized by a span READ OUT OF the index array."""
    for i in range(M_SYM):
        start = A_indptr[i]
        stop = A_indptr[i + 1]
        row = np.empty(stop - start, dace.float64)
        for k in range(stop - start):
            row[k] = A_vals[start + k] * x[k]
        y[i] = np.sum(row)


@pytest.mark.parametrize(
    "expr, want",
    [
        ("A_indptr[M + 1] - A_indptr[0]", False),  # the spmv span: residual symbols are kernel symbols
        ("A_indptr[i]", False),
        ("A_vals", False),  # a bare array NAME used as a value
        ("M*N", True),
        ("3", True),
        ("int_floor(N, 4)", True),  # a math head is not a data read
        ("Min(M, N)", True),  # Min/Max are not Function atoms at all
        ("i + 1", False),  # sizable for the other reason: not a kernel symbol
    ],
)
def test_sizable_rejects_a_data_read_however_it_is_spelled(expr, want):
    """``free_symbols`` is structurally blind to an indexed read: DaCe renders ``A_indptr[i]`` as
    ``Subscript(A_indptr, i)``, so the ARRAY NAME is the Function head and never appears among the free
    symbols. The predicate must walk the tree."""
    from nestforge.ir.emit_numpy import sizable

    arrays = {"A_indptr": None, "A_vals": None}
    assert sizable(symbolic.pystr_to_symbolic(expr), {"M", "N"}, arrays) is want


def test_data_dependent_scratch_extent_is_refused_not_emitted():
    """spmv used to EMIT, sizing ``row`` by ``A_indptr[M+1] - A_indptr[0]``.

    ``pystr_to_symbolic("A_indptr[i]")`` succeeds, so ``symbol_ranges`` folded the interstate assignment
    as a size bound and ``maxsize_loop_scratch`` widened the buffer to that CSR span. The old
    ``free_symbols - known`` check then accepted it -- the residual was ``{M} - {M} = set()`` -- and the
    caller was handed a signature whose buffer size only the data knows. Refuse, with a reason.
    """
    from nestforge.ir.emit_numpy import UnsupportedNest, sdfg_to_numpy

    sdfg = spmv_row_scratch.to_sdfg(simplify=True)
    assert "row" in sdfg.arrays and sdfg.arrays["row"].transient, "fixture no longer has the scratch buffer"
    with pytest.raises(UnsupportedNest, match="row"):
        sdfg_to_numpy(sdfg, "spmv")


def test_a_data_read_never_becomes_a_size_bound():
    """The narrow half: ``symbol_ranges`` must not ingest an interstate assignment that reads array data,
    or every shape it reaches widens to an extent the caller cannot evaluate."""
    from nestforge.ir.emit_numpy import maxsize_loop_scratch, reads_array_data

    sdfg = spmv_row_scratch.to_sdfg(simplify=True)
    widened = maxsize_loop_scratch(sdfg, ["M", "NNZ"])
    for dim in widened.arrays["row"].shape:
        assert not reads_array_data(sympy.sympify(dim), widened.arrays), f"widened to a data read: {dim}"


# --- copy DIRECTION (the in-place-copy inversion) ------------------------------------------------------
@dace.program
def shift_through_a_view(A: dace.float64[M_SYM]):
    """A slice-to-slice copy of ONE array: DaCe stages it through a view, so the copy edges are the
    shape `copy_direction` resolves."""
    A[1:M_SYM] = A[0 : M_SYM - 1]


@dace.program
def elementwise_from_a_row(A: dace.float64[M_SYM, M_SYM]):
    for i in range(M_SYM):
        A[i, 0] = A[i, M_SYM - 1]


@pytest.mark.parametrize("prog", [shift_through_a_view, elementwise_from_a_row], ids=["slice_copy", "element_copy"])
def test_copy_direction_agrees_with_dace_on_every_real_copy_edge(prog):
    """The emitter's direction must equal DaCe's OWN resolution on every access-node copy.

    Order was the bug: the destination was tested first, so on a copy whose two endpoints name the same
    array -- where both tests match -- ``subset`` was read as the destination. DaCe reads it as the
    source (``Memlet.try_initialize`` prefers ``is_data_src=True`` for that case), so the emitted
    assignment ran backwards. Every other copy has two distinct names and only one test can match, which
    is why nothing else caught it.

    Checked against DaCe rather than against a hand-built expectation, so the two cannot drift apart.
    """
    from dace.sdfg import nodes as dnodes
    from nestforge.ir.emit_numpy import copy_direction

    sdfg = prog.to_sdfg(simplify=True)
    checked = 0
    for state in sdfg.states():
        for edge in state.edges():
            if not (isinstance(edge.src, dnodes.AccessNode) and isinstance(edge.dst, dnodes.AccessNode)):
                continue
            if edge.data.is_empty():
                continue
            src_name, src_sub, dst_sub = copy_direction(edge)
            assert src_name == edge.src.data
            assert src_sub == edge.data.get_src_subset(edge, state), f"src subset disagrees on {edge}"
            assert dst_sub == edge.data.get_dst_subset(edge, state), f"dst subset disagrees on {edge}"
            checked += 1
    assert checked, "fixture produced no access-node copy edge -- it no longer covers copy_direction"


def test_copy_direction_reads_subset_as_the_source_when_both_ends_are_one_array():
    """The in-place case pinned directly, since the frontend stages most copies through a view.

    ``copy_direction`` must return ``memlet.subset`` as the SOURCE range here. Getting it backwards
    turns ``A[i] = A[j]`` into ``A[j] = A[i]`` -- a wrong answer with no error.
    """
    from nestforge.ir.emit_numpy import copy_direction

    sdfg = dace.SDFG("inplace_copy")
    sdfg.add_array("A", [M_SYM], dace.float64)
    state = sdfg.add_state()
    read, write = state.add_read("A"), state.add_write("A")
    memlet = dace.Memlet(data="A", subset="1:M", other_subset="0:M - 1")
    state.add_edge(read, None, write, None, memlet)

    src_name, src_sub, dst_sub = copy_direction(state.edges()[0])
    assert src_name == "A"
    assert str(src_sub) == "1:M", "memlet.subset indexes memlet.data, which DaCe resolves as the source"
    assert str(dst_sub) == "0:M - 1"


def test_two_kernels_of_equal_length_do_not_share_bytecode():
    """``load_emitted`` used to name the file by a COUNTER, so two different kernels could land on one
    path. CPython invalidates ``__pycache__`` on (mtime, size) only, so a same-second rewrite of equal
    byte length serves the FIRST kernel's bytecode -- the caller then validates and times a kernel it
    did not emit, silently. Forking made it worse: every child of one process inherited the same next
    counter value, so `run_isolated` handed every emitted kernel the same file name.

    Two sources of identical length, same name, back to back: each module must be its own kernel.
    """
    first = "def k():\n    return 'AAAA'\n"
    second = "def k():\n    return 'BBBB'\n"
    assert len(first) == len(second), "the fixture only reproduces the bug at equal byte length"
    from nestforge.ir.emit_numpy import load_emitted

    assert load_emitted(first, "k").k() == "AAAA"
    assert load_emitted(second, "k").k() == "BBBB"


@pytest.mark.parametrize(
    "end, step, want_stop",
    [
        ("N - 1", "1", "N"),
        ("N - 1", "-1", "N - 2"),
        ("0", "-1", "-1"),
        ("0", "2", "1"),
    ],
)
def test_range_stop_follows_the_step_sign(end, step, want_stop):
    """A DaCe range end is INCLUSIVE, so python's exclusive stop is one past the last element IN THE
    DIRECTION OF TRAVEL. The emitter added 1 unconditionally, so a descending map lost its final
    iteration: ``range(N-1, 0, -1)`` for a DaCe range ending at 0 never yields 0."""
    from nestforge.ir.emit_numpy import range_stop

    got = range_stop(symbolic.pystr_to_symbolic(end), symbolic.pystr_to_symbolic(step), "map parameter 'i'")
    assert sympy.simplify(got - symbolic.pystr_to_symbolic(want_stop)) == 0, f"{got} != {want_stop}"


def test_a_descending_range_covers_its_last_element():
    """The behaviour the sign fix buys, checked by ENUMERATING rather than by re-deriving the formula."""
    from nestforge.ir.emit_numpy import range_stop

    stop = int(range_stop(symbolic.pystr_to_symbolic("0"), symbolic.pystr_to_symbolic("-1"), "x"))
    assert list(range(7, stop, -1)) == [7, 6, 5, 4, 3, 2, 1, 0], "element 0 must not be dropped"


@pytest.mark.parametrize(
    "rng, want",
    [
        ((7, 0, -1), list(range(7, -1, -1))),
        ((7, 2, -1), [7, 6, 5, 4, 3, 2]),
        ((0, 7, 1), list(range(8))),
        ((1, 7, 2), [1, 3, 5, 7]),
    ],
)
def test_index_str_slices_a_descending_range_to_its_last_element(rng, want):
    """A numpy SLICE reads a negative stop as "count back from the end", not as its arithmetic value, so
    the `end + sign` stop that is correct for `range()` renders `A[7:-1:-1]` -- which selects NOTHING.
    A full reversal emitted a silent no-op (or a broadcast error against an ascending destination), in the
    numpy ORACLE that every bit-exactness verdict is compared against.

    Enumerated against real numpy rather than by re-deriving the formula: the formula was the bug."""
    from nestforge.ir.emit_libnode import index_str

    a = np.arange(8)
    got = eval(f"a[{index_str(dace.subsets.Range([rng]))}]", {"a": a})
    assert list(np.atleast_1d(got)) == want


def test_range_stop_refuses_a_step_of_unknown_sign():
    """No sound stop exists without a direction; guessing one silently drops or over-runs elements."""
    from nestforge.ir.emit_numpy import UnsupportedNest, range_stop

    with pytest.raises(UnsupportedNest, match="undecidable sign"):
        range_stop(symbolic.pystr_to_symbolic("N"), symbolic.pystr_to_symbolic("s"), "map parameter 'i'")


# --- nested-SDFG binding + conditional branch order ---------------------------------------------------
def test_symbol_mapping_binds_simultaneously_when_the_bindings_interfere():
    """``symbol_mapping`` is a substitution applied all at once. Emitted as ordered assignments, a swap
    ``{i: j, j: i}`` runs ``i = j`` then ``j = i`` and both end up holding the old ``j``."""
    from nestforge.ir.emit_numpy import symbol_mapping_lines

    namespace = {"i": 1, "j": 2}
    for line in symbol_mapping_lines({"i": "j", "j": "i"}, 7):
        exec(line, {}, namespace)
    assert (namespace["i"], namespace["j"]) == (2, 1)


def test_symbol_mapping_stays_plain_when_nothing_interferes():
    """Temps only where they are needed -- the plain form is what the reader and the C translator see."""
    from nestforge.ir.emit_numpy import symbol_mapping_lines

    assert symbol_mapping_lines({"a": "N", "b": "M + 1"}, 3) == ["a = N", "b = M + 1"]
    assert symbol_mapping_lines({"i": "i"}, 3) == [], "an identity binding emits nothing"


def test_a_non_final_unconditional_branch_is_refused():
    """DaCe takes the FIRST branch whose condition holds, and an unconditional branch always holds --
    so a branch stored after one is unreachable. DaCe does not merely tolerate that shape, its codegen
    REFUSES it (``Missing branch condition for non-final conditional branch``), so the emitter refuses
    it too rather than inventing an order.

    The old behaviour hoisted every unconditional branch to a trailing ``else``, which made the
    unreachable branch live and turned two unconditional branches into two ``else:`` clauses.
    """
    from dace.properties import CodeBlock
    from dace.sdfg.state import ConditionalBlock, ControlFlowRegion
    from nestforge.ir.emit_numpy import UnsupportedNest, emit_conditional

    sdfg = dace.SDFG("branch_order")
    sdfg.add_array("out", [2], dace.float64)
    block = ConditionalBlock("cond", sdfg=sdfg)
    sdfg.add_node(block, is_start_block=True)
    for label, condition in (("first", "N > 0"), ("always", None), ("dead", "N < 0")):
        body = ControlFlowRegion(label, sdfg=sdfg)
        state = body.add_state(f"{label}_state", is_start_block=True)
        tasklet = state.add_tasklet(label, {}, {"o"}, "o = 1.0")
        state.add_edge(tasklet, "o", state.add_write("out"), None, dace.Memlet("out[0]"))
        block.add_branch(None if condition is None else CodeBlock(condition), body)

    with pytest.raises(UnsupportedNest, match="unconditional branch"):
        emit_conditional(block, sdfg)


def test_int_floor_is_emitted_as_the_operator_not_a_call():
    """``int_floor`` exists because sympy mis-simplifies a floor division, not because python needs a
    helper: ``//`` is already floored for both signs. Emitting the operator keeps the numpy portable --
    a translator reads ``ast.FloorDiv``, where a bare call is an unknown name. ``int_ceil`` has no
    operator and stays a call."""
    assert normalize_casts("int_floor(a, b)") == "((a) // (b))"
    assert normalize_casts("A[int_floor(i, 2)]") == "A[((i) // (2))]"
    assert "int_ceil(" in normalize_casts("int_ceil(n, 4)")


@pytest.mark.parametrize("a,b", [(7, 2), (-7, 2), (7, -2), (-7, -2), (8, 4), (-8, 4), (0, 3)])
def test_the_operator_agrees_with_the_helper_on_both_signs(a, b):
    """The rewrite is only safe because it is the SAME function; C's `/` is where they part company."""
    rendered = normalize_casts(f"int_floor({a}, {b})")
    assert eval(rendered) == int_floor(a, b)  # noqa: S307 -- a literal expression this test built
