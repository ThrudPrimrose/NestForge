# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Emit numpy operations for DaCe library nodes (BLAS / LinAlg / reductions / FFT).

Re-emitting a library node as the equivalent numpy op keeps the reference dense so hpcagent_bench's
translators recover an idiomatic kernel; only a nest with *no* library node falls back to explicit
``for`` loops (see :mod:`nestforge.ir.emit_numpy`). Operand resolution lives here so the registry stays a
flat class-name -> statement-builder table.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import sympy

import dace
from dace import symbolic
from dace.frontend.operations import detect_reduction_type
from dace.libraries.standard.nodes.scan import ScanOp
from dace.sdfg import nodes


class UnsupportedLibraryNode(Exception):
    """No numpy emission is registered for this library node class."""


# ----- operand resolution -------------------------------------------------------------------------


def index_str(subset: dace.subsets.Range, keep_singleton: bool = False) -> str:
    """Format a subset as a numpy index/slice string (end-inclusive DaCe range -> ``beg:end+1``).

    A singleton range ``(k, k, 1)`` renders as the scalar ``k`` (drops the dim, for a tasklet's scalar
    operand); ``keep_singleton`` renders it ``k:k+1`` instead so an array copy/slice preserves the dim
    per numpy semantics (a ``[N,1]`` slice stays ``[N,1]``, not ``[N]``)."""
    parts = []
    for (beg, end, step) in subset.ranges:
        if str(beg) == str(end):
            parts.append(
                f"{symbolic.symstr(beg)}:{symbolic.symstr(beg + 1)}" if keep_singleton else symbolic.symstr(beg))
        elif str(step) == "1":
            parts.append(f"{symbolic.symstr(beg)}:{symbolic.symstr(end + 1)}")
        else:
            stop = exclusive_stop(end, step)
            if stop is None:
                raise UnsupportedLibraryNode(f"subset range ({beg}, {end}, {step}) has a step of undecidable sign; "
                                             "no sound numpy slice stop")
            # A numpy SLICE reads a negative stop as "count back from the end", not as its arithmetic value:
            # `A[7:-1:-1]` selects NOTHING, so a full reversal emits a silent no-op (or a broadcast error
            # when the destination is ascending). Omitting the stop is the only way to spell "run to the
            # start of the axis". `range()` has no such rule, which is why emit_numpy.range_stop -- the same
            # end+sign arithmetic -- is correct as it stands.
            #
            # Decided ARITHMETICALLY, not by comparing the rendered text to "-1": a symbolic stop such as
            # `M - 1` renders as neither, yet reaches -1 at M == 0 and silently selects nothing -- inside
            # the numpy ORACLE every bit-exactness verdict is compared against. Indices are nonnegative and
            # canonicalization assumes symbols are too, so `stop < 0` is decidable only when it is provably
            # negative; anything that MAY go negative is refused rather than mis-emitted.
            maybe_negative = stop is not None and not (stop >= 0)
            if maybe_negative and symbolic.symstr(stop) != "-1":
                raise UnsupportedLibraryNode(
                    f"descending subset range ({beg}, {end}, {step}) has stop {symbolic.symstr(stop)}, which is not "
                    "provably >= 0; numpy would read a negative value as an offset from the END of the axis "
                    "and silently select nothing")
            if symbolic.symstr(stop) == "-1":
                parts.append(f"{symbolic.symstr(beg)}::{symbolic.symstr(step)}")
            else:
                parts.append(f"{symbolic.symstr(beg)}:{symbolic.symstr(stop)}:{symbolic.symstr(step)}")
    return ", ".join(parts)


def exclusive_stop(end: sympy.Expr, step: sympy.Expr) -> Optional[sympy.Expr]:
    """One past the last element in the direction of travel for a DaCe range whose ``end`` is INCLUSIVE;
    ``None`` when the step's sign is not decidable (no sound stop exists, so callers must refuse).

    A blanket ``end + 1`` is right only for a POSITIVE step. For the reverse read ``(4, 0, -1)`` it renders
    ``4:1:-1``, which yields three elements instead of five -- silently dropping the tail of every
    reversed/negative-stride access, including in the numpy reference that validation compares against."""
    sign = sympy.sign(sympy.sympify(step))
    if sign not in (1, -1):
        return None
    return end + sign


def covers_whole(subset: dace.subsets.Range, desc: dace.data.Data) -> bool:
    """True if ``subset`` spans the entire array descriptor (so no slice suffix is needed)."""
    if len(subset.ranges) != len(desc.shape):
        return False
    for (beg, end, step), dim in zip(subset.ranges, desc.shape):
        if str(beg) != "0" or str(step) != "1" or symbolic.symstr(end + 1) != symbolic.symstr(dim):
            return False
    return True


def is_scalar(desc: dace.data.Data) -> bool:
    """A single-element data container (a DaCe ``Scalar`` or a size-1 array)."""
    return isinstance(desc, dace.data.Scalar) or desc.total_size == 1


def scalar_local(sdfg: dace.SDFG, name: str) -> bool:
    """A *scalar transient* -- emitted as a plain python variable (a C local ``double``), not a buffer."""
    desc = sdfg.arrays[name]
    return desc.transient and is_scalar(desc)


def symbol_scalar(sdfg: dace.SDFG, name: str) -> bool:
    """A non-transient scalar the kernel references only as a FREE SYMBOL -- it never flows through a
    memlet, so the emitted code spells it as the bare ``name`` (a by-value config parameter), not the
    ``name[0]`` element of a len-1 buffer. A scalar read/written through a memlet is a len-1 buffer and
    is NOT one of these."""
    desc = sdfg.arrays[name]
    if desc.transient or not is_scalar(desc):
        return False
    return not any(e.data is not None and e.data.data == name for st in sdfg.all_states() for e in st.edges())


def scalar_elem(name: str, desc: dace.data.Data) -> str:
    """Index the SOLE element of a size-1 buffer, one index per dimension.

    ``is_scalar`` is rank-agnostic, so a rank>=2 size-1 buffer lands here too; ``name[0]`` would select a
    shape-``(1,)`` SUB-ARRAY, silently feeding a 1-D array where a scalar is meant.
    """
    rank = len(desc.shape)
    if rank <= 1:
        return f"{name}[0]"
    return f"{name}[{', '.join(['0'] * rank)}]"


def read_expr(sdfg: dace.SDFG, name: str, subset: Optional[dace.subsets.Range], keep_singleton: bool = False) -> str:
    """Read expression for ``name[subset]``: scalar-transient variable, whole array, or slice.

    Each array's data name doubles as its python variable (no connector renaming). ``None`` subset
    means the whole array. ``keep_singleton`` preserves a length-1 dim as ``k:k+1`` (see
    :func:`index_str`) -- set only when the counterpart of a copy keeps that dim.
    """
    desc = sdfg.arrays[name]
    scalar = is_scalar(desc)
    if desc.transient and scalar:
        return name
    if scalar:
        # non-transient size-1 buffer: read its element, mirroring write_lhs. The bare name is the whole
        # (1,) array, so ``s[0] = out`` is a NumPy 2 "array element with a sequence" error.
        return scalar_elem(name, desc)
    if subset is None or covers_whole(subset, desc):
        return name
    return f"{name}[{index_str(subset, keep_singleton=keep_singleton)}]"


def write_lhs(sdfg: dace.SDFG, name: str, subset: Optional[dace.subsets.Range], keep_singleton: bool = False) -> str:
    """Write target for ``name[subset]``. Arrays are written *in place* (``name[:]`` / ``name[slice]``)
    to fill the pre-allocated buffer rather than rebind it; scalar transients are plain assignments.
    ``None`` subset means the whole array. ``keep_singleton`` as in :func:`read_expr`."""
    desc = sdfg.arrays[name]
    scalar = is_scalar(desc)
    if desc.transient and scalar:
        return name
    if scalar:
        # non-transient size-1 buffer: write its element. ``name[:] = scalar`` is valid numpy but the C
        # translator mis-lowers it to ``name = scalar`` (double -> double*).
        return scalar_elem(name, desc)
    if subset is None or covers_whole(subset, desc):
        return f"{name}[:]"
    return f"{name}[{index_str(subset, keep_singleton=keep_singleton)}]"


def operand_rank(sdfg: dace.SDFG, name: str, subset: Optional[dace.subsets.Range]) -> int:
    """Rank of the operand AS RENDERED by :func:`read_expr` / :func:`write_lhs` for a library node.

    Not the buffer's rank. The emitted call operates on what the expression denotes, and the three
    renderings have three different ranks: a scalar-local or size-1 buffer collapses to a 0-d element,
    a whole-array reference keeps the descriptor's rank, and a slice keeps one axis per subset range
    (library-node operands render with ``keep_singleton=True``, so a length-1 axis SURVIVES as
    ``k:k+1``).

    Comparing descriptor ranks instead is what made :func:`emit_reduce` mis-decide ``keepdims``: it
    asked about the buffers when the call sees the slices.
    """
    desc = sdfg.arrays[name]
    if is_scalar(desc):
        return 0
    if subset is None or covers_whole(subset, desc):
        return len(desc.shape)
    return len(subset.ranges)


def memlet_expr(memlet: dace.Memlet, sdfg: dace.SDFG) -> str:
    """Read expression for a memlet's data (see :func:`read_expr`). A library-node operand keeps its
    length-1 dims so its shape matches the numpy op (a ``[N,1]`` column stays 2-D)."""
    return read_expr(sdfg, memlet.data, memlet.subset, keep_singleton=True)


def memlet_lhs(memlet: dace.Memlet, sdfg: dace.SDFG) -> str:
    """Write target for a memlet's data (see :func:`write_lhs`). Keeps length-1 dims so the target
    shape matches the numpy op's result (``acc[0:N, 0:1] = A @ mass`` for an ``[N,1]`` product)."""
    return write_lhs(sdfg, memlet.data, memlet.subset, keep_singleton=True)


def data_edge(edges: list, node: nodes.Node, kind: str) -> dace.sdfg.graph.MultiConnectorEdge:
    """The first edge that actually carries DATA. An empty memlet is a happens-before/ordering edge (added
    by StateFusion), not an operand; taking ``edges[0]`` blindly hits ``sdfg.arrays[None]`` whenever such
    an edge sorts first. Every connector-less operand lookup goes through here."""
    for e in edges:
        if not e.data.is_empty():
            return e
    raise UnsupportedLibraryNode(f"{type(node).__name__} has no data-carrying {kind} edge (only empty "
                                 "ordering edges); not emittable as numpy")


def in_conn_edge(edges: list, node: nodes.Node, conn: str) -> dace.sdfg.graph.MultiConnectorEdge:
    """The in-edge on ``conn``. Missing is a refusal, never a bare ``StopIteration``: DaCe does not wire
    every connector an emitter might name (a ``MatMul`` declares ``_a``/``_b`` only, so the ``beta``
    accumulate has no ``_c`` to read)."""
    edge = next((e for e in edges if e.dst_conn == conn), None)
    if edge is None:
        raise UnsupportedLibraryNode(f"{type(node).__name__} has no {conn!r} input connector; not emittable as numpy")
    return edge


def out_conn_edge(edges: list, node: nodes.Node, conn: str) -> dace.sdfg.graph.MultiConnectorEdge:
    """The out-edge on ``conn`` (see :func:`in_conn_edge`)."""
    edge = next((e for e in edges if e.src_conn == conn), None)
    if edge is None:
        raise UnsupportedLibraryNode(f"{type(node).__name__} has no {conn!r} output connector; not emittable as numpy")
    return edge


def in_expr(state: dace.SDFGState,
            node: nodes.Node,
            conn: Optional[str],
            sdfg: dace.SDFG,
            edges: Optional[list] = None) -> str:
    """Read expression for one input connector. ``edges`` may be a precomputed
    ``list(state.in_edges(node))`` -- pass it when a caller resolves several connectors off the same
    node, so ``state.in_edges`` is scanned once rather than once per connector."""
    edges = list(state.in_edges(node)) if edges is None else edges
    edge = data_edge(edges, node, "input") if conn is None else in_conn_edge(edges, node, conn)
    return memlet_expr(edge.data, sdfg)


def out_expr(state: dace.SDFGState,
             node: nodes.Node,
             conn: Optional[str],
             sdfg: dace.SDFG,
             edges: Optional[list] = None) -> str:
    """Read expression for the buffer an OUTPUT connector writes -- the prior value a ``beta`` accumulate
    adds into, for a node that has no matching input connector."""
    edges = list(state.out_edges(node)) if edges is None else edges
    edge = data_edge(edges, node, "output") if conn is None else out_conn_edge(edges, node, conn)
    return memlet_expr(edge.data, sdfg)


def out_lhs(state: dace.SDFGState,
            node: nodes.Node,
            conn: Optional[str],
            sdfg: dace.SDFG,
            edges: Optional[list] = None) -> str:
    """Write target for one output connector. ``edges`` as in :func:`in_expr`, for ``state.out_edges``."""
    edges = list(state.out_edges(node)) if edges is None else edges
    edge = data_edge(edges, node, "output") if conn is None else out_conn_edge(edges, node, conn)
    if edge.data.wcr is not None:
        # No library-node emitter applies an output-edge WCR (all write via write_lhs), so an accumulate
        # would silently become an overwrite. Refuse -> the ExternalCall falls back to the DaCe variant.
        raise UnsupportedLibraryNode(
            f"{type(node).__name__} output into {edge.data.data} carries a reduction (WCR) that no library-node "
            "emitter applies; not emittable as numpy -- fall back to the DaCe variant")
    return memlet_lhs(edge.data, sdfg)


# ----- per-library-node numpy statements ----------------------------------------------------------

_REDUCTION_FUNC = {
    dace.dtypes.ReductionType.Sum: "np.add",
    dace.dtypes.ReductionType.Product: "np.multiply",
    dace.dtypes.ReductionType.Max: "np.maximum",
    dace.dtypes.ReductionType.Min: "np.minimum",
    dace.dtypes.ReductionType.Logical_And: "np.logical_and",
    dace.dtypes.ReductionType.Logical_Or: "np.logical_or",
}


def is_one(v: Any) -> bool:
    """Value-aware ``v == 1`` (a ``sympy.Float(1.0)`` from a SymbolicProperty compares unequal to the
    int ``1`` under sympy's structural ``__eq__``, so a plain ``== 1`` misfires)."""
    return symbolic.equal_valued(1, v)


def is_zero(v: Any) -> bool:
    """Value-aware ``v == 0`` (see :func:`is_one`); ``str(beta) not in ('0','0.0')`` misses e.g. ``0.00``."""
    return symbolic.equal_valued(0, v)


def scaled(expr: str, coeff: Any) -> str:
    """``expr`` multiplied by ``coeff``, or ``expr`` unchanged when ``coeff`` is 1."""
    return expr if is_one(coeff) else f"{coeff} * ({expr})"


def transposed(expr: str, trans: bool) -> str:
    """``(expr).T`` when ``trans`` (a BLAS ``transA``/``transB`` flag), else ``expr``. Parenthesized so a
    slice/expression operand transposes as a whole (``(A[0:N, 0:K]).T``)."""
    return f"({expr}).T" if trans else expr


def emit_matmul(node: nodes.LibraryNode, state: dace.SDFGState, sdfg: dace.SDFG) -> str:
    """``alpha * (opA(A) @ opB(B)) + beta * C`` -- the un-specialized ``MatMul``.

    ``transA``/``transB`` are node properties here exactly as on ``Gemm`` (``FoldTransposeIntoMatMul``
    sets them); ignoring them emitted ``A @ B`` for an SDFG DaCe computes as ``A.T @ B`` -- a silently
    wrong oracle whenever A is square. The ``beta`` term reads the OUTPUT buffer's prior value, since
    ``MatMul`` declares inputs ``_a``/``_b`` only and has no ``_c`` connector to read.
    """
    in_edges, out_edges = list(state.in_edges(node)), list(state.out_edges(node))
    a = transposed(in_expr(state, node, "_a", sdfg, in_edges), node.transA)
    b = transposed(in_expr(state, node, "_b", sdfg, in_edges), node.transB)
    expr = scaled(f"{a} @ {b}", node.alpha)
    if not is_zero(node.beta):
        expr = f"{expr} + {node.beta} * {out_expr(state, node, '_c', sdfg, out_edges)}"
    return f"{out_lhs(state, node, '_c', sdfg, out_edges)} = {expr}"


def emit_gemm(node: nodes.LibraryNode, state: dace.SDFGState, sdfg: dace.SDFG) -> str:
    """``alpha * (opA(A) @ opB(B)) + beta * C`` -- the BLAS GEMM ``MatMul`` expands to, connectors
    ``_a``/``_b``/``_c`` with ``transA``/``transB`` operand transposes and scalar ``alpha``/``beta``."""
    reject_runtime_scalars(node, state)
    in_edges = list(state.in_edges(node))
    a = transposed(in_expr(state, node, "_a", sdfg, in_edges), node.transA)
    b = transposed(in_expr(state, node, "_b", sdfg, in_edges), node.transB)
    expr = scaled(f"{a} @ {b}", node.alpha)
    if not is_zero(node.beta):
        expr = f"{expr} + {node.beta} * {in_expr(state, node, '_c', sdfg, in_edges)}"
    return f"{out_lhs(state, node, '_c', sdfg)} = {expr}"


def emit_gemv(node: nodes.LibraryNode, state: dace.SDFGState, sdfg: dace.SDFG) -> str:
    """``alpha * (opA(A) @ x) + beta * y`` -- BLAS GEMV (matrix-vector), connectors ``_A``/``_x``/``_y``."""
    in_edges = list(state.in_edges(node))
    a = transposed(in_expr(state, node, "_A", sdfg, in_edges), node.transA)
    x = in_expr(state, node, "_x", sdfg, in_edges)
    expr = scaled(f"{a} @ {x}", node.alpha)
    if not is_zero(node.beta):
        expr = f"{expr} + {node.beta} * {in_expr(state, node, '_y', sdfg, in_edges)}"
    return f"{out_lhs(state, node, '_y', sdfg)} = {expr}"


def emit_ger(node: nodes.LibraryNode, state: dace.SDFGState, sdfg: dace.SDFG) -> str:
    """``alpha * outer(x, y) + A`` -- BLAS GER rank-1 update; connectors ``_x``/``_y``/``_A`` -> ``_res``."""
    in_edges = list(state.in_edges(node))
    x = in_expr(state, node, "_x", sdfg, in_edges)
    y = in_expr(state, node, "_y", sdfg, in_edges)
    a = in_expr(state, node, "_A", sdfg, in_edges)
    return f"{out_lhs(state, node, '_res', sdfg)} = {scaled(f'np.outer({x}, {y})', node.alpha)} + {a}"


def emit_axpy(node: nodes.LibraryNode, state: dace.SDFGState, sdfg: dace.SDFG) -> str:
    """``a * x + y`` -- BLAS AXPY; connectors ``_x``/``_y`` -> ``_res``."""
    in_edges = list(state.in_edges(node))
    x = in_expr(state, node, "_x", sdfg, in_edges)
    y = in_expr(state, node, "_y", sdfg, in_edges)
    return f"{out_lhs(state, node, '_res', sdfg)} = {scaled(x, node.a)} + {y}"


def emit_batched_matmul(node: nodes.LibraryNode, state: dace.SDFGState, sdfg: dace.SDFG) -> str:
    """Batched ``A @ B`` (numpy ``@`` contracts the trailing two dims, broadcasting the batch); connectors
    ``_a``/``_b`` -> ``_c``, with ``transA``/``transB`` swapping the last two axes. A non-zero ``beta`` is
    refused: there is no ``_c`` input connector to accumulate into."""
    if not is_zero(node.beta):
        raise UnsupportedLibraryNode(f"BatchedMatMul with beta={node.beta} has no _c input to accumulate")
    in_edges = list(state.in_edges(node))
    a = in_expr(state, node, "_a", sdfg, in_edges)
    b = in_expr(state, node, "_b", sdfg, in_edges)
    if node.transA:
        a = f"np.swapaxes({a}, -1, -2)"
    if node.transB:
        b = f"np.swapaxes({b}, -1, -2)"
    return f"{out_lhs(state, node, '_c', sdfg)} = {scaled(f'{a} @ {b}', node.alpha)}"


def emit_einsum(node: nodes.LibraryNode, state: dace.SDFGState, sdfg: dace.SDFG) -> str:
    """``np.einsum(einsum_str, *operands)`` -- operands ordered by connector name, since both the
    specialize expansion and ``LiftEinsum``'s ``einsum_str`` use that same sorted order.
    ``alpha``/``beta`` compose the node properties with any runtime-scalar connectors."""
    coeff = {"_alpha": str(node.alpha), "_beta": str(node.beta)}
    operands = []
    has_alpha = has_beta = False
    for e in state.in_edges(node):
        if e.data.is_empty():
            continue  # a happens-before ordering edge carries no operand; memlet_expr would hit arrays[None]
        if e.dst_conn in coeff:
            has_alpha = has_alpha or e.dst_conn == "_alpha"
            has_beta = has_beta or e.dst_conn == "_beta"
            coeff[e.dst_conn] = f"({coeff[e.dst_conn]}) * ({memlet_expr(e.data, sdfg)})"
        else:
            operands.append((e.dst_conn, memlet_expr(e.data, sdfg)))
    ordered = [expr for _, expr in sorted(operands)]
    expr = f"np.einsum('{node.einsum_str}', {', '.join(ordered)})"
    if has_alpha or not is_one(node.alpha):
        expr = f"({coeff['_alpha']}) * ({expr})"
    if has_beta or not is_zero(node.beta):
        expr = f"{expr} + ({coeff['_beta']}) * ({out_expr(state, node, None, sdfg)})"
    return f"{out_lhs(state, node, None, sdfg)} = {expr}"


def emit_tensordot(node: nodes.LibraryNode, state: dace.SDFGState, sdfg: dace.SDFG) -> str:
    """``np.tensordot(L, R, axes=(left_axes, right_axes))`` with an optional output-mode ``permutation``
    (``np.transpose`` of the contraction result); connectors ``_left_tensor``/``_right_tensor`` ->
    ``_out_tensor``."""
    in_edges = list(state.in_edges(node))
    left = in_expr(state, node, "_left_tensor", sdfg, in_edges)
    right = in_expr(state, node, "_right_tensor", sdfg, in_edges)
    expr = f"np.tensordot({left}, {right}, axes=({list(node.left_axes)}, {list(node.right_axes)}))"
    if node.permutation is not None and list(node.permutation) != list(range(len(node.permutation))):
        expr = f"np.transpose({expr}, axes={list(node.permutation)})"
    return f"{out_lhs(state, node, '_out_tensor', sdfg)} = {expr}"


def emit_inv(node: nodes.LibraryNode, state: dace.SDFGState, sdfg: dace.SDFG) -> str:
    """``np.linalg.inv(A)`` -- matrix inverse; connectors ``_ain`` -> ``_aout``."""
    return f"{out_lhs(state, node, '_aout', sdfg)} = np.linalg.inv({in_expr(state, node, '_ain', sdfg)})"


def emit_fft(node: nodes.LibraryNode, state: dace.SDFGState, sdfg: dace.SDFG) -> str:
    """``factor * np.fft.fft(x)`` -- DaCe's forward DFT is unnormalized (numpy's default ``norm``), scaled
    by the ``factor`` normalization coefficient; connectors ``_inp`` -> ``_out``."""
    inp = in_expr(state, node, "_inp", sdfg)
    return f"{out_lhs(state, node, '_out', sdfg)} = {scaled(f'np.fft.fft({inp})', node.factor)}"


def emit_ifft(node: nodes.LibraryNode, state: dace.SDFGState, sdfg: dace.SDFG) -> str:
    """``factor * np.fft.ifft(x, norm='forward')`` -- DaCe's inverse DFT omits the ``1/N`` (``norm='forward'``
    puts no scale on the inverse), scaled by ``factor``; connectors ``_inp`` -> ``_out``."""
    inp = in_expr(state, node, "_inp", sdfg)
    call = "np.fft.ifft(%s, norm='forward')" % inp
    return f"{out_lhs(state, node, '_out', sdfg)} = {scaled(call, node.factor)}"


_ARGREDUCE_FUNC = {"max": ("np.argmax", "np.max"), "min": ("np.argmin", "np.min")}


def emit_argreduce(node: nodes.LibraryNode, state: dace.SDFGState, sdfg: dace.SDFG) -> List[str]:
    """``np.argmax``/``np.argmin`` over the (contiguous) input slice -> a slice-local index plus its value;
    connector ``_in`` -> ``_out_idx`` (position) and ``_out_val`` (extreme). Two statements."""
    argfn, valfn = _ARGREDUCE_FUNC[node.op]
    inp = in_expr(state, node, "_in", sdfg)
    out_edges = list(state.out_edges(node))
    return [
        f"{out_lhs(state, node, '_out_idx', sdfg, out_edges)} = {argfn}({inp})",
        f"{out_lhs(state, node, '_out_val', sdfg, out_edges)} = {valfn}({inp})",
    ]


_SCAN_FUNC = {
    dace.dtypes.ReductionType.Sum: "np.cumsum",
    dace.dtypes.ReductionType.Product: "np.cumprod",
    dace.dtypes.ReductionType.Max: "np.maximum.accumulate",
    dace.dtypes.ReductionType.Min: "np.minimum.accumulate",
}


def emit_scan(node: nodes.LibraryNode, state: dace.SDFGState, sdfg: dace.SDFG) -> str:
    """Inclusive prefix scan -> ``np.cumsum``/``np.cumprod``/``np.maximum.accumulate``/``.minimum.``;
    connector ``_scan_in`` -> ``_scan_out``. Exclusive / seeded / strided scans have no direct numpy
    form and are refused."""
    red = {
        ScanOp.SUM: dace.dtypes.ReductionType.Sum,
        ScanOp.PRODUCT: dace.dtypes.ReductionType.Product,
        ScanOp.MIN: dace.dtypes.ReductionType.Min,
        ScanOp.MAX: dace.dtypes.ReductionType.Max
    }.get(node.op)
    func = _SCAN_FUNC.get(red)
    if func is None:
        raise UnsupportedLibraryNode(f"Scan with unsupported op {node.op}")
    if node.exclusive or str(node.stride) != "1" or "_scan_init" in node.in_connectors:
        raise UnsupportedLibraryNode("only an inclusive unit-stride unseeded Scan maps to a numpy accumulate")
    return f"{out_lhs(state, node, '_scan_out', sdfg)} = {func}({in_expr(state, node, '_scan_in', sdfg)})"


def emit_integer_sort(node: nodes.LibraryNode, state: dace.SDFGState, sdfg: dace.SDFG) -> str:
    """Ascending 1-D sort -> ``np.sort`` (numpy sorts ascending by default); connectors ``_keys_in`` ->
    ``_keys_out``."""
    return f"{out_lhs(state, node, '_keys_out', sdfg)} = np.sort({in_expr(state, node, '_keys_in', sdfg)})"


def emit_scatter_conflict_check(node: nodes.LibraryNode, state: dace.SDFGState, sdfg: dace.SDFG) -> List[str]:
    """Count duplicate values in a 1-D integer index array (scatter no-conflict proof); connector
    ``_idx_in`` -> ``_count_out`` (a host int64 scalar, ``0`` iff the index is a permutation).

    Emits the TAGCOUNT form (last-writer-wins ownership, then a mismatch count) rather than the libnode's
    sort + adjacent-equal scan; both yield ``count = N - #distinct``. The owner buffer is initialised to
    ``-1`` so index value 0 is not mistaken for a claimed slot, and temp names are suffixed by the output
    array so two nodes in one state do not share locals."""
    idx = in_expr(state, node, "_idx_in", sdfg)
    count = out_lhs(state, node, "_count_out", sdfg)
    tag = next(e for e in state.out_edges(node) if e.src_conn == "_count_out").data.data
    mx, owner, i, acc = f"__scc_{tag}_max", f"__scc_{tag}_owner", f"__scc_{tag}_i", f"__scc_{tag}_count"
    return [
        f"{mx} = int(np.max({idx}))",
        f"{owner} = np.full({mx} + 1, -1, np.int64)",
        f"for {i} in range({idx}.shape[0]):",
        f"    {owner}[{idx}[{i}]] = {i}",
        f"{acc} = 0",
        f"for {i} in range({idx}.shape[0]):",
        f"    if {owner}[{idx}[{i}]] != {i}:",
        f"        {acc} += 1",
        f"{count} = {acc}",
    ]


def out_data_name(state: dace.SDFGState, node: nodes.Node, conn: str) -> str:
    """The array NAME an output connector writes (for reading the buffer's prior value, e.g. the
    untouched triangle a symmetric BLAS update preserves)."""
    return next(e for e in state.out_edges(node) if e.src_conn == conn).data.data


def reject_runtime_scalars(node: nodes.LibraryNode, state: dace.SDFGState) -> None:
    """Refuse a BLAS node with a wired runtime ``_alpha``/``_beta`` scalar connector: the emitters fold
    only the compile-time properties, so a runtime coefficient would be silently dropped."""
    dst_conns = {e.dst_conn for e in state.in_edges(node)}
    if "_alpha" in dst_conns or "_beta" in dst_conns:
        raise UnsupportedLibraryNode(f"{type(node).__name__} has a runtime _alpha/_beta scalar connector; "
                                     "only compile-time alpha/beta are emitted -- fall back to the DaCe variant")


def triangle_funcs(uplo: str) -> Tuple[str, str, int]:
    """``(write_fn, keep_fn, keep_offset)`` for a symmetric BLAS update that touches only the ``uplo``
    triangle: the written triangle (incl. diagonal) plus the STRICT opposite triangle of the prior value,
    so the untouched half is preserved bit-for-bit (the DaCe reference leaves it unchanged)."""
    return ("np.tril", "np.triu", 1) if uplo == "L" else ("np.triu", "np.tril", -1)


def emit_syrk(node: nodes.LibraryNode, state: dace.SDFGState, sdfg: dace.SDFG) -> str:
    """BLAS SYRK: ``C := alpha*(A@A.T) + beta*C`` (``trans='N'``, ``A`` is ``N x K``) or ``alpha*(A.T@A)``
    (``trans='T'``, ``A`` is ``K x N``), updating ONLY the ``uplo`` triangle of symmetric ``C``; the
    opposite triangle keeps its prior value. Connectors ``_a``/``_c`` -> ``_c`` (in-place); ``_c`` is read
    only when ``beta != 0``."""
    reject_runtime_scalars(node, state)
    a = in_expr(state, node, "_a", sdfg)
    prod = f"{a}.T @ {a}" if node.trans == "T" else f"{a} @ {a}.T"
    rhs = scaled(f"({prod})", node.alpha)
    c_buf = out_data_name(state, node, "_c")
    if not is_zero(node.beta):
        rhs = f"{rhs} + {node.beta} * {read_expr(sdfg, c_buf, None)}"
    write, keep, off = triangle_funcs(node.uplo)
    return f"{out_lhs(state, node, '_c', sdfg)} = {write}({rhs}) + {keep}({read_expr(sdfg, c_buf, None)}, {off})"


def emit_syr2k(node: nodes.LibraryNode, state: dace.SDFGState, sdfg: dace.SDFG) -> str:
    """BLAS SYR2K: ``C := alpha*(A@B.T + B@A.T) + beta*C`` (``trans='N'``, ``A``/``B`` are ``N x K``) or
    ``alpha*(A.T@B + B.T@A) + beta*C`` (``trans='T'``, ``A``/``B`` are ``K x N``); ``A``/``B`` read in FULL
    (rectangular), only the ``uplo`` triangle of symmetric ``C`` written. Connectors ``_a``/``_b``/``_c`` ->
    ``_c``; ``_c`` read only when ``beta != 0``."""
    reject_runtime_scalars(node, state)
    in_edges = list(state.in_edges(node))
    a = in_expr(state, node, "_a", sdfg, in_edges)
    b = in_expr(state, node, "_b", sdfg, in_edges)
    prod = f"{a}.T @ {b} + {b}.T @ {a}" if node.trans == "T" else f"{a} @ {b}.T + {b} @ {a}.T"
    rhs = scaled(f"({prod})", node.alpha)
    c_buf = out_data_name(state, node, "_c")
    if not is_zero(node.beta):
        rhs = f"{rhs} + {node.beta} * {read_expr(sdfg, c_buf, None)}"
    write, keep, off = triangle_funcs(node.uplo)
    return f"{out_lhs(state, node, '_c', sdfg)} = {write}({rhs}) + {keep}({read_expr(sdfg, c_buf, None)}, {off})"


def emit_symm(node: nodes.LibraryNode, state: dace.SDFGState, sdfg: dace.SDFG) -> str:
    """BLAS SYMM: ``C := alpha*(A@B) + beta*C`` (``side='L'``) or ``alpha*(B@A) + beta*C`` (``side='R'``),
    where ``A`` is symmetric with only its ``uplo`` triangle stored -> reconstruct the FULL symmetric ``A``
    first. Output ``C`` is FULL (no triangle masking). Connectors ``_a``/``_b``/``_c`` -> ``_c``."""
    reject_runtime_scalars(node, state)
    in_edges = list(state.in_edges(node))
    a = in_expr(state, node, "_a", sdfg, in_edges)
    b = in_expr(state, node, "_b", sdfg, in_edges)
    asym = f"(np.tril({a}) + np.tril({a}, -1).T)" if node.uplo == "L" else f"(np.triu({a}) + np.triu({a}, 1).T)"
    mat = f"{asym} @ {b}" if node.side == "L" else f"{b} @ {asym}"
    rhs = scaled(mat, node.alpha)
    if not is_zero(node.beta):
        rhs = f"{rhs} + {node.beta} * {in_expr(state, node, '_c', sdfg, in_edges)}"
    return f"{out_lhs(state, node, '_c', sdfg)} = {rhs}"


def emit_potrf(node: nodes.LibraryNode, state: dace.SDFGState, sdfg: dace.SDFG) -> List[str]:
    """LAPACK POTRF (Cholesky factorization) -> ``np.linalg.cholesky`` (lower) / ``.conj().T`` (upper),
    mirroring :func:`emit_cholesky`; connectors ``_xin`` -> ``_xout`` (+ optional ``_res`` info scalar,
    always success ``0`` for a numpy reference)."""
    a = in_expr(state, node, "_xin", sdfg)
    expr = f"np.linalg.cholesky({a})"
    if not node.lower:
        expr = f"({expr}).conj().T"
    out_edges = list(state.out_edges(node))
    lines = [f"{out_lhs(state, node, '_xout', sdfg, out_edges)} = {expr}"]
    if any(e.src_conn == "_res" for e in out_edges):
        lines.append(f"{out_lhs(state, node, '_res', sdfg, out_edges)} = np.array(0, np.int32)")
    return lines


def emit_dot(node: nodes.LibraryNode, state: dace.SDFGState, sdfg: dace.SDFG) -> str:
    in_edges = list(state.in_edges(node))
    x = in_expr(state, node, "_x", sdfg, in_edges)
    y = in_expr(state, node, "_y", sdfg, in_edges)
    return f"{out_lhs(state, node, '_result', sdfg)} = np.dot({x}, {y})"


def emit_transpose(node: nodes.LibraryNode, state: dace.SDFGState, sdfg: dace.SDFG) -> str:
    inp = in_expr(state, node, "_inp", sdfg)
    return f"{out_lhs(state, node, '_out', sdfg)} = np.transpose({inp})"


def emit_solve(node: nodes.LibraryNode, state: dace.SDFGState, sdfg: dace.SDFG) -> str:
    in_edges = list(state.in_edges(node))
    ain = in_expr(state, node, "_ain", sdfg, in_edges)
    bin_ = in_expr(state, node, "_bin", sdfg, in_edges)
    return f"{out_lhs(state, node, '_bout', sdfg)} = np.linalg.solve({ain}, {bin_})"


def emit_cholesky(node: nodes.LibraryNode, state: dace.SDFGState, sdfg: dace.SDFG) -> str:
    a = in_expr(state, node, "_a", sdfg)
    # numpy returns the lower factor L (A = L @ L.conj().T); the upper factor is L.conj().T.
    expr = f"np.linalg.cholesky({a})"
    if not node.lower:
        expr = f"({expr}).conj().T"
    return f"{out_lhs(state, node, '_b', sdfg)} = {expr}"


def emit_tensortranspose(node: nodes.LibraryNode, state: dace.SDFGState, sdfg: dace.SDFG) -> str:
    inp = in_expr(state, node, "_inp_tensor", sdfg)
    if str(node.beta) not in ("0", "0.0"):
        raise UnsupportedLibraryNode(f"TensorTranspose with beta={node.beta} (accumulate) is not emitted")
    expr = f"np.transpose({inp}, axes={list(node.axes)})"
    if str(node.alpha) != "1":
        expr = f"{node.alpha} * ({expr})"
    return f"{out_lhs(state, node, '_out_tensor', sdfg)} = {expr}"


def emit_reduce(node: nodes.LibraryNode, state: dace.SDFGState, sdfg: dace.SDFG) -> str:
    red = detect_reduction_type(node.wcr)
    func = _REDUCTION_FUNC.get(red)
    if func is None:
        raise UnsupportedLibraryNode(f"Reduce with unsupported wcr {node.wcr!r} ({red})")
    in_edges, out_edges = list(state.in_edges(node)), list(state.out_edges(node))
    inp = in_expr(state, node, None, sdfg, in_edges)
    axis = None if node.axes is None else tuple(node.axes)
    # keepdims when the output keeps the reduced axis as a size-1 dimension (a numpy ``keepdims=True``
    # reduction, e.g. softmax's ``np.max(x, axis=-1, keepdims=True)``): the rendered output has the same
    # rank as the rendered input. Judged on the OPERANDS the call sees (:func:`operand_rank`), not on the
    # buffers behind them -- a whole-array read and a sliced write can share a descriptor rank while the
    # emitted expressions differ by one, and then keepdims produces a shape the target cannot hold.
    in_memlet = data_edge(in_edges, node, "input").data
    out_memlet = data_edge(out_edges, node, "output").data
    in_rank = operand_rank(sdfg, in_memlet.data, in_memlet.subset)
    out_rank = operand_rank(sdfg, out_memlet.data, out_memlet.subset)
    keepdims = axis is not None and out_rank == in_rank
    kd = ", keepdims=True" if keepdims else ""
    return f"{out_lhs(state, node, None, sdfg, out_edges)} = {func}.reduce({inp}, axis={axis}{kd})"


#: class name -> ``(node, state, sdfg) -> "lhs = rhs"`` (or a list of statements). Extend here.
LIBNODE_EMITTERS: Dict[str, Callable] = {
    "MatMul": emit_matmul,
    "Gemm": emit_gemm,
    "Gemv": emit_gemv,
    "Ger": emit_ger,
    "Axpy": emit_axpy,
    "BatchedMatMul": emit_batched_matmul,
    "Dot": emit_dot,
    "Einsum": emit_einsum,
    "TensorDot": emit_tensordot,
    "Transpose": emit_transpose,
    "TensorTranspose": emit_tensortranspose,
    "Solve": emit_solve,
    "Cholesky": emit_cholesky,
    "Inv": emit_inv,
    "Symm": emit_symm,
    "Syrk": emit_syrk,
    "Syr2k": emit_syr2k,
    "Potrf": emit_potrf,
    "FFT": emit_fft,
    "IFFT": emit_ifft,
    "Reduce": emit_reduce,
    "ArgReduce": emit_argreduce,
    "Scan": emit_scan,
    "IntegerSort": emit_integer_sort,
    "ScatterConflictCheck": emit_scatter_conflict_check,
}

#: Library nodes DELIBERATELY not emitted as numpy, each with the reason -- distinct from an
#: *unregistered* node (a genuine gap). The whole-program lane must SPLIT AROUND one of these rather than
#: abandon the program (see the MPI policy below).
REFUSED_LIBRARY_NODES: Dict[str, str] = {
    "CSRMM": "sparse CSR matrix-matrix product; not emitted as dense numpy",
    "CSRMV": "sparse CSR matrix-vector product; not emitted as dense numpy",
    "Gearbox": "FPGA stream rate-changer; operands are Streams, not arrays",
    "Stencil": "arbitrary stencil code (relative offsets + boundary conditions); not a numpy op",
    "Getrf": "LAPACK LU factorization outputs pivots (ipiv) + packed in-place LU; no pure-numpy form",
    "Getri": "LAPACK inverse-from-LU consumes packed LU + pivots; no pure-numpy form",
    "Getrs": "LAPACK solve-from-LU consumes packed LU + pivots; no pure-numpy form",
}

#: DaCe library subpackages whose nodes are DISTRIBUTED communication. None has a single-process numpy
#: equivalent; the policy is the same for all -- never offload the comm node, isolate it and offload the
#: pure-compute states around it. Matched by MODULE so a name collision cannot mis-route.
_COMM_MODULE_PREFIXES = ("dace.libraries.mpi", "dace.libraries.pblas")


def is_comm_node(node: nodes.LibraryNode) -> bool:
    """True if ``node`` is a distributed-communication library node (dace.libraries.mpi / pblas). Matched
    by MODULE, not class name: an MPI ``Reduce``/``Gather`` collides by name with a registered one."""
    return type(node).__module__.startswith(_COMM_MODULE_PREFIXES)


def is_emittable_library_node(node: nodes.LibraryNode) -> bool:
    """True iff :func:`emit_library_node` will actually emit ``node``. The single source of truth for
    "supported", shared with the split-around-unsupported pass so the two cannot disagree."""
    if is_comm_node(node):
        return False
    if type(node).__name__ in REFUSED_LIBRARY_NODES:
        return False
    return type(node).__name__ in LIBNODE_EMITTERS


def emit_library_node(node: nodes.LibraryNode, state: dace.SDFGState, sdfg: dace.SDFG) -> List[str]:
    """Numpy statement(s) for a library node, or raise if it is a communication node / deliberately
    unsupported / unregistered. A single-statement emitter returns a ``str`` (wrapped here).

    Communication and refusal are checked BEFORE the name registry: an MPI ``Reduce`` shares its class
    name with the registered standard ``Reduce``, so a name-first lookup would mis-route it."""
    cls = type(node).__name__
    if is_comm_node(node):
        raise UnsupportedLibraryNode(f"{cls} is a distributed communication node (dace.libraries.mpi/pblas); "
                                     "not emittable as single-process numpy -- isolate it in its own state and "
                                     "externalize the compute before/after it")
    if cls in REFUSED_LIBRARY_NODES:
        raise UnsupportedLibraryNode(f"{cls}: {REFUSED_LIBRARY_NODES[cls]}")
    emitter = LIBNODE_EMITTERS.get(cls)
    if emitter is None:
        raise UnsupportedLibraryNode(f"no numpy emission for library node {cls}")
    result = emitter(node, state, sdfg)
    return result if isinstance(result, list) else [result]
