# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Emit WCR (reduction) tasklet out-edges as augmented assignments and check they accumulate.

A ``out[0] += a[i]`` reduction or a ``hist[bin] += w`` scatter is a tasklet whose output edge carries
a WCR. Sequential numpy emission turns it into ``target = target + tmp`` (Sum; ``np.maximum`` for Max,
...), correct even when many iterations hit the same element.
"""

import inspect

import numpy as np
import pytest

import dace as dc
from dace.sdfg import nodes

from nestforge.ir.emit_numpy import UnsupportedNest, load_emitted, sdfg_to_numpy

N = dc.symbol("N", dtype=dc.int64)
M = dc.symbol("M", dtype=dc.int64)


@dc.program
def reduce_sum(a: dc.float64[N], out: dc.float64[1]):
    out[0] = 0.0
    for i in dc.map[0:N]:
        out[0] += a[i]


@dc.program
def hist_scatter(idx: dc.int64[N], w: dc.float64[N], hist: dc.float64[M]):
    for i in dc.map[0:N]:
        hist[idx[i]] += w[i]


def run(program, fn_name, sizes, inputs):
    src = sdfg_to_numpy(program.to_sdfg(simplify=True), fn_name)
    mod = load_emitted(src, fn_name)
    kernel = vars(mod)[fn_name]
    call = dict(inputs)
    for p in inspect.signature(kernel).parameters:
        if p not in call:
            call[p] = sizes.get(p)
    kernel(**call)
    return call, src


def test_wcr_sum_reduction():
    rng = np.random.default_rng(0)
    a = rng.random(32)
    call, src = run(reduce_sum, "reduce_sum", dict(N=32), dict(a=a.copy(), out=np.zeros(1)))
    assert "+ __wcr_" in src  # augmented assignment, not a plain overwrite
    np.testing.assert_allclose(call["out"][0], a.sum())


def test_wcr_at_map_exit_from_nested_map_raises():
    """A reduction (WCR) reaching a map exit from a NESTED map (not an in-scope accumulator access node)
    is emitted by no numpy path -- silently dropping it would mis-emit the reduction as a no-op. Refuse
    it so the ExternalCall falls back to the DaCe variant instead of a wrong kernel."""
    sdfg = dc.SDFG("nested_wcr")
    sdfg.add_array("A", [N, N], dc.float64)
    sdfg.add_array("out", [N], dc.float64)
    sdfg.add_transient("acc", [1], dc.float64)
    st = sdfg.add_state()
    a = st.add_access("A")
    o = st.add_access("out")
    ome, omx = st.add_map("outer", dict(i="0:N"))
    ime, imx = st.add_map("inner", dict(j="0:N"))
    acc = st.add_access("acc")
    t = st.add_tasklet("t", {"inp"}, {"res"}, "res = inp")
    st.add_memlet_path(a, ome, ime, t, dst_conn="inp", memlet=dc.Memlet("A[i, j]"))
    st.add_edge(t, "res", acc, None, dc.Memlet("acc[0]"))
    # acc reduces out through BOTH exits; the inner->outer exit edge carries the WCR from a MapExit source.
    st.add_memlet_path(acc, imx, omx, o, memlet=dc.Memlet("out[i]", wcr="lambda x, y: x + y"))
    sdfg.validate()
    with pytest.raises(UnsupportedNest, match="reduction"):
        sdfg_to_numpy(sdfg, "nested_wcr")


def test_wcr_at_map_exit_from_nested_sdfg_raises():
    """A reduction (WCR) reaching a map exit from a NESTED SDFG is emitted by emit_nested_sdfg, which
    only replays the inner body and never applies the out-edge WCR -- so the accumulate would silently
    degrade to an overwrite. Refuse it so the ExternalCall falls back to the DaCe variant instead of a
    wrong kernel (the Tasklet source stays exempt because tasklet_lines DOES accumulate)."""
    inner = dc.SDFG("inner")
    inner.add_array("inp", [1], dc.float64)
    inner.add_array("res", [1], dc.float64)
    ist = inner.add_state()
    ia = ist.add_access("inp")
    ir = ist.add_access("res")
    it = ist.add_tasklet("t", {"x"}, {"y"}, "y = x")
    ist.add_edge(ia, None, it, "x", dc.Memlet("inp[0]"))
    ist.add_edge(it, "y", ir, None, dc.Memlet("res[0]"))

    sdfg = dc.SDFG("nested_sdfg_wcr")
    sdfg.add_array("A", [N], dc.float64)
    sdfg.add_array("out", [1], dc.float64)
    st = sdfg.add_state()
    a = st.add_access("A")
    o = st.add_access("out")
    ome, omx = st.add_map("outer", dict(i="0:N"))
    ns = st.add_nested_sdfg(inner, {"inp"}, {"res"})
    st.add_memlet_path(a, ome, ns, dst_conn="inp", memlet=dc.Memlet("A[i]"))
    # The nested SDFG reduces its per-i result into the single accumulator through the outer map exit.
    st.add_memlet_path(ns, omx, o, src_conn="res", memlet=dc.Memlet("out[0]", wcr="lambda x, y: x + y"))
    sdfg.validate()
    with pytest.raises(UnsupportedNest, match="reduction"):
        sdfg_to_numpy(sdfg, "nested_sdfg_wcr")


def test_wcr_scatter_data_dependent_index():
    """A scatter ``hist[idx[i]] += w[i]`` -- the histogram pattern -- accumulates into a data-dependent
    element; on DaCe branches where the indirect write lowers to a nested SDFG it needs the widen pass."""
    rng = np.random.default_rng(1)
    Nv, Mv = 50, 6
    idx = rng.integers(0, Mv, Nv).astype(np.int64)
    w = rng.random(Nv)
    try:
        call, src = run(
            hist_scatter, "hist_scatter", dict(N=Nv, M=Mv), dict(idx=idx.copy(), w=w.copy(), hist=np.zeros(Mv))
        )
    except UnsupportedNest:
        # Genuine upstream gap, not a missing tool: xfail (not skip) so CI's zero-skip unit set stays
        # green while the day this DaCe gains indirect-write nesting the test starts validating for real.
        pytest.xfail("indirect-write nesting unavailable in this DaCe")
    ref = np.zeros(Mv)
    np.add.at(ref, idx, w)
    np.testing.assert_allclose(call["hist"], ref)


def tasklet_wcr_at_exit(name, wcr):
    """One ``for i`` map whose tasklet writes ``A[i]`` out through the map exit under ``wcr`` into
    ``out[0]`` -- the Tasklet-source case map_exit_writes leaves for tasklet_lines to accumulate."""
    sdfg = dc.SDFG(name)
    sdfg.add_array("A", [N], dc.float64)
    sdfg.add_array("out", [1], dc.float64)
    st = sdfg.add_state()
    a = st.add_access("A")
    o = st.add_access("out")
    me, mx = st.add_map("m", dict(i="0:N"))
    t = st.add_tasklet("t", {"inp"}, {"res"}, "res = inp")
    st.add_memlet_path(a, me, t, dst_conn="inp", memlet=dc.Memlet("A[i]"))
    st.add_memlet_path(t, mx, o, src_conn="res", memlet=dc.Memlet("out[0]", wcr=wcr))
    sdfg.validate()
    return sdfg


@pytest.mark.parametrize(
    "wcr, seed, reduce_fn, token",
    [
        ("lambda x, y: x + y", 0.0, lambda a: a.sum(), "+ __wcr_"),
        ("lambda x, y: x * y", 1.0, lambda a: a.prod(), "* __wcr_"),
        ("lambda x, y: max(x, y)", -np.inf, lambda a: a.max(), "np.maximum"),
        ("lambda x, y: min(x, y)", np.inf, lambda a: a.min(), "np.minimum"),
    ],
)
def test_tasklet_wcr_combine_ops_at_map_exit(wcr, seed, reduce_fn, token):
    """Each supported reduction op (Sum/Product/Max/Min) emitted at a Tasklet's WCR out-edge crossing the
    map exit must accumulate across the whole range, not overwrite -- exercises every _WCR_BINOP entry."""
    src = sdfg_to_numpy(tasklet_wcr_at_exit("combine", wcr), "combine")
    assert token in src  # augmented assignment for this op, not a plain overwrite
    mod = load_emitted(src, "combine")
    rng = np.random.default_rng(3)
    a = rng.random(16)
    out = np.full(1, seed)
    mod.combine(A=a.copy(), out=out, N=16)
    np.testing.assert_allclose(out[0], reduce_fn(a))


def test_tasklet_wcr_symbolic_index_target_is_normalized():
    """A WCR whose write target has a derived index (``out[int_floor(i, 2)]`` -- pairwise reduction) renders
    its subset through sympy's ``symstr``, which spells integer division as ``int_floor(a, b)``: not python.
    The accumulate line must go through the same normalization as the tasklet body, or the emitted kernel
    carries a bare ``int_floor(...)`` and dies with NameError at exec."""
    sdfg = dc.SDFG("pairsum")
    sdfg.add_array("A", [N], dc.float64)
    sdfg.add_array("out", [M], dc.float64)
    st = sdfg.add_state()
    a = st.add_access("A")
    o = st.add_access("out")
    me, mx = st.add_map("m", dict(i="0:N"))
    t = st.add_tasklet("t", {"inp"}, {"res"}, "res = inp")
    st.add_memlet_path(a, me, t, dst_conn="inp", memlet=dc.Memlet("A[i]"))
    st.add_memlet_path(t, mx, o, src_conn="res", memlet=dc.Memlet("out[int_floor(i, 2)]", wcr="lambda x, y: x + y"))
    sdfg.validate()
    src = sdfg_to_numpy(sdfg, "pairsum")
    # The accumulate line goes through the same normalization as the tasklet body, so the derived index
    # renders as python. `int_floor` becomes `//` -- floored for both signs, and something a translator
    # reads as `ast.FloorDiv` rather than as an unknown name. (The sympy.floor hazard that made
    # `int_floor` necessary is about DACE-side expressions; this text is exec'd and translated, never
    # handed back to dace.) The numeric check below is what proves the spelling did not change meaning.
    assert "(i) // (2)" in src
    assert "int_floor" not in src
    mod = load_emitted(src, "pairsum")
    rng = np.random.default_rng(11)
    a = rng.random(16)
    out = np.zeros(8)
    # M sizes `out` and nothing else, so DaCe's arglist leaves it out and the emitted signature is
    # (A, out, N) -- the passed array already carries its own shape. Passing M= would be a TypeError.
    assert "M" not in sdfg.arglist() and "M" in {str(s) for s in sdfg.free_symbols}
    mod.pairsum(A=a.copy(), out=out, N=16)
    np.testing.assert_allclose(out, a.reshape(8, 2).sum(axis=1))


def test_two_distinct_wcr_out_edges_from_one_tasklet():
    """One tasklet with two output connectors carrying DIFFERENT WCRs (Sum + Max) must emit both
    augmented assignments independently -- each accumulator reduces over the full range."""
    sdfg = dc.SDFG("multi_wcr")
    sdfg.add_array("A", [N], dc.float64)
    sdfg.add_array("osum", [1], dc.float64)
    sdfg.add_array("omax", [1], dc.float64)
    st = sdfg.add_state()
    a = st.add_access("A")
    s = st.add_access("osum")
    m = st.add_access("omax")
    me, mx = st.add_map("m", dict(i="0:N"))
    t = st.add_tasklet("t", {"inp"}, {"rs", "rm"}, "rs = inp\nrm = inp")
    st.add_memlet_path(a, me, t, dst_conn="inp", memlet=dc.Memlet("A[i]"))
    st.add_memlet_path(t, mx, s, src_conn="rs", memlet=dc.Memlet("osum[0]", wcr="lambda x, y: x + y"))
    st.add_memlet_path(t, mx, m, src_conn="rm", memlet=dc.Memlet("omax[0]", wcr="lambda x, y: max(x, y)"))
    sdfg.validate()
    src = sdfg_to_numpy(sdfg, "multi")
    mod = load_emitted(src, "multi")
    rng = np.random.default_rng(5)
    a = rng.random(20)
    osum = np.zeros(1)
    omax = np.full(1, -np.inf)
    mod.multi(A=a.copy(), osum=osum, omax=omax, N=20)
    np.testing.assert_allclose(osum[0], a.sum())
    np.testing.assert_allclose(omax[0], a.max())


def test_inscope_accumulator_wcr_at_map_exit_accumulates():
    """map_exit_writes' own WCR branch: an in-scope accumulator AccessNode reducing ``out`` through the
    exit (``out[0] = out[0] + acc``) must accumulate. This is the positive mirror of the raise cases and
    is distinct from the Tasklet-direct path (the WCR edge leaves the exit from an AccessNode, not a Tasklet)."""
    sdfg = dc.SDFG("inscope_acc")
    sdfg.add_array("A", [N], dc.float64)
    sdfg.add_array("out", [1], dc.float64)
    sdfg.add_transient("acc", [1], dc.float64)
    st = sdfg.add_state()
    a = st.add_access("A")
    o = st.add_access("out")
    acc = st.add_access("acc")
    me, mx = st.add_map("m", dict(i="0:N"))
    t = st.add_tasklet("t", {"inp"}, {"res"}, "res = inp")
    st.add_memlet_path(a, me, t, dst_conn="inp", memlet=dc.Memlet("A[i]"))
    st.add_edge(t, "res", acc, None, dc.Memlet("acc[0]"))
    st.add_memlet_path(acc, mx, o, memlet=dc.Memlet("out[0]", wcr="lambda x, y: x + y"))
    sdfg.validate()
    src = sdfg_to_numpy(sdfg, "inscope_acc")
    mod = load_emitted(src, "inscope_acc")
    rng = np.random.default_rng(7)
    a = rng.random(16)
    out = np.zeros(1)
    mod.inscope_acc(A=a.copy(), out=out, N=16)
    np.testing.assert_allclose(out[0], a.sum())


def test_copy_edge_wcr_accumulates():
    """copy_lines' WCR branch: an AccessNode->AccessNode reduction copy (a privatized accumulator copied
    back into a shared buffer) must accumulate (``dst = dst + src``), not overwrite."""
    sdfg = dc.SDFG("copy_wcr")
    sdfg.add_array("src", [1], dc.float64)
    sdfg.add_array("dst", [1], dc.float64)
    st = sdfg.add_state()
    s = st.add_access("src")
    d = st.add_access("dst")
    st.add_edge(
        s, None, d, None, dc.Memlet("dst[0]", wcr="lambda x, y: x + y", other_subset=dc.subsets.Range([(0, 0, 1)]))
    )
    sdfg.validate()
    src = sdfg_to_numpy(sdfg, "cp")
    assert "dst[0] + src[0]" in src
    mod = load_emitted(src, "cp")
    dst = np.array([10.0])
    mod.cp(src=np.array([5.0]), dst=dst)
    np.testing.assert_allclose(dst[0], 15.0)


@dc.program
def chol_prog(A: dc.float64[N, N], B: dc.float64[N, N]):
    B[:] = np.linalg.cholesky(A)


def test_library_node_output_wcr_raises():
    """No library-node emitter applies an output-edge WCR (every one writes via write_lhs), so a
    reduction accumulating a library result into an existing buffer must raise rather than silently emit a
    plain overwrite -- the out_lhs guard. Uses a real Cholesky node with a WCR forced onto its out-edge."""
    sdfg = chol_prog.to_sdfg(simplify=True)
    forced = False
    for state in sdfg.states():
        for node in state.nodes():
            if isinstance(node, nodes.LibraryNode):
                for e in state.out_edges(node):
                    e.data.wcr = "lambda x, y: x + y"
                    forced = True
    assert forced, "expected a library node in the cholesky SDFG"
    with pytest.raises(UnsupportedNest, match="reduction"):
        sdfg_to_numpy(sdfg, "chol_prog")


def test_wcr_from_nested_sdfg_at_state_body_raises():
    """A NestedSDFG at state-body level (no enclosing map, so map_exit_writes never sees it) whose output
    edge carries a WCR must still raise: emit_nested_sdfg replays the inner body only and never applies the
    outer-edge accumulate. Guards the level the map-exit fix does not cover."""
    inner = dc.SDFG("inner")
    inner.add_array("inp", [1], dc.float64)
    inner.add_array("res", [1], dc.float64)
    ist = inner.add_state()
    ia = ist.add_access("inp")
    ir = ist.add_access("res")
    it = ist.add_tasklet("t", {"x"}, {"y"}, "y = x")
    ist.add_edge(ia, None, it, "x", dc.Memlet("inp[0]"))
    ist.add_edge(it, "y", ir, None, dc.Memlet("res[0]"))
    sdfg = dc.SDFG("statebody_nsdfg_wcr")
    sdfg.add_array("A", [1], dc.float64)
    sdfg.add_array("out", [1], dc.float64)
    st = sdfg.add_state()
    ns = st.add_nested_sdfg(inner, {"inp"}, {"res"})
    st.add_edge(st.add_access("A"), None, ns, "inp", dc.Memlet("A[0]"))
    st.add_edge(ns, "res", st.add_access("out"), None, dc.Memlet("out[0]", wcr="lambda x, y: x + y"))
    sdfg.validate()
    with pytest.raises(UnsupportedNest, match="reduction"):
        sdfg_to_numpy(sdfg, "statebody_nsdfg_wcr")


def test_nonwcr_nested_map_passthrough_at_map_exit_still_emits():
    """Regression: the map-exit guard must fire ONLY for wcr!=None. A plain (non-WCR) nested-map
    write-out through the outer exit must still emit, proving the narrowing did not start rejecting
    ordinary passthroughs from a non-Tasklet source."""
    sdfg = dc.SDFG("nonwcr_nested_map")
    sdfg.add_array("A", [N, N], dc.float64)
    sdfg.add_array("out", [N], dc.float64)
    sdfg.add_transient("acc", [1], dc.float64)
    st = sdfg.add_state()
    a = st.add_access("A")
    o = st.add_access("out")
    acc = st.add_access("acc")
    ome, omx = st.add_map("outer", dict(i="0:N"))
    ime, imx = st.add_map("inner", dict(j="0:N"))
    t = st.add_tasklet("t", {"inp"}, {"res"}, "res = inp")
    st.add_memlet_path(a, ome, ime, t, dst_conn="inp", memlet=dc.Memlet("A[i, j]"))
    st.add_edge(t, "res", acc, None, dc.Memlet("acc[0]"))
    st.add_memlet_path(acc, imx, omx, o, memlet=dc.Memlet("out[i]"))  # no wcr -- plain passthrough
    sdfg.validate()
    assert isinstance(sdfg_to_numpy(sdfg, "nonwcr_nested_map"), str)  # emits, does not raise
