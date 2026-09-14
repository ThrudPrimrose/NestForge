# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Numpy emission for the BLAS / LinAlg / standard library nodes a DaCe SDFG can carry.

Every optarena loop-nest that survives as a library node (rather than being lowered to maps) must
re-emit as the equivalent numpy op so the extracted kernel stays dense and translatable. This covers
the nodes a `@dc.program` / `auto_optimize` produces directly (MatMul/Dot/Transpose/Reduce/Solve/
Cholesky) plus the ones a partially-expanded ``MatMul`` becomes (Gemm/Gemv/BatchedMatMul/Ger) and the
explicit contraction / reduction / scan nodes (Einsum/TensorDot/ArgReduce/Scan/Inv).

Each node is built minimally, wired to arrays, emitted via :func:`sdfg_to_numpy`, executed, and checked
**bit-exact** against the numpy op it claims to be -- the emission is a rename, not an approximation.
"""
import inspect

import numpy as np
import pytest

import dace as dc
from dace import Memlet

from nestforge.ir.emit_numpy import UnsupportedNest, load_emitted, sdfg_to_numpy

N, M, K = (dc.symbol(s, dtype=dc.int64) for s in "NMK")
F = dc.float64


def build(name, node, arrays, in_wiring, out_wiring):
    """A one-state SDFG holding ``node`` wired to freshly declared arrays.

    ``arrays`` maps data name -> (shape, dtype); ``in_wiring`` / ``out_wiring`` are ``(connector, data)``
    pairs. Connectors are forced so a name used as both input and output (a GEMM's ``_c``) coexists.
    """
    sdfg = dc.SDFG(name)
    for data, (shape, dt) in arrays.items():
        sdfg.add_array(data, shape, dt)
    st = sdfg.add_state()
    st.add_node(node)
    for conn, data in in_wiring:
        node.add_in_connector(conn, force=True)
        st.add_edge(st.add_read(data), None, node, conn, Memlet.from_array(data, sdfg.arrays[data]))
    for conn, data in out_wiring:
        node.add_out_connector(conn, force=True)
        st.add_edge(node, conn, st.add_write(data), None, Memlet.from_array(data, sdfg.arrays[data]))
    sdfg.validate()
    return sdfg


def run(sdfg, fn, buffers, sizes):
    """Emit ``sdfg`` as ``def fn(...)``, import it, and call it with the buffers/sizes it actually takes
    (a size symbol absent from every shape is not in the signature). Returns the (mutated) buffers."""
    src = sdfg_to_numpy(sdfg, fn)
    kernel = vars(load_emitted(src, fn))[fn]
    params = set(inspect.signature(kernel).parameters)
    kernel(**{k: v for k, v in {**buffers, **sizes}.items() if k in params})
    return buffers, src


rng = np.random.default_rng(0)

# --- BLAS: the nodes a MatMul expands into ------------------------------------------------------- #


def test_gemm_alpha_beta():
    from dace.libraries.blas.nodes.gemm import Gemm
    g = Gemm("g")
    g.alpha, g.beta = 2.0, 3.0
    sdfg = build("gemm", g, {
        "A": ((N, K), F),
        "B": ((K, M), F),
        "C": ((N, M), F)
    }, [("_a", "A"), ("_b", "B"), ("_c", "C")], [("_c", "C")])
    A, B, C = rng.random((3, 4)), rng.random((4, 5)), rng.random((3, 5))
    buffers, src = run(sdfg, "gemm", {"A": A.copy(), "B": B.copy(), "C": C.copy()}, {"N": 3, "M": 5, "K": 4})
    assert "@" in src
    np.testing.assert_array_equal(buffers["C"], 2 * (A @ B) + 3 * C)


def test_gemm_transA_transB():
    from dace.libraries.blas.nodes.gemm import Gemm
    g = Gemm("g")
    g.transA, g.transB = True, True
    sdfg = build("gemmT", g, {
        "A": ((K, N), F),
        "B": ((M, K), F),
        "C": ((N, M), F)
    }, [("_a", "A"), ("_b", "B")], [("_c", "C")])
    A, B, C = rng.random((4, 3)), rng.random((5, 4)), np.zeros((3, 5))
    buffers, _ = run(sdfg, "gemmT", {"A": A.copy(), "B": B.copy(), "C": C}, {"N": 3, "M": 5, "K": 4})
    np.testing.assert_array_equal(buffers["C"], A.T @ B.T)


def test_gemv_alpha_beta_transA():
    from dace.libraries.blas.nodes.gemv import Gemv
    gv = Gemv("gv")
    gv.alpha, gv.beta = 2.0, 1.0
    sdfg = build("gemv", gv, {
        "A": ((N, M), F),
        "x": ((M, ), F),
        "y": ((N, ), F)
    }, [("_A", "A"), ("_x", "x"), ("_y", "y")], [("_y", "y")])
    A, x, y = rng.random((3, 4)), rng.random(4), rng.random(3)
    buffers, _ = run(sdfg, "gemv", {"A": A.copy(), "x": x.copy(), "y": y.copy()}, {"N": 3, "M": 4})
    np.testing.assert_array_equal(buffers["y"], 2 * (A @ x) + y)


def test_ger_rank1_update():
    from dace.libraries.blas.nodes.ger import Ger
    gr = Ger("gr")
    gr.alpha, gr.n, gr.m = 2.0, N, M
    sdfg = build("ger", gr, {
        "x": ((N, ), F),
        "y": ((M, ), F),
        "A": ((N, M), F),
        "res": ((N, M), F)
    }, [("_x", "x"), ("_y", "y"), ("_A", "A")], [("_res", "res")])
    x, y, A = rng.random(3), rng.random(5), rng.random((3, 5))
    buffers, src = run(sdfg, "ger", {
        "x": x.copy(),
        "y": y.copy(),
        "A": A.copy(),
        "res": np.zeros((3, 5))
    }, {
        "N": 3,
        "M": 5
    })
    assert "np.outer" in src
    np.testing.assert_array_equal(buffers["res"], 2 * np.outer(x, y) + A)


def test_axpy():
    from dace.libraries.blas.nodes.axpy import Axpy
    ax = Axpy("ax")
    ax.a, ax.n = 3.0, N
    sdfg = build("axpy", ax, {
        "x": ((N, ), F),
        "y": ((N, ), F),
        "res": ((N, ), F)
    }, [("_x", "x"), ("_y", "y")], [("_res", "res")])
    x, y = rng.random(6), rng.random(6)
    buffers, _ = run(sdfg, "axpy", {"x": x.copy(), "y": y.copy(), "res": np.zeros(6)}, {"N": 6})
    np.testing.assert_array_equal(buffers["res"], 3 * x + y)


def test_batched_matmul():
    from dace.libraries.blas.nodes.batched_matmul import BatchedMatMul
    sdfg = build("bmm", BatchedMatMul("bmm"), {
        "a": ((3, N, K), F),
        "b": ((3, K, M), F),
        "c": ((3, N, M), F)
    }, [("_a", "a"), ("_b", "b")], [("_c", "c")])
    a, b = rng.random((3, 2, 4)), rng.random((3, 4, 5))
    buffers, _ = run(sdfg, "bmm", {"a": a.copy(), "b": b.copy(), "c": np.zeros((3, 2, 5))}, {"N": 2, "M": 5, "K": 4})
    np.testing.assert_array_equal(buffers["c"], a @ b)


def test_batched_matmul_transB():
    from dace.libraries.blas.nodes.batched_matmul import BatchedMatMul
    bm = BatchedMatMul("bm")
    bm.transB = True
    sdfg = build("bmt", bm, {
        "a": ((3, N, K), F),
        "b": ((3, M, K), F),
        "c": ((3, N, M), F)
    }, [("_a", "a"), ("_b", "b")], [("_c", "c")])
    a, b = rng.random((3, 2, 4)), rng.random((3, 5, 4))
    buffers, _ = run(sdfg, "bmt", {"a": a.copy(), "b": b.copy(), "c": np.zeros((3, 2, 5))}, {"N": 2, "M": 5, "K": 4})
    np.testing.assert_array_equal(buffers["c"], a @ np.swapaxes(b, -1, -2))


def test_batched_matmul_beta_refused():
    """No ``_c`` input connector exists to accumulate into, so a non-zero beta cannot be honored."""
    from dace.libraries.blas.nodes.batched_matmul import BatchedMatMul
    bm = BatchedMatMul("bm")
    bm.beta = 1.0
    sdfg = build("bmb", bm, {
        "a": ((3, N, K), F),
        "b": ((3, K, M), F),
        "c": ((3, N, M), F)
    }, [("_a", "a"), ("_b", "b")], [("_c", "c")])
    with pytest.raises(UnsupportedNest, match="beta"):
        sdfg_to_numpy(sdfg, "bmb")


# --- Einsum: operand order is by (sorted) connector name ----------------------------------------- #


def test_einsum_three_operand():
    from dace.libraries.blas.nodes.einsum import Einsum
    es = Einsum("es")
    es.einsum_str = "ik,kj,j->i"
    sdfg = build("es", es, {
        "a": ((N, K), F),
        "b": ((K, M), F),
        "v": ((M, ), F),
        "o": ((N, ), F)
    }, [("a", "a"), ("b", "b"), ("v", "v")], [("o", "o")])
    a, b, v = rng.random((3, 4)), rng.random((4, 5)), rng.random(5)
    buffers, src = run(sdfg, "es", {
        "a": a.copy(),
        "b": b.copy(),
        "v": v.copy(),
        "o": np.zeros(3)
    }, {
        "N": 3,
        "M": 5,
        "K": 4
    })
    assert "np.einsum" in src
    np.testing.assert_array_equal(buffers["o"], np.einsum("ik,kj,j->i", a, b, v))


def test_einsum_alpha_beta_properties():
    """``out = alpha * einsum + beta * out_prior``; beta reads the output buffer in place (RHS-first)."""
    from dace.libraries.blas.nodes.einsum import Einsum
    es = Einsum("es")
    es.einsum_str, es.alpha, es.beta = "ik,kj->ij", 2.0, 3.0
    sdfg = build("esab", es, {
        "a": ((N, K), F),
        "b": ((K, M), F),
        "o": ((N, M), F)
    }, [("a", "a"), ("b", "b")], [("o", "o")])
    a, b, o = rng.random((3, 4)), rng.random((4, 5)), rng.random((3, 5))
    buffers, _ = run(sdfg, "esab", {"a": a.copy(), "b": b.copy(), "o": o.copy()}, {"N": 3, "M": 5, "K": 4})
    np.testing.assert_array_equal(buffers["o"], 2 * np.einsum("ik,kj->ij", a, b) + 3 * o)


def test_einsum_runtime_alpha_connector():
    """A data-driven ``_alpha`` scalar connector multiplies the contraction (composes with the property)."""
    from dace.libraries.blas.nodes.einsum import Einsum
    es = Einsum("es")
    es.einsum_str = "ik,kj->ij"
    sdfg = build("esco", es, {
        "a": ((N, K), F),
        "b": ((K, M), F),
        "al": ((1, ), F),
        "o": ((N, M), F)
    }, [("a", "a"), ("b", "b"), ("_alpha", "al")], [("o", "o")])
    a, b = rng.random((3, 4)), rng.random((4, 5))
    buffers, _ = run(sdfg, "esco", {
        "a": a.copy(),
        "b": b.copy(),
        "al": np.array([4.0]),
        "o": np.zeros((3, 5))
    }, {
        "N": 3,
        "M": 5,
        "K": 4
    })
    np.testing.assert_array_equal(buffers["o"], 4.0 * np.einsum("ik,kj->ij", a, b))


# --- TensorDot / Inv ----------------------------------------------------------------------------- #


def test_tensordot_contract():
    from dace.libraries.linalg.nodes.tensordot import TensorDot
    sdfg = build("td", TensorDot("td", left_axes=[2], right_axes=[0]), {
        "l": ((2, 3, 4), F),
        "r": ((4, 5), F),
        "o": ((2, 3, 5), F)
    }, [("_left_tensor", "l"), ("_right_tensor", "r")], [("_out_tensor", "o")])
    L, R = rng.random((2, 3, 4)), rng.random((4, 5))
    buffers, src = run(sdfg, "td", {"l": L.copy(), "r": R.copy(), "o": np.zeros((2, 3, 5))}, {})
    assert "np.tensordot" in src
    np.testing.assert_array_equal(buffers["o"], np.tensordot(L, R, axes=([2], [0])))


def test_tensordot_permutation():
    from dace.libraries.linalg.nodes.tensordot import TensorDot
    td = TensorDot("td", left_axes=[2], right_axes=[0])
    td.permutation = [2, 0, 1]
    sdfg = build("tdp", td, {
        "l": ((2, 3, 4), F),
        "r": ((4, 5), F),
        "o": ((5, 2, 3), F)
    }, [("_left_tensor", "l"), ("_right_tensor", "r")], [("_out_tensor", "o")])
    L, R = rng.random((2, 3, 4)), rng.random((4, 5))
    buffers, _ = run(sdfg, "tdp", {"l": L.copy(), "r": R.copy(), "o": np.zeros((5, 2, 3))}, {})
    np.testing.assert_array_equal(buffers["o"], np.transpose(np.tensordot(L, R, axes=([2], [0])), [2, 0, 1]))


def test_inv():
    from dace.libraries.linalg.nodes.inv import Inv
    sdfg = build("inv", Inv("inv"), {"ain": ((N, N), F), "aout": ((N, N), F)}, [("_ain", "ain")], [("_aout", "aout")])
    A = rng.random((4, 4)) + 4 * np.eye(4)
    buffers, src = run(sdfg, "inv", {"ain": A.copy(), "aout": np.zeros((4, 4))}, {"N": 4})
    assert "np.linalg.inv" in src
    np.testing.assert_allclose(buffers["aout"], np.linalg.inv(A), rtol=1e-12)


# --- FFT / IFFT (DaCe's forward DFT is unnormalized; its inverse omits the 1/N) ------------------ #

C128 = dc.complex128


def test_fft():
    from dace.libraries.fft.nodes.fft import FFT
    sdfg = build("fft", FFT("fft"), {"x": ((N, ), C128), "y": ((N, ), C128)}, [("_inp", "x")], [("_out", "y")])
    x = rng.random(8) + 1j * rng.random(8)
    buffers, src = run(sdfg, "fft", {"x": x.copy(), "y": np.zeros(8, complex)}, {"N": 8})
    assert "np.fft.fft" in src
    np.testing.assert_allclose(buffers["y"], np.fft.fft(x), rtol=1e-12)


def test_ifft_omits_one_over_n():
    """DaCe's IFFT is the raw inverse sum (no ``1/N``); numpy's ``ifft`` divides by N, so the match needs
    ``norm='forward'`` (== ``N * np.fft.ifft``)."""
    from dace.libraries.fft.nodes.fft import IFFT
    sdfg = build("ifft", IFFT("ifft"), {"x": ((N, ), C128), "y": ((N, ), C128)}, [("_inp", "x")], [("_out", "y")])
    x = rng.random(8) + 1j * rng.random(8)
    buffers, src = run(sdfg, "ifft", {"x": x.copy(), "y": np.zeros(8, complex)}, {"N": 8})
    assert "norm='forward'" in src
    np.testing.assert_allclose(buffers["y"], np.fft.ifft(x, norm="forward"), rtol=1e-12)


def test_fft_factor_normalization():
    from dace.libraries.fft.nodes.fft import IFFT
    ifft = IFFT("ifft")
    ifft.factor = 0.125  # 1/N normalization folded into the coefficient -> matches numpy's plain ifft
    sdfg = build("ifftn", ifft, {"x": ((N, ), C128), "y": ((N, ), C128)}, [("_inp", "x")], [("_out", "y")])
    x = rng.random(8) + 1j * rng.random(8)
    buffers, _ = run(sdfg, "ifftn", {"x": x.copy(), "y": np.zeros(8, complex)}, {"N": 8})
    np.testing.assert_allclose(buffers["y"], np.fft.ifft(x), rtol=1e-12)


# --- standard: ArgReduce / Scan ------------------------------------------------------------------ #


@pytest.mark.parametrize("op, argfn, valfn", [("max", np.argmax, np.max), ("min", np.argmin, np.min)])
def test_argreduce(op, argfn, valfn):
    from dace.libraries.standard.nodes.arg_reduce import ArgReduce
    sdfg = build("ar", ArgReduce("ar", op=op), {
        "inp": ((N, ), F),
        "val": ((1, ), F),
        "idx": ((1, ), dc.int64)
    }, [("_in", "inp")], [("_out_val", "val"), ("_out_idx", "idx")])
    inp = rng.random(7)
    buffers, _ = run(sdfg, "ar", {"inp": inp.copy(), "val": np.zeros(1), "idx": np.zeros(1, np.int64)}, {"N": 7})
    assert buffers["idx"][0] == argfn(inp)
    np.testing.assert_array_equal(buffers["val"][0], valfn(inp))


@pytest.mark.parametrize("scanop, ref", [("SUM", np.cumsum), ("PRODUCT", np.cumprod), ("MAX", np.maximum.accumulate),
                                         ("MIN", np.minimum.accumulate)])
def test_scan_inclusive(scanop, ref):
    from dace.libraries.standard.nodes.scan import Scan, ScanOp
    sdfg = build("sc", Scan("sc", op=ScanOp[scanop]), {
        "si": ((N, ), F),
        "so": ((N, ), F)
    }, [("_scan_in", "si")], [("_scan_out", "so")])
    si = rng.random(6)
    buffers, _ = run(sdfg, "sc", {"si": si.copy(), "so": np.zeros(6)}, {"N": 6})
    np.testing.assert_array_equal(buffers["so"], ref(si))


def test_scan_exclusive_refused():
    from dace.libraries.standard.nodes.scan import Scan, ScanOp
    sc = Scan("sc", op=ScanOp.SUM)
    sc.exclusive = True
    sdfg = build("scx", sc, {"si": ((N, ), F), "so": ((N, ), F)}, [("_scan_in", "si")], [("_scan_out", "so")])
    with pytest.raises(UnsupportedNest, match="inclusive"):
        sdfg_to_numpy(sdfg, "scx")


def test_integer_sort():
    from dace.libraries.sort.nodes.integer_sort import IntegerSort
    sdfg = build("srt", IntegerSort("srt"), {
        "ki": ((N, ), dc.int64),
        "ko": ((N, ), dc.int64)
    }, [("_keys_in", "ki")], [("_keys_out", "ko")])
    ki = rng.integers(0, 1000, size=9).astype(np.int64)
    buffers, src = run(sdfg, "srt", {"ki": ki.copy(), "ko": np.zeros(9, np.int64)}, {"N": 9})
    assert "np.sort" in src
    np.testing.assert_array_equal(buffers["ko"], np.sort(ki))


# --- ScatterConflictCheck: TAGCOUNT duplicate count (0 iff a permutation) ------------------------ #


def build_scatter_conflict_check(name):
    from dace.libraries.sort.nodes.scatter_conflict_check import ScatterConflictCheck
    return build(name, ScatterConflictCheck(name), {
        "idx": ((N, ), dc.int64),
        "cnt": ((1, ), dc.int64)
    }, [("_idx_in", "idx")], [("_count_out", "cnt")])


def test_scatter_conflict_check_permutation():
    """A permutation has no duplicate values, so ``count == 0`` (the scatter is conflict-free)."""
    sdfg = build_scatter_conflict_check("sccp")
    idx = rng.permutation(9).astype(np.int64)
    buffers, src = run(sdfg, "sccp", {"idx": idx.copy(), "cnt": np.zeros(1, np.int64)}, {"N": 9})
    assert "np.full" in src  # TAGCOUNT ownership buffer, not a sort
    assert buffers["cnt"][0] == 0
    assert buffers["cnt"][0] == idx.shape[0] - len(np.unique(idx))


def test_scatter_conflict_check_duplicates():
    """With duplicates, ``count == N - #distinct`` -- matching the libnode's sort + adjacent-equal scan."""
    sdfg = build_scatter_conflict_check("sccd")
    idx = np.array([0, 2, 2, 5, 5, 5, 1, 9, 9], dtype=np.int64)  # 9 elems, 5 distinct -> 4 duplicates
    buffers, _ = run(sdfg, "sccd", {"idx": idx.copy(), "cnt": np.zeros(1, np.int64)}, {"N": 9})
    assert buffers["cnt"][0] == 4
    assert buffers["cnt"][0] == idx.shape[0] - len(np.unique(idx))
