# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Direct numpy emission for library nodes that the corpus kernels wrap (Cholesky, TensorTranspose).

The corpus kernels that use these nodes (cholesky2, stockham_fft) each carry a *separate* structural
blocker (a nested map, loop-shaped scratch), so these minimal single-op programs exercise the library
node emitters in isolation -- the "just emit the np / np.linalg op" path.
"""
import numpy as np
import pytest

pytest.importorskip("dace")

import dace as dc

from nestforge.ir.emit_numpy import load_emitted, sdfg_to_numpy

N = dc.symbol("N", dtype=dc.int64)


@dc.program
def chol(A: dc.float64[N, N], B: dc.float64[N, N]):
    B[:] = np.linalg.cholesky(A)


@dc.program
def ttrans(X: dc.float64[2, 3, 4], Y: dc.float64[3, 2, 4]):
    Y[:] = np.transpose(X, axes=[1, 0, 2])


def emit(program, fn_name):
    src = sdfg_to_numpy(program.to_sdfg(simplify=True), fn_name)
    return vars(load_emitted(src, fn_name))[fn_name], src


def test_cholesky_libnode_emits_np_linalg_cholesky():
    fn, src = emit(chol, "chol")
    assert "np.linalg.cholesky" in src
    rng = np.random.default_rng(0)
    M = rng.random((5, 5))
    A = M @ M.T + 5 * np.eye(5)  # symmetric positive definite
    B = np.zeros((5, 5))
    fn(A=A.copy(), B=B, N=5)
    np.testing.assert_array_equal(B, np.linalg.cholesky(A))  # bit-exact


def test_tensortranspose_libnode_emits_np_transpose():
    fn, src = emit(ttrans, "ttrans")
    assert "np.transpose" in src
    rng = np.random.default_rng(1)
    X = rng.random((2, 3, 4))
    Y = np.zeros((3, 2, 4))
    fn(X=X.copy(), Y=Y)
    np.testing.assert_array_equal(Y, np.transpose(X, [1, 0, 2]))
