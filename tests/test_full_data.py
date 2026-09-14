# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Extraction passes whole arrays to the external call (no shrink/rebase to the accessed slice)."""
import numpy as np
import dace

from nestforge.phases.scopes import lower_nests_to_external_call
from nestforge.ir.libnode import ExternalCall

N = dace.symbol('N')


@dace.program
def stencil(A: dace.float64[N], B: dace.float64[N]):
    for i in dace.map[1:N - 1]:
        B[i] = A[i - 1] + A[i + 1]


def test_external_call_gets_full_array_subsets():
    sdfg = stencil.to_sdfg(simplify=True)
    ext, boundary = lower_nests_to_external_call(sdfg, strategy="outer")[0]

    # Boundary arrays keep their full parent shape, not the accessed [1:N-1] slice.
    assert str(boundary.standalone_sdfg.arrays["B"].shape[0]) == "N"
    # The kernel keeps global indices (writes B[i], not a rebased B[i-1]).
    assert "B[i]" in ext.numpy_source

    state = next(s for s in sdfg.states() if any(isinstance(n, ExternalCall) for n in s.nodes()))
    node = next(n for n in state.nodes() if isinstance(n, ExternalCall))
    for e in list(state.in_edges(node)) + list(state.out_edges(node)):
        assert str(e.data.subset) == "0:N", f"{e.data.data} passed as {e.data.subset}, want full 0:N"


def test_full_array_stencil_runs_bit_exact():
    sdfg = stencil.to_sdfg(simplify=True)
    lower_nests_to_external_call(sdfg, strategy="outer")
    sdfg.expand_library_nodes()
    sdfg.validate()
    n = 64
    A = np.random.default_rng(0).random(n)
    B = np.zeros(n)
    sdfg(A=A, B=B, N=n)
    ref = np.zeros(n)
    ref[1:n - 1] = A[0:n - 2] + A[2:n]
    np.testing.assert_array_equal(B, ref)
