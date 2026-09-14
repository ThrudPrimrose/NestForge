# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""M0 end-to-end: lower a map-nest to ExternalCall and run it through the DaceReference fallback, checking it
reproduces the original SDFG. The ExternCall path (a built ``lib<kernel>.a`` linked into the parent) is
covered by ``tests/test_variants_phase.py``."""
import numpy as np
import dace

from nestforge.phases.scopes import lower_nests_to_external_call
from nestforge.ir.libnode import ExternalCall

N = dace.symbol('N')


@dace.program
def vadd(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N]:
        C[i] = A[i] + B[i]


def reference_outputs(n):
    A = np.random.default_rng(0).random(n)
    B = np.random.default_rng(1).random(n)
    return A, B, A + B


def test_lower_inserts_external_call():
    sdfg = vadd.to_sdfg(simplify=True)
    lowered = lower_nests_to_external_call(sdfg)
    assert len(lowered) == 1
    ext, boundary = lowered[0]
    assert isinstance(ext, ExternalCall)
    assert set(ext.in_connectors) == {"_in_A", "_in_B"}
    assert set(ext.out_connectors) == {"_out_C"}
    assert "def " in ext.numpy_source


def test_dace_reference_runs_correctly():
    sdfg = vadd.to_sdfg(simplify=True)
    lower_nests_to_external_call(sdfg)  # default impl = DaceReference
    sdfg.expand_library_nodes()
    sdfg.validate()
    n = 1 << 12
    A, B, ref = reference_outputs(n)
    C = np.zeros(n)
    sdfg(A=A, B=B, C=C, N=n)
    np.testing.assert_allclose(C, ref)
