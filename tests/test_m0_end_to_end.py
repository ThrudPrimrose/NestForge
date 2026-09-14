# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""M0 end-to-end: lower a map-nest to ExternalCall and run it through the DaceReference fallback, checking it
reproduces the original SDFG. The ExternCall path (a built ``lib<kernel>.a`` linked into the parent) is
covered by ``tests/test_variants_phase.py``."""

import numpy as np
import pytest
import dace

from nestforge.phases.scopes import lower_nests_to_external_call, node_boundary
from nestforge.ir.emit_numpy import nest_to_numpy
from nestforge.ir.emit_yaml import manifest_dict
from nestforge.ir.libnode import ExternalCall

N = dace.symbol("N")


@dace.program
def vadd(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N]:
        C[i] = A[i] + B[i]


@dace.program
def overwrite(A: dace.float64[N], B: dace.float64[N], X: dace.float64[N], C: dace.float64[N]):
    X[:] = A + 1
    X[:] = B + 2
    C[:] = X * 3


@dace.program
def scale_in_place(A: dace.float64[N], B: dace.float64[N]):
    for i in dace.map[0:N]:
        A[i] = A[i] * 2.0 + B[i]


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


def test_an_ordering_edge_into_a_lowered_nest_gains_no_connector():
    sdfg = overwrite.to_sdfg(simplify=True)

    lowered = lower_nests_to_external_call(sdfg)

    connectors = [
        conn
        for call, boundary in lowered
        for conn in (
            *call.in_connectors,
            *call.out_connectors,
            *(edge.dst_conn for edge in boundary.state.in_edges(call)),
            *(edge.src_conn for edge in boundary.state.out_edges(call)),
        )
    ]
    assert "_in_None" not in connectors and "_out_None" not in connectors
    ordering = [
        (edge.src.data, call.label, edge.dst_conn)
        for call, boundary in lowered
        for edge in boundary.state.in_edges(call)
        if edge.data.is_empty()
    ]
    assert ordering == [("X", "extcall_1", None)]


@pytest.mark.parametrize("program", [vadd, scale_in_place], ids=["plain", "in_place"])
def test_a_kernel_node_alone_rebuilds_the_boundary_its_manifest_and_oracle_came_from(program):
    sdfg = program.to_sdfg(simplify=True)
    ((ext, boundary),) = lower_nests_to_external_call(sdfg)

    rebuilt = node_boundary(ext)

    assert (rebuilt.inputs, rebuilt.outputs, rebuilt.symbols) == (boundary.inputs, boundary.outputs, boundary.symbols)
    assert manifest_dict(rebuilt, ext.name) == ext.config
    assert nest_to_numpy(rebuilt, fn_name=ext.name) == ext.numpy_source


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
