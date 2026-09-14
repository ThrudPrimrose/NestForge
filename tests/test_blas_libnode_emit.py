# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Numpy emission for the symmetric-BLAS + LAPACK library nodes added to :mod:`nestforge.ir.emit_libnode`
(Symm / Syrk / Syr2k / Potrf), plus the explicit refusal of nodes with no faithful single-process numpy
form (MPI/pblas communication, sparse, FPGA-stream, LAPACK factorizations that output pivots).

Each BLAS case builds a one-node SDFG, runs it two ways -- the DaCe ``pure`` expansion (the reference,
compiled + run FORKED via :func:`run_isolated`) and the emitted numpy (``sdfg_to_numpy`` + ``load_emitted``) -- and
asserts they agree to machine precision. Inputs are deliberately NON-symmetric so the uplo-triangle write
(Syrk/Syr2k preserve the opposite triangle) and the full-symmetric reconstruction (Symm) are exercised.
"""
import numpy as np
import pytest

import dace
from dace.libraries.blas import Symm
from dace.libraries.blas.nodes.syrk import Syrk
from dace.libraries.blas.nodes.syr2k import Syr2k
from dace.libraries.lapack.nodes.potrf import Potrf

from nestforge.ir.emit_libnode import LIBNODE_EMITTERS, REFUSED_LIBRARY_NODES, UnsupportedLibraryNode, emit_library_node
from nestforge.ir.emit_numpy import load_emitted, sdfg_to_numpy
from nestforge.build.isolation import run_isolated

DT = dace.float64


def emit_run(build, inputs):
    """Emit the SDFG as numpy, import it, and run it on copies of ``inputs``; return the mutated buffers."""
    mod = load_emitted(sdfg_to_numpy(build(), "k"), "k")
    call = {k: v.copy() for k, v in inputs.items()}
    mod.k(**call)
    return call


def dace_reference(build, inputs):
    """Run the DaCe ``pure`` expansion of the one-node SDFG as the reference (compiled + run in a fork)."""
    sdfg = build()
    for state in sdfg.states():
        for node in state.nodes():
            if isinstance(node, dace.sdfg.nodes.LibraryNode):
                node.implementation = "pure"
    sdfg.expand_library_nodes()
    out = {k: v.copy() for k, v in inputs.items()}
    sdfg(**out)
    return out


def maxdiff(a, b):
    return max(float(np.max(np.abs(a[k].astype(np.float64) - b[k].astype(np.float64)))) for k in a)


def assert_emit_matches_reference(build, inputs, tol=1e-9):

    def work():
        ref = dace_reference(build, inputs)
        got = emit_run(build, inputs)
        return {"md": maxdiff(ref, got)}

    res = run_isolated(work, timeout=300)
    assert "error" not in res, res.get("error")
    assert res["md"] < tol, f"maxdiff {res['md']:.2e} exceeds {tol}"


def syrk_sdfg(trans, uplo, alpha, beta, n=4, k=3):
    ashape = [n, k] if trans == "N" else [k, n]
    sdfg = dace.SDFG(f"syrk_{trans}_{uplo}_{str(beta).replace('.', 'p')}")
    sdfg.add_array("A", ashape, DT)
    sdfg.add_array("C", [n, n], DT)
    st = sdfg.add_state()
    node = Syrk("syrk", trans=trans, uplo=uplo, alpha=alpha, beta=beta)
    st.add_edge(st.add_read("A"), None, node, "_a", dace.Memlet(f"A[0:{ashape[0]}, 0:{ashape[1]}]"))
    st.add_edge(node, "_c", st.add_write("C"), None, dace.Memlet(f"C[0:{n}, 0:{n}]"))
    if beta != 0:
        st.add_edge(st.add_read("C"), None, node, "_c", dace.Memlet(f"C[0:{n}, 0:{n}]"))
    return sdfg


@pytest.mark.parametrize("trans", ["N", "T"])
@pytest.mark.parametrize("uplo", ["L", "U"])
@pytest.mark.parametrize("beta", [0.0, 1.0, 0.5])
def test_syrk_matches_pure_expansion(trans, uplo, beta):
    rng = np.random.default_rng(0)
    n, k = 4, 3
    ashape = (n, k) if trans == "N" else (k, n)
    inputs = {"A": rng.random(ashape), "C": rng.random((n, n))}
    assert_emit_matches_reference(lambda: syrk_sdfg(trans, uplo, 1.5, beta), inputs)


def syr2k_sdfg(trans, uplo, alpha, beta, n=4, k=3):
    ashape = [n, k] if trans == "N" else [k, n]
    sdfg = dace.SDFG(f"syr2k_{trans}_{uplo}_{str(beta).replace('.', 'p')}")
    sdfg.add_array("A", ashape, DT)
    sdfg.add_array("B", ashape, DT)
    sdfg.add_array("C", [n, n], DT)
    st = sdfg.add_state()
    node = Syr2k("syr2k", trans=trans, uplo=uplo, alpha=alpha, beta=beta)
    st.add_edge(st.add_read("A"), None, node, "_a", dace.Memlet(f"A[0:{ashape[0]}, 0:{ashape[1]}]"))
    st.add_edge(st.add_read("B"), None, node, "_b", dace.Memlet(f"B[0:{ashape[0]}, 0:{ashape[1]}]"))
    st.add_edge(node, "_c", st.add_write("C"), None, dace.Memlet(f"C[0:{n}, 0:{n}]"))
    if beta != 0:
        st.add_edge(st.add_read("C"), None, node, "_c", dace.Memlet(f"C[0:{n}, 0:{n}]"))
    return sdfg


@pytest.mark.parametrize("trans", ["N", "T"])
@pytest.mark.parametrize("uplo", ["L", "U"])
@pytest.mark.parametrize("beta", [0.0, 1.0])
def test_syr2k_matches_pure_expansion(trans, uplo, beta):
    rng = np.random.default_rng(1)
    n, k = 4, 3
    ashape = (n, k) if trans == "N" else (k, n)
    inputs = {"A": rng.random(ashape), "B": rng.random(ashape), "C": rng.random((n, n))}
    assert_emit_matches_reference(lambda: syr2k_sdfg(trans, uplo, 1.5, beta), inputs)


def symm_sdfg(side, uplo, alpha, beta, m=4, n=3):
    sa = m if side == "L" else n
    sdfg = dace.SDFG(f"symm_{side}_{uplo}_{str(beta).replace('.', 'p')}")
    sdfg.add_array("A", [sa, sa], DT)
    sdfg.add_array("B", [m, n], DT)
    sdfg.add_array("C", [m, n], DT)
    st = sdfg.add_state()
    node = Symm("symm", side=side, uplo=uplo, alpha=alpha, beta=beta)
    st.add_edge(st.add_read("A"), None, node, "_a", dace.Memlet(f"A[0:{sa}, 0:{sa}]"))
    st.add_edge(st.add_read("B"), None, node, "_b", dace.Memlet(f"B[0:{m}, 0:{n}]"))
    st.add_edge(node, "_c", st.add_write("C"), None, dace.Memlet(f"C[0:{m}, 0:{n}]"))
    if beta != 0:
        st.add_edge(st.add_read("C"), None, node, "_c", dace.Memlet(f"C[0:{m}, 0:{n}]"))
    return sdfg


@pytest.mark.parametrize("side", ["L", "R"])
@pytest.mark.parametrize("uplo", ["L", "U"])
@pytest.mark.parametrize("beta", [0.0, 1.0])
def test_symm_matches_pure_expansion(side, uplo, beta):
    rng = np.random.default_rng(2)
    m, n = 4, 3
    sa = m if side == "L" else n
    inputs = {"A": rng.random((sa, sa)), "B": rng.random((m, n)), "C": rng.random((m, n))}
    assert_emit_matches_reference(lambda: symm_sdfg(side, uplo, 1.5, beta), inputs)


def potrf_sdfg(lower, n=5):
    sdfg = dace.SDFG(f"potrf_{lower}")
    sdfg.add_array("Ain", [n, n], DT)
    sdfg.add_array("Aout", [n, n], DT)
    sdfg.add_array("res", [1], dace.int32)
    st = sdfg.add_state()
    node = Potrf("potrf", lower=lower)
    st.add_edge(st.add_read("Ain"), None, node, "_xin", dace.Memlet(f"Ain[0:{n}, 0:{n}]"))
    st.add_edge(node, "_xout", st.add_write("Aout"), None, dace.Memlet(f"Aout[0:{n}, 0:{n}]"))
    st.add_edge(node, "_res", st.add_write("res"), None, dace.Memlet("res[0]"))
    return sdfg


@pytest.mark.parametrize("lower", [True, False])
def test_potrf_factor_reconstructs_spd(lower):
    # LAPACK has no DaCe `pure` expansion, so check the defining property instead: the emitted Cholesky
    # factor L (or U) reconstructs the SPD input (A = L@L.T lower / U.conj().T@U upper).
    rng = np.random.default_rng(3)
    n = 5
    m = rng.random((n, n))
    a = m @ m.T + n * np.eye(n)
    got = emit_run(lambda: potrf_sdfg(lower), {"Ain": a, "Aout": np.zeros((n, n)), "res": np.zeros(1, np.int32)})
    fac = got["Aout"]
    recon = fac @ fac.T if lower else fac.conj().T @ fac
    assert np.max(np.abs(recon - a)) < 1e-9
    assert got["res"][0] == 0  # success info


def test_new_blas_lapack_nodes_are_registered():
    for name in ("Symm", "Syrk", "Syr2k", "Potrf"):
        assert name in LIBNODE_EMITTERS


@pytest.mark.parametrize("conn", ["_alpha", "_beta"])
def test_gemm_runtime_coefficient_connector_is_refused(conn):
    """A Gemm carrying a runtime ``_alpha``/``_beta`` connector must be refused, not emitted from the
    compile-time properties alone: the numpy oracle and the translated C both derive from the emission, so a
    dropped runtime coefficient scales BOTH identically and maxdiff validation cannot catch it."""
    from dace.libraries.blas.nodes.gemm import Gemm
    n, m, k = 3, 5, 4
    sdfg = dace.SDFG(f"gemm_rt{conn}")
    sdfg.add_array("A", [n, k], DT)
    sdfg.add_array("B", [k, m], DT)
    sdfg.add_array("C", [n, m], DT)
    sdfg.add_array("s", [1], DT)
    st = sdfg.add_state()
    # The property stays at its neutral value, so folding it alone silently drops the runtime coefficient.
    node = Gemm("g", alpha_input=(conn == "_alpha"), beta_input=(conn == "_beta"))
    st.add_edge(st.add_read("A"), None, node, "_a", dace.Memlet(f"A[0:{n}, 0:{k}]"))
    st.add_edge(st.add_read("B"), None, node, "_b", dace.Memlet(f"B[0:{k}, 0:{m}]"))
    st.add_edge(st.add_read("s"), None, node, conn, dace.Memlet("s[0]"))
    if conn == "_beta":
        st.add_edge(st.add_read("C"), None, node, "_c", dace.Memlet(f"C[0:{n}, 0:{m}]"))
    st.add_edge(node, "_c", st.add_write("C"), None, dace.Memlet(f"C[0:{n}, 0:{m}]"))
    with pytest.raises(UnsupportedLibraryNode, match="runtime _alpha/_beta"):
        emit_library_node(node, st, sdfg)


def test_mpi_node_is_refused_by_module():
    from dace.libraries.mpi.nodes.bcast import Bcast
    sdfg = dace.SDFG("mpi")
    sdfg.add_array("x", [8], DT)
    st = sdfg.add_state()
    node = Bcast("bcast")
    st.add_edge(st.add_read("x"), None, node, "_inbuffer", dace.Memlet("x[0:8]"))
    with pytest.raises(UnsupportedLibraryNode, match="communication"):
        emit_library_node(node, st, sdfg)


def test_refused_set_names_the_unsupported_families():
    # sparse, FPGA-stream, arbitrary stencil, and pivot/packed-LU LAPACK primitives are refused by name.
    for name in ("CSRMM", "CSRMV", "Gearbox", "Stencil", "Getrf", "Getri", "Getrs"):
        assert name in REFUSED_LIBRARY_NODES
