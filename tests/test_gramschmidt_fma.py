# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""gramschmidt is the worked example for the FP-mode axis of the arena (see docs/fp-and-vectorization.md).

Its two ``np.dot`` reductions (``nrm = A[:,k].A[:,k]`` and ``R[k,j] = Q[:,k].A[:,j]``) each lower to a
sequential accumulate ``s += a[i]*b[i]``. Under ``-ffast-math`` the reduction is *reassociated*
(split into vector partial sums), which changes the accumulation order. Whether that is dangerous is
gated by the input's condition number, exactly as the summation/solver theory predicts:

  * well-conditioned A  -> every mode agrees to ~machine-epsilon (reassociation is benign, kappa~1);
  * ill-conditioned  A  -> a near-zero pivot ``R[k,k]`` divides ``A[:,k]``, amplifying the tiny
    reassociation difference by ~1/R[k,k], so ``-ffast-math`` diverges by many orders of magnitude
    while ieee-strict stays controlled.

"Numerically stable" (the arena's acceptance metric) = relative error vs the ieee-strict *sequential*
baseline does not explode. This test encodes both the mechanism and that stability definition, and
guards the size-1-buffer write fix that lets the reduction nest compile (``nrm[0] =`` not ``nrm[:] =``).
"""

import ctypes
import re
import shutil
import subprocess

import numpy as np

gcc = shutil.which("gcc")
assert gcc is not None, "gcc not on PATH (setup_apt.sh installs it)"

from dace import symbolic

from nestforge.corpus.bench import iter_dace_kernels
from nestforge.phases.scopes import lower_nests_to_external_call
from nestforge.corpus.translate import prepare, emit_sources

_CT = {"float64": ctypes.c_double, "int64": ctypes.c_int64}
_BASE = ["-O3", "-march=native", "-fPIC", "-shared"]
_MODES = {
    "ieee-strict-seq": ["-ffp-contract=off", "-fno-tree-vectorize"],  # the stability baseline
    "contract-fast": ["-ffp-contract=fast"],  # FMA only, no reassociation
    "fast-math": ["-ffast-math"],  # adds reduction reassociation
}


def make_A(M, N, conditioning, seed=0):
    rng = np.random.default_rng(seed)
    if conditioning == "well":
        return rng.standard_normal((M, N))
    # singular values 1e0..1e-14 -> cond ~1e14: Gram-Schmidt hits near-zero pivots.
    U, _ = np.linalg.qr(rng.standard_normal((M, N)))
    V, _ = np.linalg.qr(rng.standard_normal((N, N)))
    return U @ np.diag(np.logspace(0, -14, N)) @ V


def prepare_compute_nest():
    kernels = {k.short_name: k for k in iter_dace_kernels()}
    sdfg = kernels["scientific_computing/dense_linear_algebra/gramschmidt/gramschmidt"].to_sdfg(simplify=True)
    # simplify folds the two zero-inits (Q, R) into the compute nest, so ``outer`` returns a single nest
    # -- the one holding the two np.dot reductions (asserted via ``nrm[0] = np.dot`` in ``emit``).
    nests = lower_nests_to_external_call(sdfg)
    assert len(nests) == 1, f"expected one compute nest, got {len(nests)}"
    _, boundary = nests[0]
    return boundary


def emit(boundary, tmp_path):
    prep = prepare(boundary, "gs_compute", tmp_path / "kern")
    csrc = next(p for p in emit_sources(prep, tmp_path / "gen") if p.suffix == ".c" and "pluto" not in p.name)
    ctext = csrc.read_text()
    # ABI order is whatever the translator emits (it reorders vs the manifest), so read it back.
    sig = re.search(r"void\s+gs_compute_fp64\s*\((.*?)\)\s*\{", ctext, re.S).group(1)
    order = [p.strip().split()[-1].lstrip("*") for p in sig.split(",")]
    return prep, csrc, order


def run(csrc, order, boundary, flags, A, sizes, tmp_path, tag):
    bsdfg = boundary.standalone_sdfg
    env = {symbolic.symbol("M"): sizes["M"], symbolic.symbol("N"): sizes["N"]}
    buffers = {}
    for a in order:
        if a in bsdfg.arrays:
            d = bsdfg.arrays[a]
            shape = tuple(int(symbolic.evaluate(x, env)) for x in d.shape)
            buffers[a] = A.copy() if a == "A" else np.zeros(shape, np.dtype(d.dtype.type))
    argt = [
        ctypes.POINTER(_CT[np.dtype(bsdfg.arrays[a].dtype.type).name]) if a in bsdfg.arrays else ctypes.c_int64
        for a in order
    ]
    so = tmp_path / f"lib_{tag}.so"
    subprocess.run([gcc, *_BASE, *flags, str(csrc), "-o", str(so)], check=True, capture_output=True)
    fn = ctypes.CDLL(str(so)).gs_compute_fp64
    fn.argtypes = argt
    fn.restype = None
    fn(*[buffers[a].ctypes.data_as(t) if a in buffers else ctypes.c_int64(sizes[a]) for a, t in zip(order, argt)])
    # gramschmidt is functional->in-place: its two outputs land in the named ``Q`` (orthonormal) and ``R``
    # (upper-triangular) buffers, not DaCe ``__return`` values. The relerr stability metric compares both.
    return {o: buffers[o].copy() for o in ("Q", "R")}


def relerr(got, ref):
    num = max(float(np.max(np.abs(got[k] - ref[k]))) for k in got)
    den = max(float(np.max(np.abs(ref[k]))) for k in got) or 1.0
    return num / den


def sweep(conditioning, tmp_path):
    boundary = prepare_compute_nest()
    prep, csrc, order = emit(boundary, tmp_path)
    assert "nrm[0] = np.dot" in prep.numpy_source  # size-1 buffer written by element, not nrm[:] =
    M, N = 128, 40
    sizes = {"M": M, "N": N, "j": 0, "k": 0}
    A = make_A(M, N, conditioning)
    outs = {
        m: run(csrc, order, boundary, flags, A, sizes, tmp_path, f"{conditioning}_{m}") for m, flags in _MODES.items()
    }
    ref = outs["ieee-strict-seq"]
    return {m: relerr(outs[m], ref) for m in _MODES}


def test_fma_contraction_alone_is_bit_exact(tmp_path):
    """FMA contraction (``-ffp-contract=fast``, no reassociation) matches ieee-strict bit-for-bit here
    -- so the divergence below is due to reassociation, not FMA. Refines "FMA is the danger" to
    "reassociation is the danger"."""
    err = sweep("well", tmp_path)
    assert err["contract-fast"] == 0.0


def test_fastmath_is_stable_when_well_conditioned(tmp_path):
    """Well-conditioned: reassociation only shuffles O(n*eps) benign noise (kappa~1) -- stable."""
    err = sweep("well", tmp_path)
    assert err["fast-math"] < 1e-12  # ~machine epsilon, not exploding


def test_fastmath_diverges_when_ill_conditioned(tmp_path):
    """Ill-conditioned (cond~1e14): the near-zero pivot amplifies the reassociated-dot difference.
    Same kernel, same flag -- the danger appears only because the input is ill-conditioned. This is
    the empirical evidence for weighting reduction-reassociation risk by condition number."""
    well = sweep("well", tmp_path / "well")
    ill = sweep("ill", tmp_path / "ill")
    # ieee-strict-seq is the baseline (0 by construction); fast-math must blow up only when ill.
    assert ill["fast-math"] > 1e-6
    assert ill["fast-math"] > well["fast-math"] * 1e6  # many orders worse than the well-conditioned run
