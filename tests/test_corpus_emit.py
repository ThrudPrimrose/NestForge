# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Emit numpy for real optarena dace kernels and check the library-node ops compute correctly.

Emission is C-style: the kernel allocates nothing, so the caller pre-allocates every buffer -- inputs,
outputs, the DaCe ``__return`` value, and scratch transients -- and reads the results back out of the
in-place buffers. ``alloc_run`` does exactly that, driven by the emitted function's own signature.
"""
import inspect

import numpy as np
import pytest

pytest.importorskip("hpcagent_bench")

from dace import symbolic

from nestforge.corpus.bench import dace_kernel_names, iter_dace_kernels
from nestforge.ir.emit_libnode import symbol_scalar
from nestforge.ir.emit_numpy import load_emitted, maxsize_loop_scratch, sdfg_to_numpy
from nestforge.build.isolation import run_isolated


def kernels():
    return {k.short_name: k for k in iter_dace_kernels()}


def test_emit_numpy_labels_regions_and_states():
    """Emitted numpy carries ``# loop region (label)`` / ``# state (label)`` provenance comments and stays
    valid python -- a block that emits only a comment (empty state/loop) must still get a ``pass``."""
    import ast

    from nestforge.ir.emit_numpy import body_or_pass
    src = sdfg_to_numpy(kernels()["hpc/map_reduce/azimint_hist/azimint_hist"].to_sdfg(simplify=True), "k")
    ast.parse(src)  # valid python despite the interleaved comments
    assert "# loop region (" in src
    assert "# state (" in src
    assert body_or_pass(["# state (X)"]) == ["# state (X)", "pass"]  # comment-only body gets a pass
    assert body_or_pass(["x = 1"]) == ["x = 1"]  # a real statement is left untouched


def alloc_run(short, fn_name, sizes, inputs, seed=0, sdfg=None):
    """Emit ``short``, allocate every buffer parameter C-style, run it, return the buffer dict.

    ``sdfg`` lets a caller that must guard the BUILD separately (see the nbody test) hand in the SDFG it
    already built, so the build is not repeated here.
    """
    if sdfg is None:
        sdfg = kernels()[short].to_sdfg(simplify=True)
    src = sdfg_to_numpy(sdfg, fn_name)
    kernel = vars(load_emitted(src, fn_name))[fn_name]
    # size loop-shaped scratch exactly as the emitter widened it (a decreasing extent like M-i-1
    # widens to its i=0 value, not to a naive max) so buffers match the emitted signature.
    symbols = [a for a in sdfg.arglist() if a not in sdfg.arrays]
    sized = maxsize_loop_scratch(sdfg, symbols)
    env = {symbolic.symbol(k): v for k, v in sizes.items()}
    # a symbol-only scalar (referenced by the kernel as a bare name, never through a memlet) is a by-value
    # config parameter; pass a python scalar. Every other scalar is a len-1 buffer (name[0]).
    by_value = {n for n in sized.arrays if symbol_scalar(sdfg, n)}
    call = {}
    for name in inspect.signature(kernel).parameters:
        if name in sizes:
            call[name] = sizes[name]
            continue
        desc = sized.arrays[name]
        dt = np.dtype(desc.dtype.type)
        if name in by_value:
            call[name] = dt.type(inputs[name]) if name in inputs else dt.type(0)
            continue
        shape = tuple(int(symbolic.evaluate(d, env)) for d in desc.shape)
        # reshape inputs to the descriptor shape: a kernel may declare a param 2-D (nbody's mass is
        # [N,1]) while the caller supplies the flat (N,) array; sizes match, so reshape aligns them.
        call[name] = inputs[name].astype(dt).reshape(shape) if name in inputs else np.zeros(shape, dt)
    kernel(**call)
    return call, src


def test_corpus_exposes_dace_kernels():
    # optarena ships each _dace.py as a gitignored, regenerated-on-demand artifact
    # (nestforge.corpus.materialize_dace_corpus); the corpus exposes every hpc/ml kernel whose numpy
    # reference numpyto can lower to dace. That emittable set grows as the translator improves, so assert a
    # floor plus the specific kernels this suite exercises -- not a brittle exact count tied to one machine's
    # partial generation.
    hpc = set(dace_kernel_names("hpc"))
    assert len(hpc) >= 50, len(hpc)
    assert {
        "hpc/dense_linear_algebra/gemm/gemm",
        "hpc/structured_grids/jacobi_1d/jacobi_1d",
        "hpc/dense_linear_algebra/lu/lu",
    } <= hpc
    assert len(dace_kernel_names("ml")) >= 5, len(dace_kernel_names("ml"))


def test_foundation_track_reachable_via_iter_dace_kernels():
    """Guards against foundation silently yielding 0 kernels again (it did until DACE_TRACKS included
    it): sample the first two non-TSVC foundation kernels and check each still builds to an SDFG.
    ``to_sdfg`` runs isolated -- a freshly-generated ``_dace.py`` can crash the codegen it drives."""
    foundation = sorted((k for k in iter_dace_kernels("foundation") if "tsvc" not in k.short_name),
                        key=lambda k: k.short_name)
    assert foundation, "foundation track yielded no kernels"
    for kernel in foundation[:2]:
        result = run_isolated(lambda k=kernel: {"ok": k.program().to_sdfg(simplify=True) is not None})
        assert result.get("ok"), f"{kernel.short_name}: {result.get('error')}"


def test_gemm_matmul_emits_and_computes():
    rng = np.random.default_rng(0)
    NI, NJ, NK = 8, 6, 5
    A, B, C = rng.random((NI, NK)), rng.random((NK, NJ)), rng.random((NI, NJ))
    alpha, beta = np.array([1.5]), np.array([2.0])
    ref = alpha[0] * A @ B + beta[0] * C
    call, src = alloc_run("hpc/dense_linear_algebra/gemm/gemm", "gemm", dict(NI=NI, NJ=NJ, NK=NK),
                          dict(A=A, B=B, C=C.copy(), alpha=alpha, beta=beta))
    assert "@" in src and "np.empty" not in src  # MatMul -> numpy matmul, no internal allocation
    np.testing.assert_allclose(call["C"], ref)


def test_atax_return_value_is_inplace_buffer():
    rng = np.random.default_rng(1)
    M, N = 7, 9
    A, x = rng.random((M, N)), rng.random(N)
    call, src = alloc_run("hpc/dense_linear_algebra/atax/atax", "atax", dict(M=M, N=N), dict(A=A, x=x))
    # atax is functional->in-place: its result lands in a named ``out`` buffer param, not a DaCe
    # ``__return`` value. Still C-style -- no python ``return``; the caller pre-allocates ``out``.
    assert "return " not in src and "out" in call
    np.testing.assert_allclose(call["out"], (A @ x) @ A)


def test_nussinov_conditionalblock_emits_and_computes():
    """nussinov exercises the full control-flow path at once: ``ConditionalBlock`` (if/elif/else),
    access-node scalar copies, an inter-state indirect index, and ``dace.<cast>`` normalization."""
    N = 20
    rng = np.random.default_rng(0)
    seq = rng.integers(0, 4, size=N).astype(np.int32)
    # complement_sum / pair_bonus are read-only config scalars (a complementary base pair sums to 3 and
    # scores 1); the emit takes them by value, so supply them -- unsupplied, they default to 0 and the DP
    # never scores a pairing.
    call, src = alloc_run("hpc/dynamic_programming/nussinov/nussinov", "nussinov", dict(N=N),
                          dict(seq=seq, complement_sum=3, pair_bonus=1))
    assert "if " in src and "else:" in src  # the guards lower through ConditionalBlock

    def match(b1, b2):
        return 1 if b1 + b2 == 3 else 0

    t = np.zeros((N, N), np.int32)
    for i in range(N - 1, -1, -1):
        for j in range(i + 1, N):
            if j - 1 >= 0:
                t[i, j] = np.maximum(t[i, j], t[i, j - 1])
            if i + 1 < N:
                t[i, j] = np.maximum(t[i, j], t[i + 1, j])
            if j - 1 >= 0 and i + 1 < N:
                if i < j - 1:
                    t[i, j] = np.maximum(t[i, j], t[i + 1, j - 1] + match(seq[i], seq[j]))
                else:
                    t[i, j] = np.maximum(t[i, j], t[i + 1, j - 1])
            for k in range(i + 1, j):
                t[i, j] = np.maximum(t[i, j], t[i, k] + t[k + 1, j])
    np.testing.assert_array_equal(call["table"],
                                  t)  # functional->in-place: DP table is a named buffer; int DP -> bit-exact


def test_contour_integral_two_returns_solve_and_indirect_negate():
    """contour_integral returns two buffers and mixes ``np.linalg.solve``, a conditional negate, and
    a ``dace.complex128`` power cast -- a good end-to-end check of the complex + multi-output path."""
    NR, NM, slab = 5, 3, 2
    rng = np.random.default_rng(1)
    crand = lambda shape: (rng.random(shape) + 1j * rng.random(shape)).astype(np.complex128)
    Ham, int_pts, Y = crand((slab + 1, NR, NR)), crand((32, )), crand((NR, NM))
    # contour_radius is a read-only config scalar (the reference's 1.0 default = the unit circle); dace lowers
    # it to a required by-value param -- the python kwarg default is lost -- so supply it, else it reads 0 and
    # the |z| < radius residue negate never fires (garbage vs the oracle below, which uses 1.0).
    call, src = alloc_run("hpc/dense_linear_algebra/contour_integral/contour_integral", "contour_integral",
                          dict(NR=NR, NM=NM, slab_per_bc=slab, num_int_pts=32),
                          dict(Ham=Ham, int_pts=int_pts, Y=Y, contour_radius=1.0))
    assert "np.complex128(" in src and "dace." not in src  # casts normalized to numpy

    P0 = np.zeros((NR, NM), np.complex128)
    P1 = np.zeros((NR, NM), np.complex128)
    for idx in range(32):
        z = int_pts[idx]
        Tz = np.zeros((NR, NR), np.complex128)
        for n in range(slab + 1):
            Tz += np.power(z, slab / 2 - n) * Ham[n]
        X = np.linalg.solve(Tz, Y)
        if np.absolute(z) < 1.0:
            X = -X
        P0 += X
        P1 += z * X
    np.testing.assert_allclose(call["P0"], P0)  # P0/P1 are in-place accumulator outputs (zero-init by alloc_run)
    np.testing.assert_allclose(call["P1"], P1)


def test_mandelbrot_nested_sdfg_in_map_emits_and_computes():
    """mandelbrot's per-pixel escape is a ``np.where`` -> a nested SDFG with inner control flow inside
    a 2-D map. It emits only after ``ExpandNestedSDFGInputs`` widens the nest to full arrays, so the
    masked ``if I[j,k]: Z[j,k] = Z[j,k]**2 + C[j,k]`` writes the outer buffer in place."""
    pytest.importorskip("dace.transformation.interstate.expand_nested_sdfg_inputs")
    XN, YN = 20, 16
    scal = dict(xmin=-2.0, xmax=0.5, ymin=-1.25, ymax=1.25, maxiter=25, horizon=2.0)
    sdfg = kernels()["hpc/map_reduce/mandelbrot1/mandelbrot1"].to_sdfg(simplify=True)
    src = sdfg_to_numpy(sdfg, "mandelbrot")
    mod = load_emitted(src, "mandelbrot")
    env = {symbolic.symbol("xn"): XN, symbolic.symbol("yn"): YN}
    call = {}
    for name in inspect.signature(mod.mandelbrot).parameters:
        if name in ("xn", "yn"):
            call[name] = {"xn": XN, "yn": YN}[name]
        elif name in scal:
            d = sdfg.arrays.get(name)
            call[name] = np.array([scal[name]], np.dtype(d.dtype.type)) if d is not None else scal[name]
        else:
            d = sdfg.arrays[name]
            shape = tuple(int(symbolic.evaluate(x, env)) for x in d.shape)
            call[name] = np.zeros(shape, np.dtype(d.dtype.type))
    mod.mandelbrot(**call)

    X = scal["xmin"] + np.arange(XN) * ((scal["xmax"] - scal["xmin"]) / (XN - 1))
    Y = scal["ymin"] + np.arange(YN) * ((scal["ymax"] - scal["ymin"]) / (YN - 1))
    C = X[None, :] + Y[:, None] * 1j
    Nc = np.zeros(C.shape, np.int64)
    Z = np.zeros(C.shape, np.complex128)
    for n in range(scal["maxiter"]):
        I = np.less(np.absolute(Z), scal["horizon"])
        Nc[I] = n
        for j in range(YN):
            for k in range(XN):
                if I[j, k]:
                    Z[j, k] = Z[j, k]**2 + C[j, k]
    Nc[Nc == scal["maxiter"] - 1] = 0
    np.testing.assert_array_equal(call["N_out"], Nc)
    np.testing.assert_array_equal(call["Z_out"], Z)  # bit-exact


def test_emission_does_not_mutate_caller_sdfg():
    """``sdfg_to_numpy`` must be read-only: widening nested SDFGs runs on a copy, so a caller that
    inspects or compiles the same SDFG afterwards (e.g. the DaCe-reference competitor) is unaffected."""
    pytest.importorskip("dace.transformation.interstate.expand_nested_sdfg_inputs")
    from dace.sdfg import nodes
    sdfg = kernels()["hpc/map_reduce/mandelbrot1/mandelbrot1"].to_sdfg(simplify=True)

    def nsdfg_in_subsets(g):
        return {
            e.dst_conn: str(e.data.subset)
            for st in g.all_states()
            for n in st.nodes() if isinstance(n, nodes.NestedSDFG) for e in st.in_edges(n)
        }

    before = nsdfg_in_subsets(sdfg)
    sdfg_to_numpy(sdfg, "mandelbrot")
    assert nsdfg_in_subsets(sdfg) == before  # connectors/subsets unchanged -> no in-place widening


def test_nbody_nested_where_emits_and_computes():
    """nbody's ``np.power(inv_r3, -1.5, out=inv_r3, where=I)`` is a masked nested SDFG in a 2-D map;
    it emits correctly once ExpandNestedSDFGInputs offsets the multi-dim mask condition fully."""
    pytest.importorskip("dace.transformation.interstate.expand_nested_sdfg_inputs")
    from nestforge.ir.emit_numpy import UnsupportedNest
    from dace.frontend.python.common import DaceSyntaxError
    N, Nt = 6, 4
    rng = np.random.default_rng(0)
    mass, pos, vel = rng.random(N) + 0.5, rng.random((N, 3)), rng.random((N, 3))
    dt, G, soft = 0.01, 1.0, 0.1
    # The stock-DaCe gaps below are BUILD failures, so guard the build ALONE. IndexError is also a
    # routine symptom of an emitter bug, and the emitter only runs after this point -- catching it
    # around the emit/run step too would turn a nest-forge regression into an xfail blamed on DaCe.
    #
    # xfail, NOT skip: a skip is invisible to CI (which runs the unit set under NESTFORGE_CI_NO_SKIP)
    # and, worse, reads as "nothing to see here". These are known upstream gaps, which is what xfail
    # means. It is raised imperatively rather than via a decorator on purpose: a decorator would mark
    # the WHOLE test expected-to-fail, so an emitter regression further down would land in the same
    # green xfail bucket and hide -- exactly what the meta-test below exists to prevent. Raised here,
    # it fires only for the build gap, and the day DaCe can build nbody the test simply runs and
    # validates, which is the notification.
    try:
        sdfg = kernels()["hpc/n_body_methods/nbody/nbody"].to_sdfg(simplify=True)
    except (DaceSyntaxError, IndexError, FileExistsError) as e:
        # DaCe-frontend gaps (not nest-forge): the boolean-mask assignment lowers to index loops that trip an
        # IndexError, and ``np.empty(Nt + 1)`` registers ``Nt_plus_1`` as BOTH a scalar and a shape symbol so
        # add_symbol raises FileExistsError. The test runs once DaCe promotes the scalar instead of colliding.
        pytest.xfail(
            f"stock DaCe cannot lower nbody's masked assignment / Nt+1 scalar-symbol collision: {type(e).__name__}")
    inputs = dict(mass=mass, pos=pos, vel=vel, dt=np.array([dt]), G=np.array([G]), softening=np.array([soft]))
    try:
        call, _ = alloc_run("hpc/n_body_methods/nbody/nbody", "nbody", dict(N=N, Nt=Nt), inputs, sdfg=sdfg)
    except UnsupportedNest:
        # The emitter's own explicit refusal: it names the DaCe-side ExpandNestedSDFGInputs gap it hit.
        pytest.xfail("ExpandNestedSDFGInputs multi-dim condition offset not fixed in this DaCe")

    def getAcc(pos, mass, G, softening):
        x, y, z = pos[:, 0:1], pos[:, 1:2], pos[:, 2:3]
        dx = np.add.outer(-x, x).reshape(N, N)
        dy = np.add.outer(-y, y).reshape(N, N)
        dz = np.add.outer(-z, z).reshape(N, N)
        inv_r3 = dx**2 + dy**2 + dz**2 + softening**2
        np.power(inv_r3, -1.5, out=inv_r3, where=inv_r3 > 0)
        a = np.zeros((N, 3))
        a[:, 0], a[:, 1], a[:, 2] = G * (dx * inv_r3) @ mass, G * (dy * inv_r3) @ mass, G * (dz * inv_r3) @ mass
        return a

    def getEnergy(pos, vel, mass, G):
        KE = 0.5 * np.sum(np.reshape(mass, (N, 1)) * vel**2)
        x, y, z = pos[:, 0:1], pos[:, 1:2], pos[:, 2:3]
        dx = np.add.outer(-x, x).reshape(N, N)
        dy = np.add.outer(-y, y).reshape(N, N)
        dz = np.add.outer(-z, z).reshape(N, N)
        inv_r = np.sqrt(dx**2 + dy**2 + dz**2)
        np.divide(1.0, inv_r, out=inv_r, where=inv_r > 0)
        tmp = -np.multiply.outer(mass, mass) * inv_r
        PE = sum(tmp[j, k] for j in range(N) for k in range(j + 1, N)) * G
        return KE, PE

    vel = vel - np.mean(np.reshape(mass, (N, 1)) * vel, axis=0) / np.mean(mass)
    acc = getAcc(pos, mass, G, soft)
    KE = np.zeros(Nt + 1)
    PE = np.zeros(Nt + 1)
    KE[0], PE[0] = getEnergy(pos, vel, mass, G)
    for i in range(Nt):
        vel += acc * dt / 2.0
        pos += vel * dt
        acc = getAcc(pos, mass, G, soft)
        vel += acc * dt / 2.0
        KE[i + 1], PE[i + 1] = getEnergy(pos, vel, mass, G)
    # functional->in-place: nbody's ``return KE, PE`` lowers to the named ``KE``/``PE`` output buffers
    # (like atax's ``out``), not a DaCe ``__return`` -- validate those. Scalar reduction loops reassociate
    # vs numpy's pairwise ``np.sum``/``@``, so allow the default allclose tolerance.
    np.testing.assert_allclose(call["KE"], KE, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(call["PE"], PE, rtol=1e-5, atol=1e-8)


def test_nbody_xfail_covers_the_dace_build_only_not_an_emitter_indexerror(monkeypatch):
    """The nbody xfail must stay pinned to the stock-DaCe FRONTEND gap (an IndexError out of ``to_sdfg``).
    An IndexError raised once the SDFG is built comes from the emitter -- a nest-forge regression that has
    to fail the suite, since an xfail attributed to DaCe would hide it from CI entirely."""
    pytest.importorskip("dace.transformation.interstate.expand_nested_sdfg_inputs")  # match the nbody test

    class BuiltSdfg:
        """A build that SUCCEEDS -- so the DaCe-frontend gap is out of the picture and anything raised
        afterwards is the emitter's."""

        def to_sdfg(self, simplify=True):
            return self

    def emitter_indexerror(*args, **kwargs):
        raise IndexError("list index out of range")  # the shape stock DaCe's frontend gap also takes

    monkeypatch.setitem(globals(), "kernels", lambda: {"hpc/n_body_methods/nbody/nbody": BuiltSdfg()})
    monkeypatch.setitem(globals(), "alloc_run", emitter_indexerror)
    try:
        test_nbody_nested_where_emits_and_computes()
    except IndexError:
        return  # propagated to the caller: the regression is visible
    except BaseException as exc:  # pytest's Skipped/XFailed outcomes derive from BaseException, not Exception
        pytest.fail(f"an emitter IndexError was swallowed instead of raised: {type(exc).__name__}: {exc}")
    pytest.fail("an emitter IndexError was swallowed instead of raised: nbody test returned")


def test_azimint_hist_three_level_nested_return_and_computes():
    """azimint_hist nests get_bin_edges / compute_bin / histogram three deep; the innermost returns a
    size-1 array read as ``compute_bin_ret_0[0]`` in an inter-state assignment but written as a scalar
    local -- the emitter reconciles the two by stripping the scalar-local ``[0]``. Returns histw/histu."""
    pytest.importorskip("dace.transformation.interstate.expand_nested_sdfg_inputs")
    from nestforge.ir.emit_numpy import UnsupportedNest
    N, npt = 200, 8
    rng = np.random.default_rng(0)
    data, radius = rng.random(N), rng.random(N)
    try:
        call, _ = alloc_run("hpc/map_reduce/azimint_hist/azimint_hist", "azimint_hist", dict(N=N, npt=npt, bins=npt),
                            dict(data=data, radius=radius))
    except UnsupportedNest:
        pytest.skip("nested-SDFG emission unavailable in this DaCe")

    def hist(a, weights=None):
        edges = np.array([a.min() + i * (a.max() - a.min()) / npt for i in range(npt)] + [a.max()])
        out = np.zeros(npt, np.float64 if weights is not None else np.int64)
        for i in range(N):
            b = min(int(npt * (a[i] - edges[0]) / (edges[npt] - edges[0])), npt - 1)
            out[b] += weights[i] if weights is not None else 1
        return out

    ref = hist(radius, data) / hist(radius)
    got = call["out"]  # functional->in-place: the histw/histu ratio lands in the named ``out`` buffer
    np.testing.assert_allclose(got, ref, equal_nan=True)


def test_azimint_naive_wcr_reduction_emits_and_computes():
    """azimint_naive is a masked mean per radial bin: ``if r1<=radius<r2: tmp += data[j]`` -- a WCR
    accumulation inside a nested SDFG. Exercises WCR augmented-assignment plus the inner/outer size-1
    descriptor reconciliation (a nested scalar accumulator read back as a size-1 array)."""
    pytest.importorskip("dace.transformation.interstate.expand_nested_sdfg_inputs")
    from nestforge.ir.emit_numpy import UnsupportedNest
    N, npt = 150, 8
    rng = np.random.default_rng(0)
    data, radius = rng.random(N), rng.random(N)
    try:
        call, _ = alloc_run("hpc/map_reduce/azimint_naive/azimint_naive", "azimint_naive", dict(N=N, npt=npt),
                            dict(data=data, radius=radius))
    except UnsupportedNest:
        pytest.skip("nested-SDFG emission unavailable in this DaCe")

    rmax = radius.max()
    res = np.zeros(npt)
    for i in range(npt):
        r1, r2 = rmax * i / npt, rmax * (i + 1) / npt
        mask = np.logical_and(r1 <= radius, radius < r2)
        tmp, on = 0.0, 0
        for j in range(N):
            if mask[j]:
                tmp += data[j]
                on += 1
        res[i] = tmp / on
    got = call["res"]  # functional->in-place: the per-bin masked mean lands in the named ``res`` buffer
    np.testing.assert_allclose(got, res, equal_nan=True)


def test_trisolv_loop_shaped_scratch_maxsized_and_computes():
    """trisolv (lower-triangular solve) stages a growing slice into a scratch buffer shaped ``[i]``.
    The emitter widens that buffer to its loop bound ``N`` and addresses it with ``0:i`` slices, so it
    can be pre-allocated C-style. Validates the max-size loop-scratch path."""
    N = 12
    rng = np.random.default_rng(0)
    L = np.tril(rng.random((N, N))) + N * np.eye(N)
    b = rng.random(N)
    call, src = alloc_run("hpc/dense_linear_algebra/trisolv/trisolv", "trisolv", dict(N=N), dict(L=L, b=b))
    assert "np.empty" not in src  # still C-style: no in-kernel allocation
    x = call.get("x", call.get("__return"))
    np.testing.assert_allclose(x, np.linalg.solve(L, b))


def test_lu_loop_shaped_scratch_maxsized_and_computes():
    """lu decomposition stages ``A[i,:j]`` / ``A[:i,j]`` dot-product operands into ``[j]``/``[i]``
    scratch buffers; the emitter max-sizes them to ``N``. Validates against an in-place Doolittle LU."""
    N = 10
    rng = np.random.default_rng(1)
    A0 = rng.random((N, N)) + N * np.eye(N)
    call, src = alloc_run("hpc/dense_linear_algebra/lu/lu", "lu", dict(N=N), dict(A=A0.copy()))
    A = call.get("A", call.get("__return"))

    ref = A0.copy()
    for i in range(N):
        for j in range(i):
            ref[i, j] = (ref[i, j] - ref[i, :j] @ ref[:j, j]) / ref[j, j]
        for j in range(i, N):
            ref[i, j] = ref[i, j] - ref[i, :i] @ ref[:i, j]
    np.testing.assert_allclose(A, ref)


def test_covariance_decreasing_loop_scratch_and_computes():
    """covariance stages a *shrinking* slice into a ``[M-i]`` scratch; the emitter max-sizes that
    decreasing extent at its i=0 value (``M``) -- the compound (non-bare-var) monotonicity path."""
    M, Nrows = 8, 20
    rng = np.random.default_rng(0)
    data = rng.random((Nrows, M))
    fn = 8.0
    call, _ = alloc_run("hpc/dense_linear_algebra/covariance/covariance", "covariance", dict(M=M, N=Nrows),
                        dict(data=data.copy(), float_n=np.array([fn])))
    cov = call.get("cov", call.get("__return"))
    d2 = data - data.mean(axis=0)
    ref = np.zeros((M, M))
    for i in range(M):
        ref[i:M, i] = ref[i, i:M] = d2[:, i] @ d2[:, i:M] / (fn - 1.0)
    np.testing.assert_allclose(cov, ref)


def test_syrk_increasing_loop_scratch_and_computes():
    """syrk stages a *growing* ``[i+1]`` slice; the emitter max-sizes that increasing extent at its
    loop bound. C = alpha*A@A.T + beta*C on the lower triangle."""
    N, Mk = 9, 6
    rng = np.random.default_rng(1)
    A, C = rng.random((N, Mk)), rng.random((N, N))
    alpha, beta = 1.5, 1.2
    call, _ = alloc_run("hpc/dense_linear_algebra/syrk/syrk", "syrk", dict(N=N, M=Mk),
                        dict(A=A.copy(), C=C.copy(), alpha=np.array([alpha]), beta=np.array([beta])))
    Cout = call.get("C", call.get("__return"))
    ref = C.copy()
    for i in range(N):
        ref[i, :i + 1] *= beta
        for k in range(Mk):
            ref[i, :i + 1] += alpha * A[i, k] * A[:i + 1, k]
    np.testing.assert_allclose(np.tril(Cout), np.tril(ref))


def test_jacobi_1d_loopregion_emits_and_computes():
    rng = np.random.default_rng(2)
    N, T = 32, 20
    A, B = rng.random(N), rng.random(N)
    Ar, Br = A.copy(), B.copy()
    call, src = alloc_run("hpc/structured_grids/jacobi_1d/jacobi_1d", "jacobi_1d", dict(N=N, TSTEPS=T),
                          dict(A=A.copy(), B=B.copy()))
    assert "while" in src  # the TSTEPS time loop (a LoopRegion)
    for _ in range(1, T):
        Br[1:-1] = 0.33333 * (Ar[:-2] + Ar[1:-1] + Ar[2:])
        Ar[1:-1] = 0.33333 * (Br[:-2] + Br[1:-1] + Br[2:])
    np.testing.assert_array_equal(call["A"], Ar)
    np.testing.assert_array_equal(call["B"], Br)
