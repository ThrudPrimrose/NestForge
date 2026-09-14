# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Emit + run the tsvc2.5 kernels that exercise the harder control-flow / reduction / recurrence paths.

Each kernel is extracted (baseline opt-mode), emitted to numpy, executed, and checked against a hand
reference. These pin three emitter/extraction fixes:

  * ``ext_break_find_first`` -- a ``break`` (BreakBlock) early-exit is emitted as ``break``;
  * ``cond_reduce_sym``      -- a size-1 buffer is READ as ``x[0]`` (not the bare ``(1,)`` array, which a
    NumPy-2 scalar assignment rejects), and its canonicalized WCR *copy* accumulates;
  * ``iv_multiplicative``    -- the loop index is pre-declared on the parent so nest extraction does not
    KeyError on an unregistered symbol.

tsvc2.5 lives in DaCe's ``performance_regression_jobs/tsvc_2_5_corpus.py``; skip if it is not reachable.
"""
import inspect

import numpy as np
import pytest

pytest.importorskip("hpcagent_bench")

from nestforge import tsvc
from nestforge.extract import extract_nest_to_sdfg
from nestforge.emit_numpy import load_emitted, sdfg_to_numpy
from nestforge.strategies import get_strategy


def load(key: str):
    try:
        found = tsvc.iter_tsvc_kernels(only=[key], corpus="tsvc2_5")
    except Exception as exc:  # corpus script not shipped with this DaCe
        pytest.skip(f"tsvc2.5 corpus unavailable: {exc}")
    if not found:
        pytest.skip(f"{key} not in the tsvc2.5 corpus")
    return found[0]


def emit_and_call(key: str, sizes: dict, inputs: dict, opt_mode: str = "simplify-parallel"):
    """Extract + emit ``key``, allocate every buffer C-style from the emitted signature, run it, return
    the call dict (buffers hold the results in place)."""
    kernel = load(key)
    sdfg = tsvc.build_sdfg(kernel, opt_mode=opt_mode)
    refs = get_strategy("skip-taskloops")(sdfg)
    assert len(refs) == 1, f"{key}/{opt_mode}: expected one compute nest, got {len(refs)}"
    boundary = extract_nest_to_sdfg(refs[0][0], refs[0][1], name=kernel.key)
    src = sdfg_to_numpy(boundary.standalone_sdfg, kernel.key)
    fn = vars(load_emitted(src, kernel.key))[kernel.key]
    call = {}
    for name in inspect.signature(fn).parameters:
        if name in sizes:
            call[name] = sizes[name]
            continue
        desc = boundary.standalone_sdfg.arrays[name]
        shape = tuple(int(str(d)) if str(d).isdigit() else sizes["LEN_1D"] for d in desc.shape)
        dt = np.dtype(desc.dtype.type)
        call[name] = inputs[name].astype(dt) if name in inputs else np.zeros(shape, dt)
    fn(**call)
    return call, src


def test_ext_break_find_first_emits_break_and_stops():
    """BreakBlock -> ``break``: a[i] += b[i]*c[i] until the first d[i] < 0."""
    n = 40
    rng = np.random.default_rng(3)
    a, b, c = rng.random(n), rng.random(n), rng.random(n)
    d = rng.random(n)
    d[17] = -1.0  # force the break at a known, non-trivial index
    call, src = emit_and_call("ext_break_find_first", dict(LEN_1D=n), dict(a=a.copy(), b=b.copy(), c=c.copy(), d=d))
    assert "break" in src
    ref = a.copy()
    for i in range(n):
        if d[i] < 0.0:
            break
        ref[i] = ref[i] + b[i] * c[i]
    np.testing.assert_allclose(call["a"], ref, rtol=1e-12, atol=1e-12)
    # The break lands at the right index, checked through the DATA: everything up to 17 is
    # accumulated and everything from 17 on is untouched. A loop counter whose final value no
    # block of the SDFG reads is scoped to its loop and not exported, so there is no
    # ``__sym_out_i`` scalar to read it out of -- and the numerics pin the stop point anyway.
    assert not np.allclose(call["a"][:17], a[:17])  # accumulated before the break
    np.testing.assert_allclose(call["a"][17:], a[17:], rtol=1e-12, atol=1e-12)  # untouched after


@pytest.mark.parametrize("opt_mode", ["simplify-parallel", "canonicalize", "auto-opt"])
def test_cond_reduce_sym_scalar_read_and_wcr(opt_mode):
    """Size-1 buffer read as ``x[0]`` (simplify-parallel) and WCR copy accumulates (canonicalize/auto-opt):
    out = sum of a[i] where a[i] > K. Every opt-mode must reduce correctly through emit+compile+run."""
    n, K = 96, 0.5
    a = np.random.default_rng(0).random(n)
    call, _ = emit_and_call("cond_reduce_sym", dict(LEN_1D=n, K=K), dict(a=a.copy()), opt_mode=opt_mode)
    acc = next(call[k] for k in call if k in ("out", "_priv_out") or k.endswith("_out"))
    np.testing.assert_allclose(float(np.ravel(acc)[0]), a[a > K].sum(), rtol=1e-12, atol=1e-12)


def test_iv_multiplicative_extracts_and_computes_recurrence():
    """Loop index pre-declared so extraction does not KeyError: s <- s * 0.99 for LEN_1D iterations."""
    n = 50
    s0 = 3.5
    call, _ = emit_and_call("iv_multiplicative", dict(LEN_1D=n), dict(s=np.array([s0])))
    np.testing.assert_allclose(float(call["s"][0]), s0 * (0.99**n), rtol=1e-12, atol=1e-12)


def test_float_value_scalar_is_double_not_truncated_in_compiled_c(tmp_path):
    """A staged ``a_index = a[i]`` read leaked into the boundary is a FLOAT value scalar, not an int size
    symbol. The manifest must declare it under ``init.scalars`` (from its SDFG dtype) so the translator
    emits ``double a_index`` -- and the ctypes arg-type must match (``c_double``). The exec-the-numpy tests
    above cannot catch this (Python is untyped); only the COMPILED C truncates ``int64_t a_index = a[i]``
    to 0 for data in ``[0, 1)``, blowing the conditional reduction up. This pins the compiled result to 0
    maxdiff vs the numpy oracle end to end."""
    import ctypes
    import shutil
    import subprocess

    from nestforge.arena import make_inputs, run_oracle, maxdiff, scalar_ctype
    from nestforge.isolation import run_isolated
    from nestforge.perf.harness import c_argtypes, call_c, signature_order
    from nestforge.translate import prepare, emit_sources

    cc = shutil.which("gcc") or shutil.which("clang")
    if cc is None:
        pytest.skip("no C compiler")

    kernel = load("cond_reduce_sym")
    sdfg = tsvc.build_sdfg(kernel, opt_mode="simplify-parallel")
    refs = get_strategy("skip-taskloops")(sdfg)
    b = extract_nest_to_sdfg(refs[0][0], refs[0][1], name="cond_reduce_sym")
    ssdfg = b.standalone_sdfg

    # the staged float read is a double symbol in the SDFG.
    float_syms = [s for s in b.symbols if s in ssdfg.symbols and np.dtype(ssdfg.symbols[s].type).kind == "f"]
    assert float_syms, f"expected a float value scalar in the boundary; symbols={list(b.symbols)}"
    # ctypes arg-typing: float scalar -> c_double, integer size symbol -> c_int64 (the translator's ABI).
    assert scalar_ctype(ssdfg, float_syms[0]) is ctypes.c_double
    assert scalar_ctype(ssdfg, "LEN_1D") is ctypes.c_int64

    sizes = tsvc.sample_sizes(kernel, b, preset="S")
    prep = prepare(b, "cond_reduce_sym", tmp_path / "kern", sizes=sizes)
    # manifest: the float scalar declared under init.scalars, NOT the integer parameters block.
    for s in float_syms:
        assert s in prep.manifest["init"].get("scalars", {}), prep.manifest["init"]
        assert s not in prep.manifest["parameters"]["S"], prep.manifest["parameters"]

    csrc = next(p for p in emit_sources(prep, tmp_path / "gen", target="c") if p.suffix == ".c")
    sig = [ln for ln in csrc.read_text().splitlines() if "_fp64(" in ln][0]
    assert f"double {float_syms[0]}" in sig, sig  # not `int64_t a_index`

    so = tmp_path / "lib.so"
    r = subprocess.run([cc, "-O2", "-fPIC", "-shared", "-ffp-contract=off",
                        str(csrc), "-o", str(so)],
                       capture_output=True,
                       text=True)
    assert r.returncode == 0, r.stderr

    # bind by the actual C signature order + dtype-aware argtypes, run once, compare to the numpy oracle.
    # Forked (a mis-typed size symbol would blow the loop bound out of range and segfault, not just diverge).
    symbol = "cond_reduce_sym_fp64"
    order = signature_order(csrc.read_text(), symbol, "c")
    inputs = make_inputs(b, sizes, seed=0)
    oracle = run_oracle(prep, b, inputs, sizes)

    def work():
        outs, _ = call_c(so, symbol, order, c_argtypes(order, b), b, inputs, sizes, 1)
        return {"md": float(maxdiff(oracle, outs))}

    res = run_isolated(work, timeout=120)
    assert "error" not in res, f"compiled kernel crashed: {res.get('error')}"
    assert res["md"] < 1e-9, f"compiled C diverged from oracle: {res['md']}"
