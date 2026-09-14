# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Emit + run loop_level_reasoning kernels that exercise harder control-flow / reduction paths.

Each kernel is canonicalized, extracted, emitted to numpy, executed, and checked against a hand
reference. These pin two emitter/extraction fixes:

  * ``ext_break_find_first`` -- a ``break`` (BreakBlock) early-exit is emitted as ``break``;
  * ``cond_reduce_sym``      -- a size-1 buffer is READ as ``x[0]`` (not the bare ``(1,)`` array, which a
    NumPy-2 scalar assignment rejects), and its canonicalized WCR *copy* accumulates.
"""
import inspect

import numpy as np
import pytest

pytest.importorskip("hpcagent_bench")

from dace.transformation.passes.canonicalize import canonicalize

from nestforge.corpus.bench import iter_dace_kernels
from nestforge.ir.extract import extract_nest_to_sdfg
from nestforge.ir.emit_numpy import load_emitted, sdfg_to_numpy
from nestforge.phases.scopes import get_strategy


def load(key: str):
    for kernel in iter_dace_kernels("loop_level_reasoning"):
        if kernel.short_name.rsplit("/", 1)[-1] == key:
            return kernel
    raise AssertionError(f"{key} is not in the loop_level_reasoning track -- the corpus this test pins has changed")


def emit_and_call(key: str, sizes: dict, inputs: dict):
    """Canonicalize + extract + emit ``key``, allocate every buffer C-style from the emitted signature,
    run it, return the call dict (buffers hold the results in place)."""
    kernel = load(key)
    sdfg = kernel.to_sdfg(simplify=True)
    canonicalize(sdfg, target="cpu")
    refs = get_strategy("skip-taskloops")(sdfg)
    assert len(refs) == 1, f"{key}: expected one compute nest, got {len(refs)}"
    boundary = extract_nest_to_sdfg(refs[0][0], refs[0][1], name=key)
    src = sdfg_to_numpy(boundary.standalone_sdfg, key)
    fn = vars(load_emitted(src, key))[key]
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
    # accumulated and everything from 17 on is untouched.
    assert not np.allclose(call["a"][:17], a[:17])  # accumulated before the break
    np.testing.assert_allclose(call["a"][17:], a[17:], rtol=1e-12, atol=1e-12)  # untouched after


def test_cond_reduce_sym_scalar_read_and_wcr():
    """Size-1 buffer read as ``x[0]`` and WCR copy accumulates: out = sum of a[i] where a[i] > K."""
    n = 96
    kernel = load("cond_reduce_sym")
    k_value = kernel.spec.pinned_config["K"]
    a = np.random.default_rng(0).random(n)
    call, _ = emit_and_call("cond_reduce_sym", dict(LEN_1D=n), dict(a=a.copy()))
    acc = next(call[k] for k in call if k in ("out", "_priv_out") or k.endswith("_out"))
    np.testing.assert_allclose(float(np.ravel(acc)[0]), a[a > k_value].sum(), rtol=1e-12, atol=1e-12)
