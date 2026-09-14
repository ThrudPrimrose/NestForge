# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression: a fusion must be REFUSED when loop1 writes a LOOP-INVARIANT location that loop2 reads.

Unfused, loop2 sees the FINAL value loop1 left in that location; fused, it sees the RUNNING value of the
current iteration. That is a genuine dependence, and it is exactly the one a carried-offset dependence
classifier reports no offset for -- there is no iterator in either subset to carry.
"""

import numpy as np

import dace

from nestforge.phases.schedule import apply_fusion, enumerate_fusions

N = dace.symbol("N")
f64 = dace.float64


@dace.program
def invariant_scalar(a: f64[N], d: f64[N]):
    s = np.float64(0.0)
    for i in range(1, N):
        s = a[i]  # writes a loop-invariant location; last value wins
    for i in range(1, N):
        d[i] = d[i - 1] + s  # a recurrence reading it -> neither loop is DOALL


def run(sdfg, a, d):
    ab, db = a.copy(), d.copy()
    sdfg(a=ab, d=db, N=len(a))
    return db


def test_fusion_across_an_invariant_scalar_is_value_preserving():
    a, d = np.arange(1.0, 7.0), np.zeros(6)
    ref = run(invariant_scalar.to_sdfg(simplify=True), a, d)

    sdfg = invariant_scalar.to_sdfg(simplify=True)
    for move in enumerate_fusions(sdfg):
        apply_fusion(sdfg, move)
    sdfg.validate()
    got = run(sdfg, a, d)
    # unfused: d[i] = d[i-1] + a[N-1]  (a prefix sum of the LAST element)
    # fused:   d[i] = d[i-1] + a[i]    (a prefix sum of a) -- a silent miscompile
    assert np.array_equal(ref, got), f"fusion changed the value: reference {ref} vs fused {got}"
