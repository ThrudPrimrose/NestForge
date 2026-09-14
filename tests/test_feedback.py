# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Phase-4 feedback loop (:mod:`nestforge.phases.feedback`): the measurement-driven granularity loop and its two
rules -- ``best_outcome`` (fastest bit-exact wins) and ``improved`` (a round that does not improve stops
the loop). Driven with a fake ``measure`` (no compiler), plus one real SDFG proving ``default_fuse_step``
re-enumerates + fuses to the fixed point.
"""
import numpy as np
import pytest

import dace
from nestforge.phases.feedback import (FeedbackResult, Outcome, best_outcome, default_fuse_step, improved,
                                       run_feedback_loop)
from nestforge.phases.scopes import outer

N = dace.symbol("N")
f64 = dace.float64


@dace.program
def two_nests(a: f64[N], c: f64[N]):
    tmp = np.empty_like(a)
    for i in dace.map[0:N]:
        tmp[i] = a[i] * 2.0
    for i in dace.map[0:N]:
        c[i] = tmp[i] + 1.0


def oc(median_us: float, ok: bool = True) -> Outcome:
    return Outcome(name="fb", ok=ok, median_us=median_us)


def nest_count(sdfg: dace.SDFG) -> int:
    return len(outer(sdfg))


def test_best_outcome_ignores_failed_and_picks_fastest():
    outs = [oc(10.0), oc(3.0, ok=False), oc(5.0)]  # the 3.0 lost the correctness gate
    assert best_outcome(outs).median_us == 5.0


def test_best_outcome_none_when_all_fail():
    assert best_outcome([oc(1.0, ok=False), oc(2.0, ok=False)]) is None


def test_improved_only_when_bit_exact_and_faster():
    prior = [oc(5.0)]
    assert improved(prior, oc(4.0))  # faster + bit-exact
    assert not improved(prior, oc(6.0))  # slower
    assert not improved(prior, oc(1.0, ok=False))  # faster but not bit-exact
    assert improved([], oc(9.0))  # first bit-exact result always improves


def test_default_fuse_step_fuses_to_fixed_point():
    sdfg = two_nests.to_sdfg(simplify=True)
    start = nest_count(sdfg)
    assert start >= 2  # two independent map-nests before fusion
    assert default_fuse_step(sdfg) is True  # a fusion move existed
    assert nest_count(sdfg) < start  # granularity coarsened
    while default_fuse_step(sdfg):  # drain to the fixed point
        pass
    assert default_fuse_step(sdfg) is False  # no move left


def test_loop_fuses_while_it_helps_then_hits_fixed_point():
    # fewer nests measured as faster -> fusing always improves -> loop runs to the fusion fixed point.
    sdfg = two_nests.to_sdfg(simplify=True)
    res = run_feedback_loop(sdfg, lambda s: oc(float(nest_count(s))))
    assert isinstance(res, FeedbackResult)
    assert res.best.median_us == min(o.median_us for o in res.outcomes)
    assert nest_count(res.sdfg) == 1  # stopped-at granularity is the (best) fully-fused one
    assert default_fuse_step(res.sdfg) is False  # and it is the fixed point


def test_loop_stops_the_round_a_move_stops_helping():
    times = iter([10.0, 5.0, 5.0, 1.0])  # round 2 (5.0) does not beat round 1 (5.0) -> stop before 1.0
    res = run_feedback_loop(two_nests.to_sdfg(simplify=True), lambda s: oc(next(times)), apply_move=lambda s: True)
    assert [o.median_us for o in res.outcomes] == [10.0, 5.0, 5.0]
    assert res.best.median_us == 5.0
    assert res.rounds == 2


def test_loop_is_bounded_by_max_rounds():
    # a move that always applies + always improves must still terminate at max_rounds.
    t = iter(float(x) for x in range(100, 0, -1))
    res = run_feedback_loop(two_nests.to_sdfg(simplify=True),
                            lambda s: oc(next(t)),
                            apply_move=lambda s: True,
                            max_rounds=3)
    assert res.rounds == 3
    assert len(res.outcomes) == 4  # baseline + 3 rounds


def test_bad_max_rounds_raises():
    with pytest.raises(ValueError):
        run_feedback_loop(two_nests.to_sdfg(simplify=True), lambda s: oc(1.0), max_rounds=0)
