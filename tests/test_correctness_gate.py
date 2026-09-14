# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The gate that decides whether a measured kernel counts as CORRECT.

Every arena cell, every differential swap and every E-driver row is admitted or rejected by
``arena.diff_stats`` and the tolerance ``arena.rung_atol`` hands it. A gate that is too tight silently
deletes a whole kernel class from the corpus; one that is too loose admits a miscompile and reports
it as a speed-up. Both failures are invisible in the tables they corrupt -- there is no red test, just
a number that is wrong -- so the properties are pinned here rather than inferred from the sweeps.

Each test names the failure it prevents. Two of them are regressions of bugs that shipped: a NaN that
was reported as a PERFECT match, and a verdict read off zero elements.
"""

import numpy as np
import pytest

from nestforge.build import arena
from nestforge.build import flags


def test_a_nan_anywhere_fails_even_when_it_is_not_the_first_element():
    """Builtin ``max`` keeps its left operand when the right is NaN (``nan > x`` is False), so a NaN
    that is not the FIRST thing compared used to leave ``worst`` at 0.0 -- a NaN-poisoned kernel
    scored a perfect match and could win the arena. The position of the NaN must not matter."""
    good = np.ones(8)
    for position in (0, 3, 7):
        poisoned = good.copy()
        poisoned[position] = np.nan
        assert arena.diff_stats({"a": good}, {"a": poisoned}) == (float("inf"), float("inf"))
        assert arena.maxdiff({"a": good}, {"a": poisoned}) == float("inf")
        assert arena.relative_maxdiff({"a": good}, {"a": poisoned}) == float("inf")


def test_a_nan_in_a_later_ARRAY_fails_too():
    """Same hazard one level up: the first array can match perfectly and the second be poison."""
    clean = {"first": np.ones(4), "second": np.ones(4)}
    poisoned = {"first": np.ones(4), "second": np.array([1.0, 1.0, np.nan, 1.0])}
    assert arena.diff_stats(clean, poisoned) == (float("inf"), float("inf"))


def test_an_infinity_fails_rather_than_dominating_the_maximum():
    """``inf - 1`` is ``inf`` and ``inf/inf`` is NaN; neither is a tolerance a gate can compare."""
    assert arena.diff_stats({"a": np.ones(4)}, {"a": np.array([1.0, np.inf, 1.0, 1.0])}) == (float("inf"), float("inf"))


def test_a_comparison_that_touched_no_element_fails_instead_of_passing():
    """The gate is ``<= atol``, so a verdict of 0.0 read off zero elements PASSES -- a kernel whose
    outputs are all zero-size would be admitted as bit-exact having computed nothing."""
    empty = {"a": np.zeros(0), "b": np.zeros((0, 4))}
    assert arena.diff_stats(empty, empty) == (float("inf"), float("inf"))
    assert arena.maxdiff(empty, empty) == float("inf")
    assert arena.relative_maxdiff(empty, empty) == float("inf")


def test_one_empty_array_is_skipped_but_a_real_one_beside_it_still_decides():
    """Skipping an individual empty output is right (a kernel may legitimately declare one); letting
    it suppress the array that DOES have elements is not."""
    a = {"empty": np.zeros(0), "real": np.ones(4)}
    b = {"empty": np.zeros(0), "real": np.array([1.0, 1.0, 1.25, 1.0])}
    worst_abs, _worst_rel = arena.diff_stats(a, b)
    assert worst_abs == pytest.approx(0.25)


def test_small_magnitudes_keep_the_strict_absolute_reading():
    """The relative denominator floors at 1.0, so the gate is never LOOSENED where fp64 can actually
    deliver: at magnitude ~0.5 the scaled difference must equal the absolute one."""
    a = {"a": np.array([0.5, 0.25, 0.125])}
    b = {"a": np.array([0.5 + 1e-13, 0.25, 0.125])}
    worst_abs, worst_rel = arena.diff_stats(a, b)
    assert worst_rel == pytest.approx(worst_abs)
    assert worst_rel > 1e-14  # and it is still refused by the default rung


def test_a_reduction_sized_result_is_judged_relatively_not_absolutely():
    """Summing 32000 order-1 values lands near 1.6e4, where ONE fp64 ULP is ~1.8e-12 -- 180x the old
    1e-14 absolute gate. A correctly vectorized reduce reassociates to a few ULP and was recorded
    WRONG, so every reduction kernel vanished from the corpus silently. Scaled, it passes; the
    absolute number is still REPORTED, because it is a measurement, not a verdict."""
    total = 1.6e4
    a = {"sum": np.array([total])}
    b = {"sum": np.array([total + 14 * np.spacing(total)])}
    worst_abs, worst_rel = arena.diff_stats(a, b)
    assert worst_abs > 1e-14, "the absolute difference is genuinely larger than the old gate"
    assert worst_rel < 1e-14, "scaled by its own magnitude the same result is correct"


def test_the_gate_is_elementwise_and_a_single_bad_element_survives_averaging():
    """NOT a norm: one wrong element among ten thousand right ones must show at full size, because a
    miscompile that touches one index is still a miscompile."""
    a = {"a": np.ones(10_000)}
    b = np.ones(10_000)
    b[4_242] = 2.0
    worst_abs, worst_rel = arena.diff_stats(a, {"a": b})
    assert worst_abs == pytest.approx(1.0)
    assert worst_rel == pytest.approx(0.5)  # scaled by max(|1|, |2|, 1.0) = 2


def test_maxdiff_and_relative_maxdiff_read_the_right_half_of_diff_stats():
    """``maxdiff`` and ``relative_maxdiff`` are thin wrappers over ``diff_stats``; pin that each reads its
    OWN half -- a swapped index would silently report the relative gate as the absolute one, or vice versa.
    Chosen at reduction scale, where the two halves genuinely differ (see
    test_a_reduction_sized_result_is_judged_relatively_not_absolutely), so a swap cannot pass by accident."""
    total = 1.6e4
    a = {"sum": np.array([total])}
    b = {"sum": np.array([total + 14 * np.spacing(total)])}
    worst_abs, worst_rel = arena.diff_stats(a, b)
    assert worst_abs != worst_rel
    assert arena.maxdiff(a, b) == worst_abs
    assert arena.relative_maxdiff(a, b) == worst_rel


def test_the_gate_is_never_tighter_than_the_output_dtype_can_express():
    """A bit-exact rung asks for 0.0, which fp32 cannot deliver on any real arithmetic; the dtype
    floor is what keeps a correct fp32 kernel from being recorded as a miscompile."""
    fp32 = {"a": np.ones(4, dtype=np.float32)}
    fp64 = {"a": np.ones(4, dtype=np.float64)}

    def gate(mode: str, outputs: dict[str, np.ndarray]) -> float:
        return arena.rung_atol(mode, arena.dtype_floor(outputs))

    assert gate("strict-ieee", fp32) >= flags.DTYPE_ATOL["float32"]
    assert gate("strict-ieee", fp32) > gate("strict-ieee", fp64)


def test_an_integer_output_contributes_no_tolerance_floor():
    """Integer and boolean results are exact or they are wrong; giving them a floor would admit an
    off-by-one index as a rounding difference."""
    assert arena.dtype_floor({"i": np.arange(4)}) == 0.0
    assert arena.dtype_floor({"b": np.ones(4, dtype=bool)}) == 0.0
    mixed = {"i": np.arange(4), "f": np.ones(4, dtype=np.float32)}
    assert arena.dtype_floor(mixed) == flags.DTYPE_ATOL["float32"]
