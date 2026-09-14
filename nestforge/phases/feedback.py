# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Feedback edges (e) and (g): measured outcomes that decide whether to re-enter the inter-kernel schedule."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable, List, Optional

import dace

from nestforge.phases.schedule import apply_fusion, first_fusion


@dataclass(frozen=True, slots=True)
class Outcome:
    """One measured round: which configuration, whether it matched the oracle, how fast."""
    name: str
    ok: bool
    median_us: float = float("inf")
    error: Optional[str] = None


#: Mutate the SDFG by one granularity move; ``False`` when no move is left.
GranularityStep = Callable[[dace.SDFG], bool]
#: Build, validate and time one granularity.
Measure = Callable[[dace.SDFG], Outcome]


def default_fuse_step(sdfg: dace.SDFG) -> bool:
    """Apply the first legal fusion move in place."""
    move = first_fusion(sdfg)
    if move is None:
        return False
    apply_fusion(sdfg, move)
    return True


def best_outcome(outcomes: List[Outcome]) -> Optional[Outcome]:
    """Fastest outcome that matched the oracle; a wrong result never wins on speed."""
    valid = [o for o in outcomes if o.ok]
    return min(valid, key=lambda o: o.median_us) if valid else None


def improved(prior: List[Outcome], candidate: Outcome) -> bool:
    """Whether ``candidate`` beats every prior correct outcome."""
    if not candidate.ok:
        return False
    best = best_outcome(prior)
    return best is None or candidate.median_us < best.median_us


@dataclass(slots=True)
class FeedbackResult:
    """Every round's outcome, the winner, and the SDFG snapshot the winner was measured on."""
    outcomes: List[Outcome]
    best: Optional[Outcome]
    rounds: int
    sdfg: dace.SDFG


def run_feedback_loop(sdfg: dace.SDFG,
                      measure: Measure,
                      apply_move: GranularityStep = default_fuse_step,
                      max_rounds: int = 8) -> FeedbackResult:
    """Apply one move per round and re-measure until a round does not improve or no move is left."""
    if max_rounds < 1:
        raise ValueError(f"max_rounds must be >= 1, got {max_rounds}")
    outcomes = [measure(sdfg)]
    best_sdfg = copy.deepcopy(sdfg)
    rounds = 0
    for _ in range(max_rounds):
        if not apply_move(sdfg):
            break
        rounds += 1
        candidate = measure(sdfg)
        better = improved(outcomes, candidate)
        outcomes.append(candidate)
        if not better:
            break
        best_sdfg = copy.deepcopy(sdfg)
    return FeedbackResult(outcomes, best_outcome(outcomes), rounds, best_sdfg)
