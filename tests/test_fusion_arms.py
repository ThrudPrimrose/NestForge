# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The Phase-2 fusion tool surface (:mod:`nestforge.phases.schedule`): enumerate legal fusion moves and apply
them, with the correctness net that any sequence of applied moves preserves the program's value bit-for-bit
against the un-fused reference. Exercises all three arms -- loop, vertical map, horizontal map -- and the
agent's real pattern of applying a random legal sequence.
"""
import numpy as np
import pytest

import dace
from dace.transformation.interstate.state_fusion import StateFusion

from nestforge.phases.schedule import (FusionMove, apply_fusion, enumerate_fusions, horizontal_map_moves,
                                       loop_fusion_moves, vertical_map_moves)

N = dace.symbol("N")
f64 = dace.float64


def run(sdfg, inputs, n):
    bufs = {k: v.copy() for k, v in inputs.items()}
    sdfg(**bufs, N=n)
    return bufs


def mk(n=48, names=("a", "b", "c"), seed=0):
    rng = np.random.default_rng(seed)
    return {k: rng.random(n) for k in names}


def apply_to_fixpoint(sdfg, order="greedy", seed=0):
    """Apply legal fusion moves until none remain; return the count. ``order='random'`` picks a random
    legal move each round (a seeded stand-in for the agent's choice)."""
    rng = np.random.default_rng(seed)
    applied = 0
    while True:
        moves = enumerate_fusions(sdfg)
        if not moves:
            return applied
        move = moves[0] if order == "greedy" else moves[int(rng.integers(len(moves)))]
        apply_fusion(sdfg, move)
        applied += 1


# --- programs exercising each arm (co-locate maps into one state so horizontal siblings can match) -------


@dace.program
def two_recurrences(a: f64[N], b: f64[N], c: f64[N]):
    for i in range(1, N):
        b[i] = b[i - 1] + a[i]
    for i in range(1, N):
        c[i] = c[i - 1] + b[i]


@dace.program
def producer_consumer_maps(a: f64[N], b: f64[N]):
    tmp = np.empty_like(a)
    for i in dace.map[0:N]:
        tmp[i] = a[i] * 2.0
    for i in dace.map[0:N]:
        b[i] = tmp[i] + 1.0


@dace.program
def sibling_maps(a: f64[N], b: f64[N], c: f64[N]):
    for i in dace.map[0:N]:
        b[i] = a[i] * 2.0
    for i in dace.map[0:N]:
        c[i] = a[i] + 1.0


def co_located(prog):
    """Build the SDFG and StateFusion sequential states together, so producer/consumer and sibling maps
    land in ONE state where the map-fusion arms can match them (mirrors the Phase-1 fusion-ready canon)."""
    sdfg = prog.to_sdfg(simplify=True)
    sdfg.apply_transformations_repeated(StateFusion)
    return sdfg


# --- enumeration finds the right arm ----------------------------------------------------------------


def test_enumerate_finds_loop_fusion():
    sdfg = two_recurrences.to_sdfg(simplify=True)
    moves = loop_fusion_moves(sdfg)
    assert any(m.kind == "fuse-loops" for m in moves)


def test_enumerate_finds_vertical_map_fusion():
    sdfg = co_located(producer_consumer_maps)
    moves = vertical_map_moves(sdfg)
    assert any(m.kind == "fuse-map-vertical" for m in moves)


def test_enumerate_finds_horizontal_map_fusion():
    sdfg = co_located(sibling_maps)
    moves = horizontal_map_moves(sdfg)
    assert any(m.kind == "fuse-map-horizontal" for m in moves)


def test_enumerated_moves_carry_apply_kwargs():
    sdfg = co_located(producer_consumer_maps)
    for m in enumerate_fusions(sdfg):
        assert isinstance(m, FusionMove) and m.where and m.xform is not None


# --- applying moves preserves value bit-for-bit -----------------------------------------------------


@pytest.mark.parametrize("prog,names,colocate", [
    (two_recurrences, ("a", "b", "c"), False),
    (producer_consumer_maps, ("a", "b"), True),
    (sibling_maps, ("a", "b", "c"), True),
])
def test_apply_all_fusions_is_value_preserving(prog, names, colocate):
    inputs = mk(names=names)
    ref = run(prog.to_sdfg(simplify=True), inputs, 48)
    sdfg = co_located(prog) if colocate else prog.to_sdfg(simplify=True)
    applied = apply_to_fixpoint(sdfg, order="greedy")
    got = run(sdfg, inputs, 48)
    assert applied >= 1
    assert all(np.allclose(got[k], ref[k]) for k in inputs)


@dace.program
def all_three_arms(a: f64[N], b: f64[N], c: f64[N], d: f64[N]):
    tmp = np.empty_like(a)
    for i in dace.map[0:N]:  # producer
        tmp[i] = a[i] * 2.0
    for i in dace.map[0:N]:  # vertical consumer of tmp
        b[i] = tmp[i] + 1.0
    for i in dace.map[0:N]:  # sibling of the above over a
        c[i] = a[i] - 1.0
    for i in range(1, N):  # sequential recurrence pair
        d[i] = d[i - 1] + a[i]


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_random_fusion_sequence_is_value_preserving(seed):
    inputs = mk(names=("a", "b", "c", "d"))
    ref = run(all_three_arms.to_sdfg(simplify=True), inputs, 48)
    sdfg = co_located(all_three_arms)
    apply_to_fixpoint(sdfg, order="random", seed=seed)
    got = run(sdfg, inputs, 48)
    assert all(np.allclose(got[k], ref[k]) for k in inputs), f"seed {seed}: diverged from reference"


def test_no_moves_on_a_single_map():

    @dace.program
    def one_map(a: f64[N], b: f64[N]):
        for i in dace.map[0:N]:
            b[i] = a[i] * 2.0

    assert enumerate_fusions(one_map.to_sdfg(simplify=True)) == []
