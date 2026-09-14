# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Every nest-forge SDFG is normalized before anything measures it.

Python's `//` on a sympy expression is `sympy.floor(...)`, which sympy distributes and codegen then
prints WITHOUT the floor -- the index truncates term by term. Kernel source is safe (dace parses `//`
into int_floor); transformation code is not, so the pass is forced at both entry points.
"""
import copy

import dace
import pytest
import sympy
from dace.subsets import Indices, Range

from nestforge.corpus import tsvc
from nestforge.granularity import granularity_ladder


def floors_in(sdfg):
    """Every residual ``sympy.floor`` codegen could reach: array shapes and memlet subsets."""
    found = []
    for sub in sdfg.all_sdfgs_recursive():
        for name, desc in sub.arrays.items():
            found += [(name, dim) for dim in desc.shape if sympy.sympify(dim).atoms(sympy.floor)]
        for state in sub.states():
            for edge in state.edges():
                if edge.data is None:
                    continue
                subset = edge.data.subset
                if isinstance(subset, Range):
                    bounds = [b for dim in subset.ranges for b in dim]
                elif isinstance(subset, Indices):
                    bounds = list(subset.indices)
                else:
                    continue
                found += [(edge.data.data, b) for b in bounds if sympy.sympify(b).atoms(sympy.floor)]
    return found


def test_the_detector_can_actually_fail():
    """Guard against a vacuous suite: floors_in must SEE a floor when one is present."""
    sdfg = dace.SDFG("injected")
    sdfg.add_array("a", [dace.symbolic.symbol("N") // 2], dace.float64)
    assert floors_in(sdfg), "floors_in reports nothing on an SDFG that provably holds a floor"


@pytest.mark.parametrize("opt_mode", ["simplify-parallel", "canonicalize"])
def test_build_sdfg_leaves_no_residual_floor(opt_mode):
    kernel = tsvc.iter_tsvc_kernels(only=["s111"])[0]
    assert not floors_in(tsvc.build_sdfg(kernel, opt_mode))


def test_every_granularity_rung_is_normalized():
    """The rungs are where fission/fusion rebuild indices, so this is the one that actually bites. s221 is a
    multi-statement kernel with a deep (depth-5) atoms->maximal ladder; s111 became a single statement-atom
    (one nest, so a one-rung ladder) once fission reached statement -- not per-tasklet -- granularity."""
    kernel = tsvc.iter_tsvc_kernels(only=["s221"])[0]
    canonical = tsvc.build_sdfg(kernel, "canonicalize")
    ladder = granularity_ladder(canonical, 4)
    assert len(ladder) >= 2, "test is vacuous on a single-rung ladder"
    for point in ladder:
        rung = copy.deepcopy(canonical)
        point.apply(rung)
        assert not floors_in(rung), f"rung {point.name} carries a residual floor"
