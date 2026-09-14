# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Every nest-forge SDFG is normalized before anything measures it.

Python's `//` on a sympy expression is `sympy.floor(...)`, which sympy distributes and codegen then
prints WITHOUT the floor -- the index truncates term by term. Kernel source is safe (dace parses `//`
into int_floor); transformation code is not, so canonicalization normalizes it.
"""
import dace
import sympy
from dace.subsets import Indices, Range
from dace.transformation.passes.canonicalize import canonicalize

from nestforge.corpus.bench import iter_dace_kernels


def load(key):
    for kernel in iter_dace_kernels("loop_level_reasoning"):
        if kernel.short_name.rsplit("/", 1)[-1] == key:
            return kernel
    raise AssertionError(f"{key} is not in the loop_level_reasoning track -- the corpus this test pins has changed")


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
    # Built explicitly: dace symbols now floor-divide to int_floor, which is not the residue under test.
    sdfg.add_array("a", [sympy.floor(dace.symbolic.symbol("N") / 2)], dace.float64)
    assert floors_in(sdfg), "floors_in reports nothing on an SDFG that provably holds a floor"


def test_canonicalize_leaves_no_residual_floor():
    """tsvc_2_s111's stride-2 loop needs a floor-division trip count; canonicalization must normalize it."""
    sdfg = load("tsvc_2_s111").to_sdfg(simplify=True)
    canonicalize(sdfg, target="cpu")
    assert not floors_in(sdfg)
