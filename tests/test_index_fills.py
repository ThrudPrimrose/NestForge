# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The nest's non-transient arrays come from the kernel's OptArena manifest
(:func:`nestforge.corpus.bench.index_fills` + ``make_inputs(given=...)``), transients keep the random fill.

The case under test is the integer index array. The default uniform-float fill cast to an integer dtype
collapses to ALL-ZEROS, which silently degrades a gather ``a[i] = b[ip[i]]`` into a single cached read of
``b[0]`` and inverts a scatter ``a[ip[i]] = ...`` from hpcagent_bench's guaranteed conflict-FREE permutation
into a maximal write conflict on ``a[0]``. Both are invisible to validation -- the oracle reads the same
degenerate ``ip`` -- so only an explicit test pins the property.
"""
import numpy as np
import pytest

from dace.transformation.passes.canonicalize import canonicalize

from nestforge.build.arena import make_inputs
from nestforge.corpus.bench import iter_dace_kernels, index_fills, preset_sizes
from nestforge.ir.extract import extract_nest_to_sdfg
from nestforge.phases.scopes import parallel_top_level_maps

#: (kernel, index array). Every one is a gather/scatter whose index array is the whole point of the test;
#: loop_level_reasoning is a superset of TSVC-2, so these are the same kernels the old TSVC corpus pinned.
GATHER_KERNELS = ["tsvc_2_vag", "tsvc_2_s4113", "tsvc_2_s353", "reroll_gather"]
INDEX_ARRAY = "ip"


def load(key):
    for kernel in iter_dace_kernels("loop_level_reasoning"):
        if kernel.short_name.rsplit("/", 1)[-1] == key:
            return kernel
    raise AssertionError(f"{key} is not in the loop_level_reasoning track -- the corpus this test pins has changed")


# These kernels are named constants of the corpus this repo pins, and every one is a gather/scatter whose
# index array is the whole point of the test. A missing kernel or a nest-less kernel is therefore a broken
# corpus or a broken splitter -- a failure to surface, never a skip to hide behind. (A skip here would also
# fail CI, which runs the unit set under NESTFORGE_CI_NO_SKIP=1.)
def first_nest(kernel):
    sdfg = kernel.to_sdfg(simplify=True)
    canonicalize(sdfg, target="cpu")
    refs = parallel_top_level_maps(sdfg)
    assert refs, f"{kernel.short_name}: the splitter found no compute nest -- this kernel has one"
    return extract_nest_to_sdfg(refs[0][0], refs[0][1], name=kernel.short_name.rsplit("/", 1)[-1])


@pytest.mark.parametrize("key", GATHER_KERNELS)
def test_index_array_is_a_valid_subscript_permutation(key):
    kernel = load(key)
    boundary = first_nest(kernel)
    sizes = preset_sizes(kernel, "S")
    inputs = make_inputs(boundary, sizes, seed=0, given=index_fills(kernel.short_name, boundary, sizes))

    ip = inputs[INDEX_ARRAY]
    n = ip.shape[0]
    assert ip.dtype.kind in "iu"
    # a permutation of [0, n): every subscript in range, each used exactly once -> a real gather, and a
    # conflict-free scatter (the property optarena/tests/test_foundation_scatter_conflict_free.py guards).
    assert np.array_equal(np.sort(ip), np.arange(n, dtype=ip.dtype))


@pytest.mark.parametrize("key", GATHER_KERNELS)
def test_index_array_is_all_zeros_without_the_manifest_fill(key):
    # pins the bug being fixed: the plain float fill cast to an int dtype degenerates to b[0] every
    # iteration. If this ever stops holding, `given` has become dead weight and should go.
    kernel = load(key)
    boundary = first_nest(kernel)
    sizes = preset_sizes(kernel, "S")
    ip = make_inputs(boundary, sizes, seed=0)[INDEX_ARRAY]
    assert ip.size > 1 and len(np.unique(ip)) == 1 and ip.ravel()[0] == 0


@pytest.mark.parametrize("key", GATHER_KERNELS)
def test_index_fills_are_seeded_so_oracle_and_candidate_agree(key):
    # the oracle is built once and every cell validates against it: an unseeded fill would give the
    # candidate a different `ip` than the oracle saw and break validation for every gather kernel.
    kernel = load(key)
    boundary = first_nest(kernel)
    sizes = preset_sizes(kernel, "S")
    a = index_fills(kernel.short_name, boundary, sizes)
    b = index_fills(kernel.short_name, boundary, sizes)
    assert np.array_equal(a[INDEX_ARRAY], b[INDEX_ARRAY])


@pytest.mark.parametrize("key", GATHER_KERNELS)
def test_index_fills_only_covers_manifest_declared_integer_arrays(key):
    # an integer array is not automatically a subscript; only what the manifest declares gets a fill.
    kernel = load(key)
    boundary = first_nest(kernel)
    sizes = preset_sizes(kernel, "S")
    fills = index_fills(kernel.short_name, boundary, sizes)
    assert set(fills) == {INDEX_ARRAY}  # not the float data arrays, and not the __sym_out_* output scalars


def test_index_fills_empty_without_a_manifest_name():
    # the None short-circuit: a boundary with no resolvable manifest name gets no fills.
    kernel = load("tsvc_2_vag")
    boundary = first_nest(kernel)
    sizes = preset_sizes(kernel, "S")
    assert index_fills(None, boundary, sizes) == {}


def test_given_array_of_the_wrong_shape_is_rejected():
    # `given` is passed straight across the ABI as the kernel's buffer, so a mismatch must raise here
    # rather than corrupt memory in the compiled call.
    kernel = load("tsvc_2_vag")
    boundary = first_nest(kernel)
    sizes = preset_sizes(kernel, "S")
    bad = {"ip": np.arange(3, dtype=np.int32)}
    with pytest.raises(ValueError, match="given array 'ip'"):
        make_inputs(boundary, sizes, seed=0, given=bad)


def test_given_array_of_the_wrong_dtype_is_rejected():
    kernel = load("tsvc_2_vag")
    boundary = first_nest(kernel)
    sizes = preset_sizes(kernel, "S")
    n = make_inputs(boundary, sizes, seed=0)["ip"].shape[0]
    with pytest.raises(ValueError, match="given array 'ip'"):
        make_inputs(boundary, sizes, seed=0, given={"ip": np.arange(n, dtype=np.float64)})


def test_transient_scratch_keeps_its_own_fill():
    # only the manifest's non-transient arrays are `given`; everything else is untouched by this change.
    kernel = load("tsvc_2_vag")
    boundary = first_nest(kernel)
    sizes = preset_sizes(kernel, "S")
    plain = make_inputs(boundary, sizes, seed=0)
    with_fills = make_inputs(boundary, sizes, seed=0, given=index_fills(kernel.short_name, boundary, sizes))
    for name, value in plain.items():
        if name != "ip":
            assert np.array_equal(value, with_fills[name]), f"{name} changed but is not a manifest index array"
