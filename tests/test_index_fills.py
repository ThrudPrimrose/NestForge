# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The nest's non-transient arrays come from the kernel's OptArena manifest
(:func:`nestforge.tsvc.index_fills` + ``make_inputs(given=...)``), transients keep the random fill.

The case under test is the integer index array. The default uniform float fill cast to an integer dtype
collapses to ALL-ZEROS, which silently degrades a gather ``a[i] = b[ip[i]]`` into a single cached read of
``b[0]`` and inverts a scatter ``a[ip[i]] = ...`` from hpcagent_bench's guaranteed conflict-FREE permutation into
a maximal write conflict on ``a[0]``. Both are invisible to validation -- the oracle reads the same
degenerate ``ip`` -- so only an explicit test pins the property.
"""
import numpy as np
import pytest

from nestforge.build.arena import make_inputs
from nestforge.multinest import extract_all_nests
from nestforge.corpus.tsvc import index_fills, iter_tsvc_kernels, sample_sizes

#: (corpus, key, index array). The two corpora name their manifests differently -- ``tsvc_2_vag.yaml`` vs a
#: bare ``reroll_gather.yaml`` -- so both are covered: a tsvc2_5 kernel only reaches its declared index
#: array if ``yaml_path`` resolves the un-prefixed name.
GATHER_KERNELS = [("tsvc2", "vag", "ip"), ("tsvc2", "s4113", "ip"), ("tsvc2", "s353", "ip"),
                  ("tsvc2_5", "reroll_gather", "ip")]


# These kernels are named constants of the corpora this repo pins, and every one is a gather/scatter whose
# index array is the whole point of the test. A missing kernel or a nest-less kernel is therefore a broken
# corpus or a broken splitter -- a failure to surface, never a skip to hide behind. (A skip here would also
# fail CI, which runs the unit set under NESTFORGE_CI_NO_SKIP=1.)
def first_nest(kernel):
    nests = extract_all_nests(lambda: kernel.program.to_sdfg(simplify=True), "outer", kernel.key)
    assert nests, f"{kernel.key}: the splitter found no compute nest -- this kernel has one"
    return nests[0][3]


def load(corpus, key):
    kernels = iter_tsvc_kernels(only=[key], corpus=corpus)
    assert kernels, f"{key} is not in the {corpus} corpus -- the corpus this test pins has changed"
    return kernels[0]


@pytest.mark.parametrize("corpus,key,index_array", GATHER_KERNELS)
def test_index_array_is_a_valid_subscript_permutation(corpus, key, index_array):
    kernel = load(corpus, key)
    boundary = first_nest(kernel)
    sizes = sample_sizes(kernel, boundary, seed=0, preset="S")
    inputs = make_inputs(boundary, sizes, seed=0, given=index_fills(kernel, boundary, sizes))

    ip = inputs[index_array]
    n = ip.shape[0]
    assert ip.dtype.kind in "iu"
    # a permutation of [0, n): every subscript in range, each used exactly once -> a real gather, and a
    # conflict-free scatter (the property optarena/tests/test_foundation_scatter_conflict_free.py guards).
    assert np.array_equal(np.sort(ip), np.arange(n, dtype=ip.dtype))


@pytest.mark.parametrize("corpus,key,index_array", GATHER_KERNELS)
def test_index_array_is_all_zeros_without_the_manifest_fill(corpus, key, index_array):
    # pins the bug being fixed: the plain float fill cast to an int dtype degenerates to b[0] every
    # iteration. If this ever stops holding, `given` has become dead weight and should go.
    kernel = load(corpus, key)
    boundary = first_nest(kernel)
    sizes = sample_sizes(kernel, boundary, seed=0, preset="S")
    ip = make_inputs(boundary, sizes, seed=0)[index_array]
    assert ip.size > 1 and len(np.unique(ip)) == 1 and ip.ravel()[0] == 0


@pytest.mark.parametrize("corpus,key,index_array", GATHER_KERNELS)
def test_index_fills_are_seeded_so_oracle_and_candidate_agree(corpus, key, index_array):
    # the oracle is built once and every cell validates against it: an unseeded fill would give the
    # candidate a different `ip` than the oracle saw and break validation for every gather kernel.
    kernel = load(corpus, key)
    boundary = first_nest(kernel)
    sizes = sample_sizes(kernel, boundary, seed=0, preset="S")
    a = index_fills(kernel, boundary, sizes)
    b = index_fills(kernel, boundary, sizes)
    assert np.array_equal(a[index_array], b[index_array])


@pytest.mark.parametrize("corpus,key,index_array", GATHER_KERNELS)
def test_index_fills_only_covers_manifest_declared_integer_arrays(corpus, key, index_array):
    # an integer array is not automatically a subscript; only what the manifest declares gets a fill.
    # Parametrized over BOTH corpora: the tsvc2_5 bare-stem path (foundation/<key>/<key>) resolves
    # through a different registry key than tsvc2's tsvc_2_<key>, so a per-corpus resolution regression
    # (the subfolder-restructure bug) is caught here, not only on the first tsvc2 kernel.
    kernel = load(corpus, key)
    boundary = first_nest(kernel)
    sizes = sample_sizes(kernel, boundary, seed=0, preset="S")
    fills = index_fills(kernel, boundary, sizes)
    assert set(fills) == {index_array}  # not the float data arrays, and not the __sym_out_* output scalars


def test_index_fills_empty_for_a_kernel_without_a_manifest():
    kernel = load("tsvc2", "vag")
    boundary = first_nest(kernel)
    sizes = sample_sizes(kernel, boundary, seed=0, preset="S")
    kernel.key = "no_such_kernel_anywhere"  # -> no manifest
    assert kernel.bench_name is None
    assert index_fills(kernel, boundary, sizes) == {}


def test_given_array_of_the_wrong_shape_is_rejected():
    # `given` is passed straight across the ABI as the kernel's buffer, so a mismatch must raise here
    # rather than corrupt memory in the compiled call.
    kernel = load("tsvc2", "vag")
    boundary = first_nest(kernel)
    sizes = sample_sizes(kernel, boundary, seed=0, preset="S")
    bad = {"ip": np.arange(3, dtype=np.int32)}
    with pytest.raises(ValueError, match="given array 'ip'"):
        make_inputs(boundary, sizes, seed=0, given=bad)


def test_given_array_of_the_wrong_dtype_is_rejected():
    kernel = load("tsvc2", "vag")
    boundary = first_nest(kernel)
    sizes = sample_sizes(kernel, boundary, seed=0, preset="S")
    n = make_inputs(boundary, sizes, seed=0)["ip"].shape[0]
    with pytest.raises(ValueError, match="given array 'ip'"):
        make_inputs(boundary, sizes, seed=0, given={"ip": np.arange(n, dtype=np.float64)})


def test_transient_scratch_keeps_its_own_fill():
    # only the manifest's non-transient arrays are `given`; everything else is untouched by this change.
    kernel = load("tsvc2", "vag")
    boundary = first_nest(kernel)
    sizes = sample_sizes(kernel, boundary, seed=0, preset="S")
    plain = make_inputs(boundary, sizes, seed=0)
    with_fills = make_inputs(boundary, sizes, seed=0, given=index_fills(kernel, boundary, sizes))
    for name, value in plain.items():
        if name != "ip":
            assert np.array_equal(value, with_fills[name]), f"{name} changed but is not a manifest index array"
