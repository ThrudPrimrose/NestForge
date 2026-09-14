# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Kernel optimization on three tiny kernels: the default schedule tiles the kernel, the wrapper TU exposes one C entry
in ``__program``'s argument order with the types ``ExternalCall`` declares, ``lib<kernel>.a`` defines that
entry, and every built kernel matches its NumPy oracle."""
import re
import subprocess

import pytest

import dace
from dace.sdfg import nodes

from nestforge.build.toolchain import raw_signature, split_params
from nestforge.corpus.translate import prepare
from nestforge.ir.libnode import proto_and_call
from nestforge.phases.kernel import (build_kernel_library, default_schedule, schedule_kernel, split_decl,
                                     use_kernel_library, validate_kernel)
from nestforge.phases.normalize import Targets, normalize
from nestforge.phases.schedule import full_fusion
from nestforge.phases.scopes import lower_nests_to_external_call

N = dace.symbol("N")
M = dace.symbol("M")


@dace.program
def vadd(b: dace.float64[N], c: dace.float64[N], a: dace.float64[N]):
    for i in dace.map[0:N]:
        a[i] = b[i] + c[i]


@dace.program
def stencil_row_sum(a: dace.float64[M, N], r: dace.float64[M]):
    for i in dace.map[0:M]:
        acc = 0.0
        for j in range(1, N - 1):
            acc = acc + (0.25 * a[i, j - 1] + 0.5 * a[i, j] + 0.25 * a[i, j + 1])
        r[i] = acc


@dace.program
def axpy_in_place(a: dace.float64[N], b: dace.float64[N]):
    for i in dace.map[0:N]:
        a[i] = a[i] * 2.0 + b[i]


KERNELS = [vadd, stencil_row_sum, axpy_in_place]
KERNEL_IDS = ["vadd", "stencil_row_sum", "axpy_in_place"]
#: Extents off the tile width, so the remainder path runs too.
KERNEL_SIZES = [{"N": 1037}, {"M": 13, "N": 67}, {"N": 1037}]


def lowered_kernel(program):
    """``program`` through phases 0-2: ``(parent SDFG, its one ExternalCall, that kernel's Boundary)``."""
    sdfg = program.to_sdfg(simplify=True)
    normalize(sdfg, Targets())
    full_fusion(sdfg, Targets())
    lowered = lower_nests_to_external_call(sdfg)
    assert len(lowered) == 1, [ext.name for ext, _ in lowered]
    ext, boundary = lowered[0]
    return sdfg, ext, boundary


def map_labels(sdfg):
    return [n.map.label for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry)]


def entry_params(src):
    return [split_decl(p) for p in split_params(raw_signature(src.wrapper.read_text(), src.symbol))]


@pytest.mark.parametrize("program", KERNELS, ids=KERNEL_IDS)
def test_the_default_schedule_tiles_a_copy_of_the_kernel(program):
    """The vectorizer splits the kernel's map into a tiled main map plus a remainder; the boundary SDFG the
    extraction produced stays untiled, since phases 3 and 4 may schedule it again."""
    _, _, boundary = lowered_kernel(program)
    before = map_labels(boundary.standalone_sdfg)

    scheduled = default_schedule(boundary, Targets())

    assert not any("__tile_main" in label for label in before), before
    assert any("__tile_main" in label for label in map_labels(scheduled)), map_labels(scheduled)
    assert len(map_labels(scheduled)) > len(before)
    assert map_labels(boundary.standalone_sdfg) == before


def test_a_gpu_target_is_refused_until_offloading_exists():
    _, _, boundary = lowered_kernel(vadd)
    with pytest.raises(NotImplementedError, match="phase 2.5"):
        default_schedule(boundary, Targets(gpu=True))


def test_the_wrapper_defines_one_entry_in_program_order_not_manifest_order(tmp_path):
    """The entry takes ``__program``'s order (arrays sorted, then symbols). For ``vadd`` the output ``a``
    sorts before the inputs, so the manifest's role order differs -- binding by it would swap same-typed
    pointers silently."""
    _, ext, boundary = lowered_kernel(vadd)

    src = schedule_kernel(ext, boundary, Targets(), tmp_path)

    text = src.wrapper.read_text()
    assert re.findall(r"^\s*void\s+(\w+)\s*\([^)]*\)\s*\{", text, re.M) == [ext.name]
    assert f"__dace_init_{src.program.name}(static_cast<int>(N))" in text
    assert [name for _, name in entry_params(src)] == src.abi_order == ["a", "b", "c", "N"]
    assert src.abi_order != list(ext.config["input_args"])
    assert src.symbol == ext.name


def test_the_entry_declares_each_parameter_as_the_external_call_prototype_does(tmp_path):
    """C linkage matches on the name alone, so a prototype/definition type mismatch links cleanly and
    corrupts the call: the entry's by-value symbol must be the ``int64_t`` the parent declares, not DaCe's
    ``int``, and every array must arrive as a pointer."""
    sdfg, ext, boundary = lowered_kernel(vadd)
    src = schedule_kernel(ext, boundary, Targets(), tmp_path)
    use_kernel_library(ext, tmp_path / "unused.a", src.symbol, src.abi_order)

    proto, _ = proto_and_call(ext, next(s for s in sdfg.all_states() if ext in s.nodes()))

    declared = [split_decl(p) for p in split_params(re.search(r"\((.*)\)", proto).group(1))]
    defined = entry_params(src)
    assert [name for _, name in declared] == [name for _, name in defined]
    for (proto_type, name), (entry_type, _) in zip(declared, defined):
        assert ("*" in proto_type) == ("*" in entry_type), name
        if "*" not in proto_type:
            assert proto_type == entry_type == "int64_t", name
    assert ext.implementation == "ExternCall" and ext.abi_order == ["a", "b", "c", "N"]


@pytest.mark.e2e
def test_the_archive_defines_the_entry_once_beside_the_program(tmp_path):
    _, ext, boundary = lowered_kernel(vadd)
    src = schedule_kernel(ext, boundary, Targets(), tmp_path / "gen")

    archive = build_kernel_library(src, "g++", None, tmp_path / "lib")

    assert archive == tmp_path / "lib" / f"lib{ext.name}.a"
    members = subprocess.run(["ar", "t", str(archive)], capture_output=True, text=True, check=True).stdout.split()
    assert sorted(members) == sorted([f"{src.program.name}.o", f"{ext.name}_entry.o"])
    defined = subprocess.run(["nm", "--defined-only", str(archive)], capture_output=True, text=True, check=True).stdout
    assert re.findall(rf"^\S+ T ({re.escape(ext.name)})$", defined, re.M) == [ext.name]
    assert re.search(rf" T __program_{re.escape(src.program.name)}$", defined, re.M)


@pytest.mark.e2e
@pytest.mark.parametrize("program, sizes", list(zip(KERNELS, KERNEL_SIZES)), ids=KERNEL_IDS)
def test_the_built_kernel_matches_its_numpy_oracle_bit_for_bit(tmp_path, program, sizes):
    """The shipped entry (init, run, exit) on seeded inputs, forked, against the kernel's NumPy reference
    at the strict rung; the in-place kernel is restored before every timed rep."""
    _, ext, boundary = lowered_kernel(program)
    src = schedule_kernel(ext, boundary, Targets(), tmp_path / "gen")
    archive = build_kernel_library(src, "g++", None, tmp_path / "lib")
    prep = prepare(boundary, ext.name, tmp_path / "ref")

    verdict = validate_kernel(archive, src, prep, sizes, reps=3)

    assert verdict.error == "", verdict.error
    assert verdict.ok and verdict.maxdiff == 0.0, verdict
    assert verdict.time_us > 0.0
