# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Kernel optimization on four tiny kernels: the default kernel is a standalone CPF unit, its one C entry takes
CPF's argument order with the types ``ExternalCall`` declares, ``lib<kernel>.a`` defines that entry, and every
built kernel matches its NumPy oracle."""

import re
import subprocess
from typing import Tuple

import pytest

import dace

from nestforge.build.toolchain import raw_signature, split_params
from nestforge.corpus.translate import prepare
from nestforge.ir.libnode import proto_and_call
from nestforge.phases.kernel import (
    build_kernel_library,
    default_schedule,
    schedule_kernel,
    use_kernel_library,
    validate_kernel,
)
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


@dace.program
def scaled(alpha: dace.float64, x: dace.float64[N], y: dace.float64[N]):
    for i in dace.map[0:N]:
        y[i] = alpha * x[i]


KERNELS = [vadd, stencil_row_sum, axpy_in_place, scaled]
KERNEL_IDS = ["vadd", "stencil_row_sum", "axpy_in_place", "scaled"]
#: Extents off any vector width, so a compiler's remainder loop runs too.
KERNEL_SIZES = [{"N": 1037}, {"M": 13, "N": 67}, {"N": 1037}, {"N": 1037}]
ENTRY_DEFINITION = re.compile(r'^extern "C" void (\w+)\s*\([^)]*\)\s*\{', re.M)


def lowered_kernel(program):
    """``program`` through phases 0-2: ``(parent SDFG, its one ExternalCall, that kernel's Boundary)``."""
    sdfg = program.to_sdfg(simplify=True)
    normalize(sdfg, Targets())
    full_fusion(sdfg, Targets())
    lowered = lower_nests_to_external_call(sdfg)
    assert len(lowered) == 1, [ext.name for ext, _ in lowered]
    ext, boundary = lowered[0]
    return sdfg, ext, boundary


def split_decl(decl: str) -> Tuple[str, str]:
    """``(type, name)`` of one C parameter declaration, qualifiers other than ``const`` dropped."""
    text = " ".join(re.sub(r"\b__restrict__\b", "", decl).split()).replace(" *", "*")
    name = re.split(r"[\s*]+", text)[-1]
    return text[: text.rfind(name)].strip(), name


def entry_params(src):
    return [split_decl(p) for p in split_params(raw_signature(src.unit.read_text(), src.symbol))]


@pytest.mark.parametrize("program", KERNELS, ids=KERNEL_IDS)
def test_the_default_kernel_is_a_standalone_cpf_unit(tmp_path, program):
    """The unit builds with a bare compiler: no DaCe header, no DaCe runtime entry, and the boundary SDFG the
    extraction produced stays untouched, since phases 3 and 4 may schedule it again."""
    _, ext, boundary = lowered_kernel(program)
    before = boundary.standalone_sdfg.to_json()

    src = schedule_kernel(ext, boundary, Targets(), tmp_path)

    text = src.unit.read_text()
    assert not re.search(r'#include\s*[<"]dace/', text)
    assert "__dace_" not in text
    assert "dace::" not in text
    assert boundary.standalone_sdfg.to_json() == before


def test_a_gpu_target_is_refused_until_cpf_renders_cuda():
    _, _, boundary = lowered_kernel(vadd)
    with pytest.raises(NotImplementedError, match="CUDA form"):
        default_schedule(boundary, Targets(gpu=True))


def test_the_unit_defines_one_entry_in_cpf_order_not_manifest_order(tmp_path):
    """The entry takes CPF's order (arrays by name, then scalars by name). For ``vadd`` the output ``a``
    sorts before the inputs, so the manifest's role order differs -- binding by it would swap same-typed
    pointers silently."""
    _, ext, boundary = lowered_kernel(vadd)

    src = schedule_kernel(ext, boundary, Targets(), tmp_path)

    assert ENTRY_DEFINITION.findall(src.unit.read_text()) == [ext.name]
    assert [name for _, name in entry_params(src)] == src.abi_order == ["a", "b", "c", "N"]
    assert src.abi_order != list(ext.config["input_args"])
    assert src.symbol == ext.name


@pytest.mark.parametrize("program", [vadd, scaled], ids=["vadd", "scaled"])
def test_the_entry_declares_each_parameter_as_the_external_call_prototype_does(tmp_path, program):
    """C linkage matches on the name alone, so a prototype/definition type mismatch links cleanly and
    corrupts the call: an ``int`` symbol must arrive as the ``int64_t`` the parent declares, and every data
    argument, the scalar ``alpha`` included, as a pointer."""
    sdfg, ext, boundary = lowered_kernel(program)
    src = schedule_kernel(ext, boundary, Targets(), tmp_path)
    use_kernel_library(ext, tmp_path / "unused.a", src.symbol, src.abi_order)

    proto, _ = proto_and_call(ext, next(s for s in sdfg.all_states() if ext in s.nodes()))

    declared = [split_decl(p) for p in split_params(re.search(r"\((.*)\)", proto).group(1))]
    assert declared == entry_params(src)


@pytest.mark.e2e
def test_the_archive_defines_the_entry_once_and_no_dace_runtime(tmp_path):
    _, ext, boundary = lowered_kernel(vadd)
    src = schedule_kernel(ext, boundary, Targets(), tmp_path / "gen")

    archive = build_kernel_library(src, "g++", None, tmp_path / "lib")

    assert archive == tmp_path / "lib" / f"lib{ext.name}.a"
    members = subprocess.run(["ar", "t", str(archive)], capture_output=True, text=True, check=True).stdout.split()
    assert members == [f"{ext.name}.o"]
    defined = subprocess.run(["nm", "--defined-only", str(archive)], capture_output=True, text=True, check=True).stdout
    assert re.findall(rf"^\S+ T ({re.escape(ext.name)})$", defined, re.M) == [ext.name]
    assert "__dace_" not in defined and "__program_" not in defined


@pytest.mark.e2e
@pytest.mark.parametrize("program, sizes", list(zip(KERNELS, KERNEL_SIZES)), ids=KERNEL_IDS)
def test_the_built_kernel_matches_its_numpy_oracle_bit_for_bit(tmp_path, program, sizes):
    """The shipped entry on seeded inputs, forked, against the kernel's NumPy reference at the strict rung;
    the in-place kernel is restored before every timed rep."""
    _, ext, boundary = lowered_kernel(program)
    src = schedule_kernel(ext, boundary, Targets(), tmp_path / "gen")
    archive = build_kernel_library(src, "g++", None, tmp_path / "lib")
    prep = prepare(boundary, ext.name, tmp_path / "ref")

    verdict = validate_kernel(archive, src, prep, sizes, reps=3)

    assert verdict.error == "", verdict.error
    assert verdict.ok and verdict.maxdiff == 0.0, verdict
    assert verdict.time_us > 0.0
