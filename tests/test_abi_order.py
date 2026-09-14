# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The ABI-order contract: whatever order nest-forge binds ctypes arguments in MUST be the order the
emitted kernel actually declares.

numpyto emits the C signature in ``param_order()`` -- arrays sorted alphabetically, then scalars sorted --
and its IR docstring says it "deliberately ignores input_args for ordering". nest-forge's manifest
``input_args`` is a ROLE order (inputs, then outputs, then symbols). The two coincide only by luck of the
alphabet, and every in-tree demo kernel (vadd, fma) happens to be lucky: ``[A, B, C, N]`` is already
sorted. A kernel whose OUTPUT sorts before its INPUT breaks the tie -- and since both are ``double*``,
ctypes cannot catch it: the kernel writes through the wrong pointer, silently.

`harness.signature_order` exists precisely to parse the real emitted signature. These tests pin that the
binding follows it.
"""

import pytest

import dace

from nestforge.ir.extract import extract_nest_to_sdfg
from nestforge.build.harness import signature_order
from nestforge.phases.scopes import parallel_top_level_maps
from nestforge.corpus.translate import emit_sources, prepare

N = dace.symbol("N")


@dace.program
def writes_a_reads_b(a: dace.float64[N], b: dace.float64[N]):
    """Role order is [b, a, N] (input, output, symbol); sorted order is [a, b, N]. The output `a` sorts
    BEFORE the input `b`, so the two orders disagree -- the exact shape the lucky demo kernels dodge."""
    for i in dace.map[0:N]:
        a[i] = b[i] + 1.0


def prepared_nest(tmp_path):
    sdfg = writes_a_reads_b.to_sdfg(simplify=True)
    parent, node = parallel_top_level_maps(sdfg)[0]
    boundary = extract_nest_to_sdfg(parent, node, name="wab")
    prep = prepare(boundary, "wab", tmp_path, sizes={"N": 32})
    return prep, boundary


@pytest.mark.integration  # runs the numpyto emitter
def test_emitted_signature_disagrees_with_manifest_role_order(tmp_path):
    """Pins the HAZARD itself, so nobody "simplifies" a binder back onto manifest order.

    This is not a bug -- numpyto is entitled to its own parameter order -- it is the reason every binder
    must parse the emitted signature. The day these coincide for this kernel, the guard below stops
    guarding anything.
    """
    prep, boundary = prepared_nest(tmp_path)
    csrc = next(s for s in emit_sources(prep, tmp_path, target="c") if s.suffix == ".c" and "pluto" not in s.name)
    emitted = signature_order(csrc.read_text(), "wab_fp64")
    assert emitted == ["a", "b", "N"]  # sorted arrays, then scalars
    assert list(prep.manifest["input_args"]) == ["b", "a", "N"]  # role order: input, output, symbol
    assert emitted != list(prep.manifest["input_args"]), (
        "the two orders now coincide for this kernel -- pick one whose output still sorts before its input"
    )


def test_a_prototype_above_the_definition_does_not_widen_the_capture():
    """`raw_signature` anchors on `void <symbol>(...) {` so a doc comment naming the function cannot match.
    With a non-greedy `(.*?)` that anchor introduced the opposite failure: the dot backtracks ACROSS a
    preceding PROTOTYPE, capturing from the declaration's `(` to the definition's `)`. The caller then
    binds a garbage parameter list -- or, since both drivers now catch ValueError, silently drops the
    native column. `[^)]*` cannot cross a closing paren at all."""
    from nestforge.build.toolchain import raw_signature

    declared_then_defined = (
        "void s000_fp64(double *restrict a, const double *restrict b, int64_t LEN_1D);\n"
        "\n"
        'extern "C" void s000_fp64(double *restrict a, const double *restrict b, '
        "int64_t LEN_1D) {\n  return;\n}\n"
    )
    params = raw_signature(declared_then_defined, "s000_fp64")
    assert params == "double *restrict a, const double *restrict b, int64_t LEN_1D"
    assert ";" not in params and "void" not in params

    # the original motivation still holds: a doc COMMENT naming the entry is not a definition
    commented = "// s000_fp64 (s000): copy b into a\nvoid s000_fp64(double *a, int64_t n) {\n}\n"
    assert raw_signature(commented, "s000_fp64") == "double *a, int64_t n"
