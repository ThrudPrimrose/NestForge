# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np
import dace

from nestforge.phases.scopes import outer
from nestforge.ir.extract import extract_nest_to_sdfg
from nestforge.ir.emit_numpy import load_emitted, nest_to_numpy
from nestforge.corpus.translate import prepare, emit_sources

N = dace.symbol('N')


@dace.program
def vadd(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N]:
        C[i] = A[i] + B[i]


def boundary():
    sdfg = vadd.to_sdfg(simplify=True)
    psdfg, node = outer(sdfg)[0]
    return extract_nest_to_sdfg(psdfg, node, name="vadd_nest")


def test_numpy_emit_runs():
    b = boundary()
    src = nest_to_numpy(b, fn_name="vadd")
    mod = load_emitted(src, "vadd")
    A = np.random.default_rng(0).random(32)
    B = np.random.default_rng(1).random(32)
    C = np.zeros(32)
    mod.vadd(A=A, B=B, C=C, N=32)
    np.testing.assert_allclose(C, A + B)


def test_translate_to_c(tmp_path):
    b = boundary()
    prep = prepare(b, "vadd", tmp_path / "kern")
    assert prep.numpy_path.exists() and prep.yaml_path.exists()
    srcs = emit_sources(prep, tmp_path / "gen", target="c")
    c_files = [p for p in srcs if p.suffix == ".c"]
    assert c_files, f"no C emitted; got {srcs}"
    text = c_files[0].read_text()
    # correct ABI: three double* arrays + an int64 size (order = input_args)
    assert "double *restrict A" in text
    assert "double *restrict C" in text
    assert "int64_t N" in text
    assert "(A[i] + B[i])" in text
    assert "C[i] = " in text


if __name__ == "__main__":
    test_numpy_emit_runs()
    import tempfile
    import pathlib
    test_translate_to_c(pathlib.Path(tempfile.mkdtemp()))
    print("translate OK")


@dace.program
def gather_two_map(A: dace.float64[N], idx: dace.int64[N], C: dace.float64[N]):
    T = np.empty_like(A)
    for k in dace.map[0:N]:
        T[k] = A[idx[k]]
    for k in dace.map[0:N]:
        C[k] = T[k] * 2.0


def test_a_fused_maps_scalar_transient_is_spelled_the_same_inside_and_out():
    """MapFusion stages the fused intermediate in a size-1 transient reached through a nested SDFG. That
    connector is non-transient INSIDE and transient OUTSIDE, and ``transient`` is what picks the spelling
    (scalar transient = bare local, everything else = indexed), so the two bodies disagreed: the inner one
    wrote ``__map_fusion_T[0] = ...`` while the outer read it bare, and ``scratch_arrays`` allocated
    neither. In numpy that is a NameError; translated to C it was an undeclared identifier.

    Runs the emitted kernel rather than grepping the text: the spelling only matters because it decides
    whether the value survives from the write to the read.
    """
    from nestforge.granularity import granularity_ladder
    from nestforge.phases.scopes import lower_nests_to_external_call

    sdfg = gather_two_map.to_sdfg(simplify=True)
    granularity_ladder(sdfg, max_points=2)[-1].apply(sdfg)  # the maximal (fused) rung
    calls = lower_nests_to_external_call(sdfg, "outer")
    assert calls, "nothing lowered; the fixture no longer produces an offloadable nest"
    ext, b = calls[0]
    mod = load_emitted(nest_to_numpy(b, fn_name="fused"), "fused")

    rng = np.random.default_rng(0)
    A = rng.random(32)
    idx = rng.permutation(32).astype(np.int64)
    C = np.zeros(32)
    mod.fused(A=A, idx=idx, C=C, N=32)
    np.testing.assert_allclose(C, A[idx] * 2.0)


def test_the_standalone_preamble_is_the_live_helpers():
    """An emitted standalone kernel must compute what the validated in-process one computes. The preamble
    is generated from int_floor/int_ceil rather than hand-copied, so an edit to either cannot leave the
    emitted text behind -- assert the property, not the generation trick."""
    from nestforge.ir import emit_numpy

    namespace = {}
    exec(emit_numpy.STANDALONE_PREAMBLE, namespace)  # noqa: S102 -- the point is that it is runnable
    for a, b in ((7, 2), (-7, 2), (7, -2), (-7, -2), (8, 4), (0, 3)):
        assert namespace["int_floor"](a, b) == emit_numpy.int_floor(a, b), (a, b)
        assert namespace["int_ceil"](a, b) == emit_numpy.int_ceil(a, b), (a, b)
