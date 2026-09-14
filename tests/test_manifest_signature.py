# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The manifest's stated invariant: ``input_args`` IS the emitted numpy kernel's positional signature.

The two are consumed together (``translate.prepare`` writes the numpy source and the manifest side by
side), and the translator derives its C parameter list -- which names are array pointers, which are
scalars -- by walking ``input_args``. A name in the numpy signature but missing from ``input_args`` is
therefore never declared, and the emitted C references an undeclared identifier.

Scratch transients are the case that breaks the tie: the C-style memory model makes every non-scalar
transient a caller-allocated parameter, so it sits in the numpy signature between the outputs and the size
symbols and must appear in the manifest at exactly that position.
"""

import ast

import numpy as np

import dace

from nestforge.ir.emit_numpy import scratch_arrays, sdfg_to_numpy
from nestforge.ir.emit_yaml import manifest_dict
from nestforge.ir.extract import whole_program_boundary

N = dace.symbol("N")


@dace.program
def two_nest(a: dace.float64[N], out: dace.float64[N]):
    """Two maps chained through a non-scalar transient -- ``tmp`` stays internal (it is neither an input
    nor an output of the program), so it reaches the emitter as a scratch buffer."""
    tmp = np.empty(N, dace.float64)
    for i in dace.map[0:N]:
        tmp[i] = a[i] * 2.0
    for i in dace.map[0:N]:
        out[i] = tmp[i] + 1.0


def signature_of(source: str) -> list:
    """Positional parameter names of the single ``def`` in emitted numpy source."""
    fn = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef))
    return [a.arg for a in fn.args.args]


def scratch_boundary():
    return whole_program_boundary(two_nest.to_sdfg(simplify=True))


def test_scratch_transient_reaches_the_emitter():
    """Guards the fixture itself: if simplify ever folds ``tmp`` away, the tests below stop testing."""
    boundary = scratch_boundary()
    assert scratch_arrays(boundary.standalone_sdfg) == ["tmp"]
    assert "tmp" not in boundary.inputs and "tmp" not in boundary.outputs


def test_input_args_equals_numpy_signature_with_scratch():
    boundary = scratch_boundary()
    manifest = manifest_dict(boundary, "two_nest")
    emitted = signature_of(sdfg_to_numpy(boundary.standalone_sdfg, fn_name="two_nest"))
    assert emitted == ["a", "out", "tmp", "N"]  # arrays, scratch, then symbols
    assert list(manifest["input_args"]) == emitted


def test_scratch_is_declared_as_an_allocatable_array():
    """A scratch buffer in ``input_args`` must also be typed as an array, or the translator reads it as a
    scalar parameter; its shape must be resolvable from the kernel's own size symbols."""
    manifest = manifest_dict(scratch_boundary(), "two_nest")
    assert "tmp" in manifest["array_args"]
    assert manifest["init"]["arrays"]["tmp"] == {"shape": "(N,)", "dtype": "float64"}
    assert "tmp" not in manifest["output_args"]  # internal, not a result the caller reads back
