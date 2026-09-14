# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Emit an OptArena ``BenchSpec`` manifest (symbols, array shapes, dtypes) for an extracted nest.

The manifest mirrors the fields hpcagent_bench's translator needs (verified against its ``gemm`` /
``vsumr`` kernels):
  * ``input_args``  -- the FULL positional signature of the numpy kernel (arrays -- inputs, outputs and
    scratch transients alike, every one caller-allocated -- then symbols),
  * ``array_args``  -- which of those are array pointers,
  * ``output_args`` -- which arrays are written,
  * ``init.arrays`` -- ``{name: {shape, dtype}}``,
  * ``parameters``  -- AOT representative sizes per symbol (preset -> {symbol: size}).
The numpy kernel's arg order (from :mod:`nestforge.ir.emit_numpy`) must match ``input_args``.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

import dace
from dace import symbolic

from nestforge.ir.emit_numpy import expand_nested_sdfg_inputs, maxsize_loop_scratch, scratch_arrays
from nestforge.ir.extract import Boundary

DEFAULT_SIZE = 1 << 16

#: dtypes a boundary symbol can carry that make it a VALUE scalar (not an integer sizing symbol).
FLOAT_DTYPES = frozenset({"float64", "float32", "float16", "float128"})


def symbol_dtype_name(sdfg: dace.SDFG, s: str) -> str:
    """numpy dtype name of a boundary symbol from the SDFG symbol table (``"float64"`` for a staged
    ``a_index = a[i]`` value read leaked into the boundary, ``"int64"`` for a size / index symbol).
    Falls back to ``int64`` for a symbol the SDFG does not type."""
    if s in sdfg.symbols:
        return np.dtype(sdfg.symbols[s].type).name
    return "int64"


def sized_sdfg(boundary: Boundary) -> dace.SDFG:
    """The standalone SDFG in the SAME form :func:`nestforge.emit_numpy.nest_to_numpy` emits from.

    Both passes affect what the manifest must declare: nested-input expansion settles which containers the
    body references, and ``maxsize_loop_scratch`` widens a loop-variable-sized scratch buffer to a bound the
    caller can allocate. Reading shapes off the raw ``standalone_sdfg`` would declare the pre-widening shape
    the kernel does not index against.
    """
    return maxsize_loop_scratch(expand_nested_sdfg_inputs(boundary.standalone_sdfg), boundary.symbols)


def arg_order(boundary: Boundary, sdfg: dace.SDFG, arrays: List[str]) -> List[str]:
    """Arrays (inputs, extra outputs, scratch), then symbols -- identical to the numpy signature.

    ``arrays`` is an already-computed :func:`array_names` result: every caller has one, so re-deriving it
    here would walk ``scratch_arrays``/``sized_sdfg`` a second time for the same boundary.
    """
    args = list(arrays)
    args += [s for s in boundary.symbols if s not in args]
    return args


def array_names(boundary: Boundary, sdfg: dace.SDFG) -> List[str]:
    """Every caller-allocated buffer, in numpy-signature order.

    Scratch transients are parameters too: the C-style memory model makes the kernel allocate nothing, so a
    non-scalar transient the body indexes must cross the ABI like any other array. Omitting it here would
    leave the translator no declaration for a name the body references.
    """
    names = list(boundary.inputs)
    names += [o for o in boundary.outputs if o not in boundary.inputs]
    names += [s for s in scratch_arrays(sdfg) if s not in names]
    return names


def shape_str(shape: Sequence[Any]) -> str:
    dims = [symbolic.symstr(d) for d in shape]
    return "(" + ", ".join(dims) + ("," if len(dims) == 1 else "") + ")"


def dtype_str(desc: dace.data.Data) -> str:
    return np.dtype(desc.dtype.type).name


def manifest_dict(boundary: Boundary, name: str, sizes: Optional[Dict[str, int]] = None, preset: str = "S") -> Dict:
    """Build the OptArena manifest dict for ``boundary``'s standalone SDFG."""
    sdfg = sized_sdfg(boundary)
    arrays = array_names(boundary, sdfg)
    init_arrays = {}
    for a in arrays:
        desc = sdfg.arrays[a]
        init_arrays[a] = {"shape": shape_str(desc.shape), "dtype": dtype_str(desc)}
    sizes = sizes or {s: DEFAULT_SIZE for s in boundary.symbols}
    # A boundary symbol carries its dtype from the SDFG. An integer symbol is a size / index -> the
    # ``parameters`` preset table. A FLOAT symbol is a value scalar (a staged ``a_index = a[i]`` read that
    # the nest extractor carried into the boundary) -> ``init.scalars`` with a float default, so the
    # translator declares it ``double`` and does NOT truncate the value to ``int64``.
    int_params: Dict[str, int] = {}
    float_scalars: Dict[str, float] = {}
    for s in boundary.symbols:
        if symbol_dtype_name(sdfg, s) in FLOAT_DTYPES:
            float_scalars[s] = 0.0
        else:
            int_params[s] = int(sizes.get(s, DEFAULT_SIZE))
    init: Dict = {"arrays": init_arrays}
    if float_scalars:
        init["scalars"] = float_scalars
    return {
        "name": name,
        "short_name": name,
        "func_name": name,
        "relative_path": "extended",
        "kind": "microkernel",
        "level": 1,
        "parameters": {
            preset: int_params
        },
        "input_args": arg_order(boundary, sdfg, arrays),
        "array_args": arrays,
        "output_args": list(boundary.outputs),
        "init": init,
        # the literal stays: hpcagent_bench.spec hoists taxonomy.track into BenchSpec
        "taxonomy": {
            "track": "foundation"
        },
    }
