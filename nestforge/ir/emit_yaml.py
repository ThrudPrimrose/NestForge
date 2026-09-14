# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Emit an OptArena BenchSpec manifest (symbols, array shapes, dtypes) for an extracted nest.

Field names mirror what hpcagent_bench's translator expects.
"""

from __future__ import annotations

from typing import Any
from collections.abc import Sequence

import numpy as np

import dace
from dace import symbolic

from nestforge.ir.emit_numpy import expand_nested_sdfg_inputs, maxsize_loop_scratch, scratch_arrays
from nestforge.ir.extract import Boundary

DEFAULT_SIZE = 1 << 16

# dtypes that mark a boundary symbol as a value scalar, not an integer sizing symbol.
FLOAT_DTYPES = frozenset({"float64", "float32", "float16", "float128"})


def symbol_dtype_name(sdfg: dace.SDFG, s: str) -> str:
    if s in sdfg.symbols:
        return np.dtype(sdfg.symbols[s].type).name
    return "int64"


def sized_sdfg(boundary: Boundary) -> dace.SDFG:
    # widen scratch after expanding nested inputs, or the shapes below won't match the kernel body
    return maxsize_loop_scratch(expand_nested_sdfg_inputs(boundary.standalone_sdfg), boundary.symbols)


def arg_order(boundary: Boundary, sdfg: dace.SDFG, arrays: list[str]) -> list[str]:
    # arrays is the caller's already-computed array_names() result
    args = list(arrays)
    args += [s for s in boundary.symbols if s not in args]
    return args


def array_names(boundary: Boundary, sdfg: dace.SDFG) -> list[str]:
    # scratch transients cross the ABI too: the C-style model allocates nothing inside the kernel
    names = list(boundary.inputs)
    names += [o for o in boundary.outputs if o not in boundary.inputs]
    names += [s for s in scratch_arrays(sdfg) if s not in names]
    return names


def shape_str(shape: Sequence[Any]) -> str:
    dims = [symbolic.symstr(d) for d in shape]
    return "(" + ", ".join(dims) + ("," if len(dims) == 1 else "") + ")"


def dtype_str(desc: dace.data.Data) -> str:
    return np.dtype(desc.dtype.type).name


def manifest_dict(
    boundary: Boundary, name: str, sizes: dict[str, int] | None = None, preset: str = "S"
) -> dict[str, Any]:
    """Build the OptArena manifest dict for boundary's standalone SDFG."""
    sdfg = sized_sdfg(boundary)
    arrays = array_names(boundary, sdfg)
    init_arrays = {}
    for a in arrays:
        desc = sdfg.arrays[a]
        init_arrays[a] = {"shape": shape_str(desc.shape), "dtype": dtype_str(desc)}
    sizes = sizes or dict.fromkeys(boundary.symbols, DEFAULT_SIZE)
    int_params: dict[str, int] = {}
    float_scalars: dict[str, float] = {}
    for s in boundary.symbols:
        # a float symbol is a staged scalar read, not a size -- route to init.scalars so the
        # translator declares it double instead of truncating it to int64
        if symbol_dtype_name(sdfg, s) in FLOAT_DTYPES:
            float_scalars[s] = 0.0
        else:
            int_params[s] = int(sizes.get(s, DEFAULT_SIZE))
    init: dict[str, Any] = {"arrays": init_arrays}
    if float_scalars:
        init["scalars"] = float_scalars
    return {
        "name": name,
        "func_name": name,
        "relative_path": "extended",
        "level": 1,
        "parameters": {preset: int_params},
        "input_args": arg_order(boundary, sdfg, arrays),
        "array_args": arrays,
        "output_args": list(boundary.outputs),
        "init": init,
    }
