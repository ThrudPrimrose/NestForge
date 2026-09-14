# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""ABI binding for translator-emitted kernels: parse the emitted signature and call it through ctypes."""
from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from nestforge.build.arena import CTYPE, call_native, scalar_ctype
from nestforge.build.toolchain import raw_signature


def signature_order(text: str, symbol: str, lang: str = "c") -> List[str]:
    """Parameter names of the kernel entry, in declaration order; the emitted C order (sorted arrays, then
    symbols) is NOT the manifest ``input_args`` order, so args must bind to this or a size lands in a
    pointer slot."""
    params = raw_signature(text, symbol, lang)
    if lang == "fortran":
        return [a.strip() for a in params.replace("&", " ").split(",") if a.strip()]
    return [p.strip().split()[-1].lstrip("*") for p in params.split(",") if p.strip() and p.strip() != "void"]


def c_argtypes(order: List[str], boundary) -> list:
    """ctypes type per C parameter: array name -> pointer-to-dtype, size/index symbol -> int64, value scalar -> its SDFG dtype."""
    sdfg = boundary.standalone_sdfg
    return [
        ctypes.POINTER(CTYPE[np.dtype(sdfg.arrays[a].dtype.type).name]) if a in sdfg.arrays else scalar_ctype(sdfg, a)
        for a in order
    ]


def call_c(so: Path,
           symbol: str,
           order: List[str],
           argtypes: list,
           boundary,
           inputs,
           sizes,
           reps: int,
           copy_outputs: bool = True) -> Tuple[Optional[Dict[str, np.ndarray]], float]:
    """Bind by the C signature order, run once for correctness, then time ``reps`` calls, mutating
    ``inputs`` in place. Callers must run this in a forked child."""
    return call_native(so,
                       symbol,
                       order,
                       argtypes,
                       boundary,
                       inputs,
                       sizes,
                       reps,
                       copy_inputs=False,
                       copy_outputs=copy_outputs)
