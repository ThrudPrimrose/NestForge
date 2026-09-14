# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Kernel optimization: render one kernel as a standalone CPF unit with one C entry, build and validate ``lib<kernel>.a``."""
from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

import dace
from dace import dtypes
from dace.codegen import cpf
from dace.ordered import OrderedSet
from dace.transformation.passes.canonicalize.finalize import finalize_for_target
from dace.transformation.passes.length_one_array_scalar_conversion import ConvertScalarsToLengthOneArrays

from nestforge.build.arena import call_native, diff_stats, dtype_floor, make_inputs, rung_atol, run_oracle
from nestforge.build.isolation import run_isolated
from nestforge.build.sdfg import BuildOptions, build_archive
from nestforge.build.toolchain import parse_params, raw_signature
from nestforge.corpus.translate import Prepared
from nestforge.ir.extract import Boundary
from nestforge.ir.libnode import ExternalCall
from nestforge.phases.normalize import Targets


@dataclass(slots=True)
class KernelSource:
    """The kernel's CPF translation unit, defining ``extern "C" <name>`` with parameters in ``abi_order``."""
    name: str
    unit: Path
    abi_order: List[str]
    boundary: Boundary

    @property
    def symbol(self) -> str:
        return self.name


@dataclass(slots=True)
class KernelVerdict:
    """A build compared to the NumPy oracle and timed; ``ok`` gates at ``fp_mode``."""
    fp_mode: str
    maxdiff: float
    md_rel: float
    dtype_floor: float
    time_us: float
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error and self.md_rel <= rung_atol(self.fp_mode, self.dtype_floor)


def failed_verdict(fp_mode: str, error: str) -> KernelVerdict:
    return KernelVerdict(fp_mode, float("inf"), float("inf"), 0.0, float("inf"), error)


def at_rung(verdict: KernelVerdict, fp_mode: str) -> KernelVerdict:
    """The same measurement gated at another FP rung."""
    return dataclasses.replace(verdict, fp_mode=fp_mode)


def abi_ready_copy(boundary: Boundary) -> dace.SDFG:
    """A copy of the kernel whose entry takes what ``ExternalCall`` passes: integer symbols as ``int64_t``
    and every data argument, a scalar input included, by pointer."""
    sdfg = copy.deepcopy(boundary.standalone_sdfg)
    for name in boundary.symbols:
        if sdfg.symbols[name] in dtypes.INTEGER_TYPES:
            sdfg.symbols[name] = dace.int64
    scalars = OrderedSet(name for name in boundary.inputs if isinstance(sdfg.arrays[name], dace.data.Scalar))
    # The pass rewrites a scalar into a length-1 array in place only for transients; these stay arguments.
    for name in scalars:
        sdfg.arrays[name].transient = True
    ConvertScalarsToLengthOneArrays(filter=scalars).apply_pass(sdfg, {})
    for name in scalars:
        sdfg.arrays[name].transient = False
    return sdfg


def default_schedule(boundary: Boundary, targets: Targets) -> dace.SDFG:
    """The kernel copy CPF renders: :func:`abi_ready_copy` finalized for the CPU."""
    if targets.gpu:
        raise NotImplementedError("GPU kernels need CPF's CUDA form")
    return finalize_for_target(abi_ready_copy(boundary), "cpu")


def schedule_kernel(ext: ExternalCall, boundary: Boundary, targets: Targets, out_dir: Path) -> KernelSource:
    """:func:`default_schedule` rendered by CPF into ``<out_dir>/<kernel>.cpp``, whose one entry is ``ext``'s symbol."""
    sdfg = default_schedule(boundary, targets)
    sdfg.name = ext.name
    rendering = cpf.render(sdfg, language="c++")
    out_dir.mkdir(parents=True, exist_ok=True)
    unit = out_dir / f"{ext.name}.cpp"
    unit.write_text(rendering.code)
    return KernelSource(ext.name, unit, list(rendering.arguments), boundary)


def build_kernel_library(src: KernelSource, compiler: str, flags: Optional[List[str]], out_dir: Path) -> Path:
    """Build ``<out_dir>/lib<kernel>.a`` from the CPF unit, plus its shared twin for validation."""
    archive = out_dir / f"lib{src.name}.a"
    opts = BuildOptions(compiler=compiler, flags=flags, link_external=True)
    build_archive([src.unit], None, archive, archive.with_suffix(".so"), opts)
    return archive


def use_kernel_library(ext: ExternalCall, lib_path: Path, symbol: str, abi_order: List[str]) -> None:
    """Point ``ext`` at a built library and select the extern-call expansion."""
    ext.lib_path, ext.symbol, ext.abi_order = str(lib_path), symbol, list(abi_order)
    ext.implementation = "ExternCall"


def measure_kernel(archive: Path, src: KernelSource, inputs: Dict[str, np.ndarray], oracle: Dict[str, np.ndarray],
                   sizes: Dict[str, int], reps: int, fp_mode: str) -> KernelVerdict:
    """Call the twin's C entry in a forked child: one run compared to ``oracle``, then ``reps`` timed runs.
    A crash or timeout comes back as a verdict with ``error`` set."""
    argtypes = [p.ctype for p in parse_params(raw_signature(src.unit.read_text(), src.symbol))]
    shared = archive.with_suffix(".so")

    def work() -> Dict[str, float]:
        outs, us = call_native(shared, src.symbol, src.abi_order, argtypes, src.boundary, inputs, sizes, reps)
        assert outs is not None, "call_native snapshots outputs unless told not to"
        md, md_rel = diff_stats(oracle, outs)
        return {"maxdiff": md, "md_rel": md_rel, "dtype_floor": dtype_floor(outs), "time_us": us}

    res = run_isolated(work)
    if "error" in res:
        return failed_verdict(fp_mode, str(res["error"]))
    return KernelVerdict(fp_mode, float(res["maxdiff"]), float(res["md_rel"]), float(res["dtype_floor"]),
                         float(res["time_us"]))


def validate_kernel(archive: Path,
                    src: KernelSource,
                    prep: Prepared,
                    sizes: Dict[str, int],
                    reps: int = 10,
                    fp_mode: str = "strict-ieee") -> KernelVerdict:
    """:func:`measure_kernel` on seeded inputs against the kernel's NumPy oracle."""
    inputs = make_inputs(src.boundary, sizes)
    oracle = run_oracle(prep, src.boundary, inputs, sizes)
    return measure_kernel(archive, src, inputs, oracle, sizes, reps, fp_mode)
