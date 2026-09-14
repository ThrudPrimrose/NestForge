# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Kernel optimization: render one kernel as a standalone CPF unit with one C entry, build and validate ``lib<kernel>.a``."""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

import dace
from dace import dtypes
from dace.codegen import cpf
from dace.transformation.passes.canonicalize.finalize import finalize_for_target, offload_to_gpu

from nestforge.build.arena import (
    call_native,
    call_on_device,
    diff_stats,
    dtype_floor,
    make_inputs,
    rung_atol,
    run_oracle,
)
from nestforge.build.flags import cuda_base_flags
from nestforge.build.isolation import run_isolated
from nestforge.build.sdfg import BuildOptions, build_archive, build_cuda_archive, program_compiler
from nestforge.build.toolchain import LIBOMP, cudart_dir, cudart_link_flags, parse_params, raw_signature
from nestforge.corpus.translate import Prepared
from nestforge.ir.extract import Boundary
from nestforge.ir.libnode import ExternalCall
from nestforge.phases.offload import kernel_device


@dataclass(slots=True)
class KernelSource:
    """The kernel's CPF translation unit, defining ``extern "C" <name>`` with parameters in ``abi_order``."""

    name: str
    unit: Path
    abi_order: List[str]
    boundary: Boundary
    device: str

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
    """A copy of the kernel whose entry takes the integer symbols as the ``int64_t`` ``ExternalCall`` passes."""
    sdfg = copy.deepcopy(boundary.standalone_sdfg)
    for name in boundary.symbols:
        if sdfg.symbols[name] in dtypes.INTEGER_TYPES:
            sdfg.symbols[name] = dace.int64
    return sdfg


def cpu_schedule(boundary: Boundary) -> dace.SDFG:
    """The kernel copy the C++ unit renders: :func:`abi_ready_copy` finalized for the CPU."""
    return finalize_for_target(abi_ready_copy(boundary), "cpu")


def gpu_schedule(boundary: Boundary) -> dace.SDFG:
    """The kernel copy the CUDA unit renders: offloaded whole, so its arrays arrive as device pointers, then
    finalized for the GPU."""
    sdfg = abi_ready_copy(boundary)
    offload_to_gpu(sdfg)
    return finalize_for_target(sdfg, "gpu")


def build_cpu_library(unit: Path, compiler: str, flags: Optional[List[str]], archive: Path) -> None:
    opts = BuildOptions(compiler=compiler, flags=flags, openmp=LIBOMP, link_external=True)
    build_archive([unit], None, archive, archive.with_suffix(".so"), opts)


def build_gpu_library(unit: Path, compiler: str, flags: Optional[List[str]], archive: Path) -> None:
    chosen = flags if flags is not None else cuda_base_flags(cpf.CUDA_BUILD_FLAGS)
    build_cuda_archive(unit, archive, archive.with_suffix(".so"), compiler, chosen)


def process_runtime_libraries() -> List[str]:
    """libomp, the process's one OpenMP runtime, spelled for the program's linker."""
    return LIBOMP.link_flags(program_compiler())


def cpu_runtime_libraries(compiler: str) -> List[str]:
    return process_runtime_libraries()


def gpu_runtime_libraries(compiler: str) -> List[str]:
    """libomp, and the ``libcudart`` the kernel's own nvcc links."""
    return [*process_runtime_libraries(), *cudart_link_flags(cudart_dir(compiler))]


@dataclass(frozen=True, slots=True)
class KernelForm:
    """How the kernels of one device are scheduled, rendered, built and called."""

    language: str
    suffix: str
    schedule: Callable[[Boundary], dace.SDFG]
    build: Callable[[Path, str, Optional[List[str]], Path], None]
    call: Callable[..., Tuple[Optional[Dict[str, np.ndarray]], float]]
    runtime: Callable[[str], List[str]]


FORMS: Dict[str, KernelForm] = {
    "cpu": KernelForm("c++", ".cpp", cpu_schedule, build_cpu_library, call_native, cpu_runtime_libraries),
    "gpu": KernelForm("cuda", ".cu", gpu_schedule, build_gpu_library, call_on_device, gpu_runtime_libraries),
}


def schedule_kernel(ext: ExternalCall, boundary: Boundary, out_dir: Path) -> KernelSource:
    """The kernel for the device phase 3 placed ``ext`` on, rendered by CPF into ``<out_dir>/<kernel>.cpp`` or
    ``.cu``, whose one entry is ``ext``'s symbol."""
    device = kernel_device(ext)
    form = FORMS[device]
    sdfg = form.schedule(boundary)
    sdfg.name = ext.name
    rendering = cpf.render(sdfg, language=form.language)
    out_dir.mkdir(parents=True, exist_ok=True)
    unit = out_dir / f"{ext.name}{form.suffix}"
    unit.write_text(rendering.code)
    return KernelSource(ext.name, unit, list(rendering.arguments), boundary, device)


def build_kernel_library(src: KernelSource, compiler: str, flags: Optional[List[str]], out_dir: Path) -> Path:
    """Build ``<out_dir>/lib<kernel>.a`` from the kernel's unit, plus its shared twin for validation."""
    archive = out_dir / f"lib{src.name}.a"
    FORMS[src.device].build(src.unit, compiler, flags, archive)
    return archive


def kernel_runtime_libraries(src: KernelSource, compiler: str) -> List[str]:
    """What a program linking ``src``'s library, built by ``compiler``, must link after its objects."""
    return FORMS[src.device].runtime(compiler)


def use_kernel_library(
    ext: ExternalCall, lib_path: Path, symbol: str, abi_order: List[str], runtime_libraries: Sequence[str]
) -> None:
    """Point ``ext`` at a built library and the runtimes it needs, and select the extern-call expansion."""
    ext.lib_path, ext.symbol, ext.abi_order = str(lib_path), symbol, list(abi_order)
    ext.runtime_libraries = list(runtime_libraries)
    ext.implementation = "ExternCall"


def measure_kernel(
    archive: Path,
    src: KernelSource,
    inputs: Dict[str, np.ndarray],
    oracle: Dict[str, np.ndarray],
    sizes: Dict[str, int],
    reps: int,
    fp_mode: str,
) -> KernelVerdict:
    """Call the twin's C entry in a forked child: one run compared to ``oracle``, then ``reps`` timed runs.
    A crash or timeout comes back as a verdict with ``error`` set."""
    argtypes = [p.ctype for p in parse_params(raw_signature(src.unit.read_text(), src.symbol))]
    shared = archive.with_suffix(".so")
    call = FORMS[src.device].call

    def work() -> Dict[str, float]:
        outs, us = call(shared, src.symbol, src.abi_order, argtypes, src.boundary, inputs, sizes, reps)
        assert outs is not None, "the kernel call snapshots outputs unless told not to"
        md, md_rel = diff_stats(oracle, outs)
        return {"maxdiff": md, "md_rel": md_rel, "dtype_floor": dtype_floor(outs), "time_us": us}

    res = run_isolated(work)
    if "error" in res:
        return failed_verdict(fp_mode, str(res["error"]))
    return KernelVerdict(
        fp_mode, float(res["maxdiff"]), float(res["md_rel"]), float(res["dtype_floor"]), float(res["time_us"])
    )


def validate_kernel(
    archive: Path,
    src: KernelSource,
    prep: Prepared,
    sizes: Dict[str, int],
    reps: int = 10,
    fp_mode: str = "strict-ieee",
) -> KernelVerdict:
    """:func:`measure_kernel` on seeded inputs against the kernel's NumPy oracle."""
    inputs = make_inputs(src.boundary, sizes)
    oracle = run_oracle(prep, src.boundary, inputs, sizes)
    return measure_kernel(archive, src, inputs, oracle, sizes, reps, fp_mode)
