# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Correctness and timing machinery shared by the build lanes: seeded inputs from a kernel's boundary, the
NumPy oracle, the FP-rung gate, and the bind-once / rewind-per-rep ctypes call."""

from __future__ import annotations

import ctypes
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

import dace
from dace import symbolic

from nestforge.build import flags
from nestforge.build.toolchain import needed_libraries
from nestforge.ir.emit_numpy import load_emitted, maxsize_loop_scratch, scratch_arrays
from nestforge.ir.extract import Boundary
from nestforge.corpus.translate import Prepared

#: Strict rung overridden to bit-exact: a same-order NumPy oracle reaches 0.0, unlike whole-program FP_ATOL.
ARENA_ATOL: Dict[str, float] = {**flags.FP_ATOL, "strict-ieee": 0.0}

# numpy dtype name -> ctypes scalar for the ABI (bool needed: DaCe lowers a comparison transient to C bool).
CTYPE = {
    "float64": ctypes.c_double,
    "float32": ctypes.c_float,
    "int64": ctypes.c_int64,
    "int32": ctypes.c_int32,
    "bool": ctypes.c_bool,
}


def resolve_shape(shape: Sequence[Any], sizes: Dict[str, int]) -> Tuple[int, ...]:
    env: Dict[Union[symbolic.symbol, str], Union[int, float]] = {symbolic.symbol(k): v for k, v in sizes.items()}
    return tuple(int(symbolic.evaluate(d, env)) for d in shape)


def emitted_sdfg(boundary: Boundary) -> dace.SDFG:
    """The descriptors the EMITTED kernel is written against (widened scratch included); sizing from the
    raw nest allocates a buffer too small and overflows the heap."""
    return maxsize_loop_scratch(boundary.standalone_sdfg, boundary.symbols)


def scratch_names(boundary: Boundary) -> List[str]:
    """Transient array buffers the C-style kernel expects the caller to pre-allocate."""
    return scratch_arrays(emitted_sdfg(boundary))


#: Upper bound of random inputs [0, INPUT_HIGH); must stay <= 1/4 so TSVC s232's squaring recurrence converges.
INPUT_HIGH = 0.25


def make_inputs(
    boundary: Boundary, sizes: Dict[str, int], seed: int = 0, given: Optional[Dict[str, np.ndarray]] = None
) -> Dict[str, np.ndarray]:
    """Random arrays for inputs; zeros for outputs and scratch buffers (all caller-pre-allocated).
    :param given: ready-made values (e.g. index arrays) that must match the resolved shape/dtype exactly."""
    sdfg = emitted_sdfg(boundary)  # widened scratch: allocate what the kernel indexes, not the raw shape
    rng = np.random.default_rng(seed)
    given = given or {}
    arrays: Dict[str, np.ndarray] = {}
    out_only = [o for o in boundary.outputs if o not in boundary.inputs]
    zero_filled = out_only + [s for s in scratch_arrays(sdfg) if s not in boundary.inputs]
    for name in list(boundary.inputs) + zero_filled:
        desc = sdfg.arrays[name]
        shape = resolve_shape(desc.shape, sizes)
        dt = np.dtype(desc.dtype.type)
        if name in given:
            value = given[name]
            if value.shape != shape or value.dtype != dt:
                raise ValueError(
                    f"given array {name!r} is {value.dtype}{value.shape}, but the nest declares "
                    f"{dt}{shape}; it is passed straight across the ABI, so it must match exactly"
                )
            arrays[name] = value.copy()
        else:
            arrays[name] = np.zeros(shape, dt) if name in zero_filled else (rng.random(shape) * INPUT_HIGH).astype(dt)
    return arrays


def run_oracle(
    prep: Prepared, boundary: Boundary, inputs: Dict[str, np.ndarray], sizes: Dict[str, int]
) -> Dict[str, np.ndarray]:
    """Run the emitted numpy kernel to get reference outputs."""
    missing = [s for s in boundary.symbols if s not in sizes]
    if missing:
        raise KeyError(
            f"no value for boundary symbol(s) {missing} (e.g. a loop index carried into an "
            f"extracted nest); pass them in `sizes`"
        )
    module = load_emitted(prep.numpy_source, prep.name)
    args = {k: v.copy() for k, v in inputs.items()}
    call = {**args, **{s: int(sizes[s]) for s in boundary.symbols}}
    vars(module)[prep.name](**call)
    return {o: args[o] for o in boundary.outputs}


def scalar_ctype(sdfg: dace.SDFG, name: str) -> type[ctypes._SimpleCData]:
    """ctype of a by-value kernel arg: float -> c_double; every integer is always int64_t regardless of the
    SDFG's own width, since a narrower c_int leaves the upper register half garbage."""
    if name in sdfg.symbols and np.dtype(sdfg.symbols[name].type).kind == "f":
        return ctypes.c_double
    return ctypes.c_int64


def accumulating_outputs(boundary: Boundary, buffers: Dict[str, np.ndarray]) -> List[str]:
    """Outputs the kernel both reads and writes; a timed rep loop restores these so an unrestored in-place
    kernel does not decay into denormals within a few reps and time subnormal arithmetic instead."""
    return [o for o in boundary.outputs if o in boundary.inputs and o in buffers]


def rewind_snapshot(boundary: Boundary, buffers: Dict[str, np.ndarray]) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Each accumulating buffer paired with a pristine copy, taken once before the warm call for :func:`rewind`."""
    return [(buffers[o], buffers[o].copy()) for o in accumulating_outputs(boundary, buffers)]


def rewind(snapshot: List[Tuple[np.ndarray, np.ndarray]]) -> None:
    """Restore the pristine contents of every accumulating buffer. Call OUTSIDE the timed region."""
    for buf, pristine in snapshot:
        buf[...] = pristine


#: ctypes' metaclass of every ``POINTER(T)``: tells a by-pointer parameter from a by-value one.
POINTER_TYPE = type(ctypes.POINTER(ctypes.c_double))


def bind_argument(arg: str, ctype: type, buffers: Dict[str, np.ndarray], sizes: Dict[str, int]) -> object:
    """One ctypes argument: a buffer by pointer, a Scalar's one-element buffer by value, else a size by value."""
    if arg not in buffers:
        return ctype(sizes[arg])
    if isinstance(ctype, POINTER_TYPE):
        return buffers[arg].ctypes.data_as(ctype)
    return ctype(buffers[arg].item())


def bind_arguments(
    order: List[str], argtypes: List[type], buffers: Dict[str, np.ndarray], sizes: Dict[str, int]
) -> List[object]:
    return [bind_argument(arg, ctype, buffers, sizes) for arg, ctype in zip(order, argtypes)]


def call_native(
    so: Path,
    symbol: str,
    order: List[str],
    argtypes: list,
    boundary: Boundary,
    inputs: Dict[str, np.ndarray],
    sizes: Dict[str, int],
    reps: int,
    copy_inputs: bool = True,
    copy_outputs: bool = True,
) -> Tuple[Optional[Dict[str, np.ndarray]], float]:
    """Bind + call the compiled entry, then time ``reps`` calls on the same buffers.
    ``order`` must be the EMITTED-signature order (not the manifest's), or same-typed buffers land in the
    wrong slot silently. A read-write output is restored before every timed rep, outside the timed region,
    since an unrestored in-place kernel decays into denormals within a few reps and times subnormal
    arithmetic instead. ``copy_inputs=False`` runs on the caller's own buffers; ``copy_outputs=False`` skips
    the result snapshot."""
    lib = ctypes.CDLL(str(so))
    fn = lib[symbol]  # ctypes CDLL indexing (not getattr) to bind the kernel symbol
    fn.argtypes = argtypes
    fn.restype = None
    work = {k: v.copy() for k, v in inputs.items()} if copy_inputs else inputs

    # bind ONCE (every rep reuses these buffers): per-rep data_as would time Python marshaling
    args = bind_arguments(order, argtypes, work, sizes)
    snapshot = rewind_snapshot(boundary, work)
    fn(*args)  # correctness run
    outputs = {o: work[o].copy() for o in boundary.outputs} if copy_outputs else None
    total = 0.0
    rewind(snapshot)  # the warm call primes the caches from the same state a timed rep sees
    fn(*args)
    for _ in range(reps):
        rewind(snapshot)
        t0 = time.perf_counter()
        fn(*args)
        total += time.perf_counter() - t0
    elapsed_us = total / reps * 1e6
    return outputs, elapsed_us


#: ``cudaMemcpyKind`` values.
HOST_TO_DEVICE = 1
DEVICE_TO_HOST = 2


def cuda_check(status: int, what: str) -> None:
    if status != 0:
        raise RuntimeError(f"{what} failed with CUDA status {status}")


def loaded_cudart(shared: Path) -> ctypes.CDLL:
    """The ``libcudart`` an already loaded kernel library links, bound without loading a second copy."""
    soname = next((name for name in needed_libraries(shared) if name.startswith("libcudart")), None)
    if soname is None:
        raise LookupError(f"{shared} does not link libcudart; a device kernel must name its runtime")
    return ctypes.CDLL(soname, mode=os.RTLD_NOLOAD)


@dataclass(slots=True)
class DeviceMemory:
    """Device copies of a kernel's pointer arguments, allocated and freed through one ``libcudart``."""

    cudart: ctypes.CDLL
    pointers: Dict[str, ctypes.c_void_p]

    def upload(self, name: str, host: np.ndarray) -> None:
        source = host.ctypes.data_as(ctypes.c_void_p)
        status = self.cudart.cudaMemcpy(self.pointers[name], source, ctypes.c_size_t(host.nbytes), HOST_TO_DEVICE)
        cuda_check(status, f"copy of {name} to the device")

    def download(self, name: str, host: np.ndarray) -> None:
        target = host.ctypes.data_as(ctypes.c_void_p)
        status = self.cudart.cudaMemcpy(target, self.pointers[name], ctypes.c_size_t(host.nbytes), DEVICE_TO_HOST)
        cuda_check(status, f"copy of {name} to the host")

    def free(self) -> None:
        for pointer in self.pointers.values():
            self.cudart.cudaFree(pointer)


def device_memory(cudart: ctypes.CDLL, buffers: Dict[str, np.ndarray], names: Sequence[str]) -> DeviceMemory:
    """A device buffer per name, holding the host contents."""
    memory = DeviceMemory(cudart, {})
    for name in names:
        pointer = ctypes.c_void_p()
        cuda_check(
            cudart.cudaMalloc(ctypes.byref(pointer), ctypes.c_size_t(buffers[name].nbytes)), f"allocation of {name}"
        )
        memory.pointers[name] = pointer
        memory.upload(name, buffers[name])
    return memory


def time_device_reps(
    fn: Any, args: list, memory: DeviceMemory, boundary: Boundary, host: Dict[str, np.ndarray], reps: int
) -> Tuple[Dict[str, np.ndarray], float]:
    """One correctness call and its outputs, a warm call, then ``reps`` timed calls; an accumulating output is
    uploaded again from its pristine host copy before every call, outside the timed region."""
    restore = accumulating_outputs(boundary, host)
    fn(*args)  # the CPF entry synchronizes before it returns
    outputs = {name: host[name].copy() for name in boundary.outputs if name in memory.pointers}
    for name, buffer in outputs.items():
        memory.download(name, buffer)
    total = 0.0
    for rep in range(reps + 1):
        for name in restore:
            memory.upload(name, host[name])
        t0 = time.perf_counter()
        fn(*args)
        total += (time.perf_counter() - t0) if rep > 0 else 0.0
    return outputs, total / reps * 1e6


def call_on_device(
    so: Path,
    symbol: str,
    order: List[str],
    argtypes: list,
    boundary: Boundary,
    inputs: Dict[str, np.ndarray],
    sizes: Dict[str, int],
    reps: int,
) -> Tuple[Optional[Dict[str, np.ndarray]], float]:
    """:func:`call_native` for a device kernel: every pointer argument is a device buffer, copied down once and
    read back once; scalars and sizes still go by value."""
    fn = ctypes.CDLL(str(so))[symbol]  # ctypes CDLL indexing (not getattr) to bind the kernel symbol
    fn.argtypes = argtypes
    fn.restype = None
    host = {k: v.copy() for k, v in inputs.items()}
    on_device = [arg for arg, ctype in zip(order, argtypes) if arg in host and isinstance(ctype, POINTER_TYPE)]
    memory = device_memory(loaded_cudart(so), host, on_device)
    args = [
        ctypes.cast(memory.pointers[arg], ctype) if arg in memory.pointers else bind_argument(arg, ctype, host, sizes)
        for arg, ctype in zip(order, argtypes)
    ]
    try:
        return time_device_reps(fn, args, memory, boundary, host, reps)
    finally:
        memory.free()


def dtype_floor(arrays: Dict[str, np.ndarray]) -> float:
    """The loosest :data:`flags.DTYPE_ATOL` floor among ``arrays`` (one ULP of the narrowest format present)."""
    return max(
        (flags.DTYPE_ATOL[v.dtype.name] for v in arrays.values() if v.dtype.name in flags.DTYPE_ATOL), default=0.0
    )


def rung_atol(mode: str, floor: float) -> float:
    """The relative gate at FP rung ``mode``, never tighter than the dtype ``floor`` the outputs allow."""
    return max(ARENA_ATOL[mode], floor)


def diff_stats(a: Dict[str, np.ndarray], b: Dict[str, np.ndarray]) -> Tuple[float, float]:
    """``(worst_abs, worst_scaled)`` in one pass over every array: the absolute elementwise difference, and
    the same difference scaled by the magnitude of the values compared (the denominator floors at 1.0, since
    an absolute gate is unreachable for a reduction -- fp64 ULP noise exceeds 1e-14 at reduction scale --
    without loosening what fp64 actually promises at small magnitude). ``inf`` on a non-finite difference
    (builtin ``max`` drops a non-first NaN, which would otherwise report a NaN-poisoned kernel as a perfect
    match) and on a verdict that touched zero elements (an all-empty comparison is not a match either)."""
    worst_abs, worst_rel = 0.0, 0.0
    compared = False
    for k in a:
        if not a[k].size:
            continue
        compared = True
        diff = np.abs(a[k] - b[k])
        d_abs = float(np.max(diff))
        if not np.isfinite(d_abs):
            return float("inf"), float("inf")
        scale = np.maximum(np.maximum(np.abs(a[k]), np.abs(b[k])), 1.0)
        with np.errstate(invalid="ignore"):  # inf/inf -> nan, which is a FAILURE, not a warning
            d_rel = float(np.max(diff / scale))
        if not np.isfinite(d_rel):
            return float("inf"), float("inf")
        worst_abs = max(worst_abs, d_abs)
        worst_rel = max(worst_rel, d_rel)
    if not compared:
        # a verdict read off zero elements must fail loudly, not report 0.0 as bit-exact
        return float("inf"), float("inf")
    return worst_abs, worst_rel


def maxdiff(a: Dict[str, np.ndarray], b: Dict[str, np.ndarray]) -> float:
    """The absolute half of :func:`diff_stats`."""
    return diff_stats(a, b)[0]


def relative_maxdiff(a: Dict[str, np.ndarray], b: Dict[str, np.ndarray]) -> float:
    """The scaled half of :func:`diff_stats`."""
    return diff_stats(a, b)[1]
