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
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import dace
from dace import symbolic

from nestforge.build.toolchain import ldconfig_output
from nestforge.build import flags
from nestforge.ir.emit_numpy import load_emitted, maxsize_loop_scratch, scratch_arrays
from nestforge.ir.extract import Boundary
from nestforge.corpus.translate import Prepared

#: Strict rung overridden to BIT-EXACT: a kernel and its NumPy oracle in the same op order reach 0.0, unlike
#: the whole-program oracle FP_ATOL is written for (pairwise sum vs tree). ``tests/test_variants_phase.py``
#: pins ``maxdiff == 0.0`` for a strict winner. Relax this only against a build that demonstrably cannot reach it.
ARENA_ATOL: Dict[str, float] = {**flags.FP_ATOL, "strict-ieee": 0.0}

# numpy dtype name -> ctypes scalar for the emitted kernel's ABI. ``bool`` is needed because a comparison
# materialises a boolean transient, which DaCe lowers to a 1-byte C ``bool``.
CTYPE = {
    "float64": ctypes.c_double,
    "float32": ctypes.c_float,
    "int64": ctypes.c_int64,
    "int32": ctypes.c_int32,
    "bool": ctypes.c_bool
}


# --- BLAS backends (a link axis for matmul-heavy kernels) -------------------------------------
@dataclass(slots=True)
class BlasBackend:
    """An installed BLAS implementation and the link flags that select it."""
    name: str
    link_flags: List[str]


#: BLAS backend -> candidate ``lib<soname>.so`` names, most specific first.
_BLAS_SONAMES = {
    "openblas": ["openblas"],
    "blis": ["blis"],
    "atlas": ["tatlas", "satlas"],
    "mkl": ["mkl_rt"],
    # generic ``libblas.so``: may be netlib OR an alternatives symlink, so labelled by soname only
    "blas": ["blas"],
}


def ldconfig_sonames() -> set:
    """Base library names (``openblas`` from ``libopenblas.so.0``) known to the dynamic linker."""
    out = ldconfig_output()
    names = set()
    for line in out.splitlines():
        token = line.strip().split(" ", 1)[0]
        if token.startswith("lib") and ".so" in token:
            names.add(token[3:token.index(".so")])
    return names


def discover_blas_libraries() -> Dict[str, BlasBackend]:
    """Discover installed BLAS backends (OpenBLAS / MKL / BLIS / ATLAS / netlib) + their link flags.

    An extra link axis for kernels whose emitted numpy uses ``@``/``np.dot``. Probes the dynamic linker
    cache plus ``MKLROOT``.
    """
    sonames = ldconfig_sonames()
    found: Dict[str, BlasBackend] = {}
    for name, candidates in _BLAS_SONAMES.items():
        for so in candidates:
            if so in sonames:
                found[name] = BlasBackend(name, [f"-l{so}"])
                break
    mklroot = os.environ.get("MKLROOT")
    if "mkl" not in found and mklroot:
        # oneAPI 2024+ put the libraries directly under lib/; older layouts use lib/intel64.
        for libdir in (Path(mklroot) / "lib", Path(mklroot) / "lib" / "intel64"):
            if (libdir / "libmkl_rt.so").exists():
                # -rpath paired with -L: an MKLROOT install is off the loader path, so without it the
                # linked .so needs LD_LIBRARY_PATH to run
                found["mkl"] = BlasBackend("mkl", [f"-L{libdir}", f"-Wl,-rpath,{libdir}", "-lmkl_rt"])
                break
    return found


# --- data generation from the manifest --------------------------------------------------------
def resolve_shape(shape: Sequence[Any], sizes: Dict[str, int]) -> Tuple[int, ...]:
    env = {symbolic.symbol(k): v for k, v in sizes.items()}
    return tuple(int(symbolic.evaluate(d, env)) for d in shape)


def emitted_sdfg(boundary: Boundary) -> dace.SDFG:
    """The descriptors the EMITTED kernel is written against, not the raw nest's.

    The emitter widens a loop-sized scratch transient (``maxsize_loop_scratch``) before rendering, so
    caller-side allocation must use the SAME widened descriptor -- sizing from
    ``boundary.standalone_sdfg`` gives a smaller buffer than the kernel writes: a heap overflow.
    """
    return maxsize_loop_scratch(boundary.standalone_sdfg, boundary.symbols)


def scratch_names(boundary: Boundary) -> List[str]:
    """Transient array buffers the C-style kernel expects the caller to pre-allocate."""
    return scratch_arrays(emitted_sdfg(boundary))


#: Upper bound of the random-input range ``[0, INPUT_HIGH)``. Must stay <= 1/4 so a squaring recurrence
#: ``x = x*x + b`` (TSVC s232) has an attracting fixed point instead of overflowing to inf (nan maxdiff ->
#: spurious validation failure); non-negative keeps ``sqrt``/``log`` kernels real.
INPUT_HIGH = 0.25


def make_inputs(boundary: Boundary,
                sizes: Dict[str, int],
                seed: int = 0,
                given: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, np.ndarray]:
    """Random arrays for inputs; zeros for outputs and scratch buffers (all caller-pre-allocated).

    Inputs are drawn from ``[0, INPUT_HIGH)`` -- see :data:`INPUT_HIGH` for why the range is conditioned.

    ``given`` supplies ready-made values a uniform float fill cannot express, chiefly the index arrays of
    :func:`nestforge.tsvc.index_fills`. It is checked against the resolved shape/dtype: it crosses the ABI
    as the kernel's own buffer, so a mismatch would corrupt memory instead of raising.
    """
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
                raise ValueError(f"given array {name!r} is {value.dtype}{value.shape}, but the nest declares "
                                 f"{dt}{shape}; it is passed straight across the ABI, so it must match exactly")
            arrays[name] = value.copy()
        else:
            arrays[name] = (np.zeros(shape, dt) if name in zero_filled else (rng.random(shape) * INPUT_HIGH).astype(dt))
    return arrays


def run_oracle(prep: Prepared, boundary: Boundary, inputs: Dict[str, np.ndarray],
               sizes: Dict[str, int]) -> Dict[str, np.ndarray]:
    """Run the emitted numpy kernel to get reference outputs."""
    missing = [s for s in boundary.symbols if s not in sizes]
    if missing:
        raise KeyError(f"no value for boundary symbol(s) {missing} (e.g. a loop index carried into an "
                       f"extracted nest); pass them in `sizes`")
    module = load_emitted(prep.numpy_source, prep.name)
    args = {k: v.copy() for k, v in inputs.items()}
    call = {**args, **{s: int(sizes[s]) for s in boundary.symbols}}
    vars(module)[prep.name](**call)
    return {o: args[o] for o in boundary.outputs}


# --- compile + call ---------------------------------------------------------------------------
def scalar_ctype(sdfg: dace.SDFG, name: str) -> type[ctypes._SimpleCData]:
    """ctypes type of a by-value (non-array) kernel arg, matching the translator's signature.

    A float value scalar is ``double`` -> ``c_double``. EVERY integer symbol is emitted ``int64_t`` by the
    translator regardless of the SDFG's own int width, so it must be ``c_int64`` here -- a 32-bit
    ``c_int`` leaves the upper half of the register garbage and blows the loop bound out of range."""
    if name in sdfg.symbols and np.dtype(sdfg.symbols[name].type).kind == "f":
        return ctypes.c_double
    return ctypes.c_int64


def accumulating_outputs(boundary: Boundary, buffers: Dict[str, np.ndarray]) -> List[str]:
    """Outputs the kernel both READS and WRITES -- the ones a timed rep loop must restore.

    Every timing path in the repo needs this same set, and getting it wrong is invisible: an in-place nest
    left un-restored feeds on its own output, so rep k computes ``a * b**k``, reaches denormals within a
    handful of reps, and the median times subnormal arithmetic instead of the kernel. ONE definition, so a
    caller cannot quietly disagree about which buffers decay.

    A fully-overwritten output is deliberately NOT in the set: nothing it holds survives into the next rep,
    so it cannot accumulate, and snapshotting it would double peak RSS at the profiling preset for nothing."""
    return [o for o in boundary.outputs if o in boundary.inputs and o in buffers]


def rewind_snapshot(boundary: Boundary, buffers: Dict[str, np.ndarray]) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Each accumulating buffer paired with a pristine copy of itself, ready for :func:`rewind`.

    Taken ONCE, before the warm call, so the copy is the state every rep starts from. Pairs hold the ARRAY,
    not its name, so :func:`rewind` costs no dict lookup inside the rep loop."""
    return [(buffers[o], buffers[o].copy()) for o in accumulating_outputs(boundary, buffers)]


def rewind(snapshot: List[Tuple[np.ndarray, np.ndarray]]) -> None:
    """Restore the pristine contents of every accumulating buffer. Call OUTSIDE the timed region."""
    for buf, pristine in snapshot:
        buf[...] = pristine


def call_native(so: Path,
                symbol: str,
                order: List[str],
                argtypes: list,
                boundary: Boundary,
                inputs: Dict[str, np.ndarray],
                sizes: Dict[str, int],
                reps: int,
                copy_inputs: bool = True,
                copy_outputs: bool = True) -> Tuple[Optional[Dict[str, np.ndarray]], float]:
    """Bind + call the compiled entry, then time ``reps`` calls on the same buffers.

    ``order`` is the EMITTED-signature parameter order; binding by the manifest's role order instead puts
    each buffer in the wrong parameter slot, which same-typed arrays make completely silent.

    An array that is both READ and WRITTEN is restored before every timed rep, OUTSIDE the timed region.
    Without it an in-place kernel (``a[:] = a[:] * b``) feeds on its own output: inputs are drawn from
    [0, 0.25), so by rep k the buffer holds ``a * b**k`` and reaches denormals within a handful of reps --
    the median then times subnormal arithmetic rather than the kernel, and the faster candidate is whichever
    decayed slower. Only the read-write intersection is snapshotted: a fully-overwritten output cannot
    accumulate, and at the profiling preset a blanket copy would double the child's peak RSS.

    :param copy_inputs: ``False`` runs on the CALLER's buffers, so a validating caller can read the results
        back out of them.
    :param copy_outputs: ``False`` skips the RESULT snapshot for a pure-timing caller (same RSS reason);
        the restore snapshot above is not optional, since it decides what the timing means.
    """
    lib = ctypes.CDLL(str(so))
    fn = lib[symbol]  # ctypes CDLL indexing (not getattr) to bind the kernel symbol
    fn.argtypes = argtypes
    fn.restype = None
    work = {k: v.copy() for k, v in inputs.items()} if copy_inputs else inputs

    def build_args() -> list:
        out = []
        for arg, at in zip(order, argtypes):
            if arg in work:
                out.append(work[arg].ctypes.data_as(at))
            else:
                out.append(at(sizes[arg]))  # at is the by-value ctype (c_int64 size / c_double value scalar)
        return out

    # bind ONCE (every rep reuses these buffers): per-rep data_as would time Python marshaling
    args = build_args()
    snapshot = rewind_snapshot(boundary, work)
    fn(*args)  # correctness run
    outputs = {o: work[o].copy() for o in boundary.outputs} if copy_outputs else None
    total = 0.0
    rewind(snapshot)  # the warm call primes the caches from the same state a timed rep sees
    fn(*args)  # warm
    for _ in range(reps):
        rewind(snapshot)
        t0 = time.perf_counter()
        fn(*args)
        total += time.perf_counter() - t0
    elapsed_us = total / reps * 1e6
    return outputs, elapsed_us


def maxdiff(a: Dict[str, np.ndarray], b: Dict[str, np.ndarray]) -> float:
    """Largest absolute elementwise difference; ``inf`` if any difference is non-finite.

    The non-finite mapping is load-bearing: builtin ``max`` DROPS a non-first NaN (``nan > x`` is False),
    so a NaN-poisoned kernel would report 0.0 and win the sweep.
    """
    worst = 0.0
    compared = False
    for k in a:
        if not a[k].size:
            continue
        compared = True
        d = float(np.max(np.abs(a[k] - b[k])))
        if not np.isfinite(d):
            return float("inf")
        worst = max(worst, d)
    return worst if compared else float("inf")  # a verdict read off zero elements is not a match


def dtype_floor(arrays: Dict[str, np.ndarray]) -> float:
    """The loosest :data:`flags.DTYPE_ATOL` floor among ``arrays`` -- one ULP of the narrowest format
    present. An unlisted dtype (integer, bool) contributes nothing: it is exact or it is wrong."""
    return max((flags.DTYPE_ATOL[v.dtype.name] for v in arrays.values() if v.dtype.name in flags.DTYPE_ATOL),
               default=0.0)


def rung_atol(mode: str, floor: float) -> float:
    """The relative gate at FP rung ``mode``, never tighter than the dtype ``floor`` the outputs allow."""
    return max(ARENA_ATOL[mode], floor)


def gate_atol(mode: str, outputs: Dict[str, np.ndarray]) -> float:
    """The relative gate for one cell: its FP rung, never tighter than what the output dtype can express."""
    return rung_atol(mode, dtype_floor(outputs))


def diff_stats(a: Dict[str, np.ndarray], b: Dict[str, np.ndarray]) -> Tuple[float, float]:
    """``(worst_abs, worst_scaled)`` in ONE pass over the abs-difference, instead of :func:`maxdiff` and
    :func:`relative_maxdiff` each recomputing ``np.abs(a[k] - b[k])`` separately. Same semantics as calling
    both (NaN/Inf still map to ``inf`` in both slots); use this wherever a caller needs both numbers."""
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
        # Every array was zero-size, so the loop body never ran and 0.0 would be returned as "bit-exact"
        # from a comparison that touched no element. Skipping an individual empty array is fine; a verdict
        # read off nothing is not, and the gate is <= atol, so it must fail loudly.
        return float("inf"), float("inf")
    return worst_abs, worst_rel


def relative_maxdiff(a: Dict[str, np.ndarray], b: Dict[str, np.ndarray]) -> float:
    """Largest elementwise difference SCALED by the magnitude of the values it is between.

    An absolute gate is unreachable for a reduction: summing 32000 order-1 elements lands near 1.6e4,
    where one fp64 ULP (1.8e-12) already exceeds the 1e-14 default, so a correct vectorized reduce is
    recorded WRONG and the kernel silently vanishes from the corpus. The denominator floors at 1.0, so
    small values keep the absolute reading -- the gate is never loosened below what fp64 promises, and a
    real miscompile (far more than a few ULP) is still caught. NaN/Inf still fail.
    """
    worst = 0.0
    compared = False
    for k in a:
        if not a[k].size:
            continue
        compared = True
        scale = np.maximum(np.maximum(np.abs(a[k]), np.abs(b[k])), 1.0)
        with np.errstate(invalid="ignore"):  # inf/inf -> nan, which is a FAILURE, not a warning
            d = float(np.max(np.abs(a[k] - b[k]) / scale))
        # builtin max(0.0, nan) is 0.0, i.e. a PERFECT match for a NaN-poisoned result -- map to inf
        if not np.isfinite(d):
            return float("inf")
        worst = max(worst, d)
    return worst if compared else float("inf")  # a verdict read off zero elements is not a match
