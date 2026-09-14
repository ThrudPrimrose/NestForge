# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The arena: compile an extracted nest across a compiler x FP-mode matrix, validate each
against the numpy oracle, time it, and pick the winner per FP mode.

Scope: CPU, C target, compilers discovered from PATH (gcc/clang), three FP modes. Timing is
external wall-clock over repeats.
"""
from __future__ import annotations

import ctypes
import functools
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import dace
from dace import symbolic

from nestforge.build.toolchain import COMPILE_TIMEOUT_S, compiler_family, ldconfig_output
from nestforge.build.dedup import collapse, representatives, variant_key
from nestforge.build import flags
from nestforge.ir.emit_numpy import load_emitted, maxsize_loop_scratch, scratch_arrays
from nestforge.ir.extract import Boundary
from nestforge.build.isolation import run_isolated
from nestforge.corpus.translate import Prepared

# --- FP modes (the flag axis) -----------------------------------------------------------------
#: Strict rung overridden to BIT-EXACT: arena compares emitted C vs emitted numpy, same op order, so 0.0
#: is reachable here -- unlike the whole-program oracle FP_ATOL is written for (pairwise sum vs tree).
#: This was briefly relaxed to 1e-15 on the theory that 0.0 was producing false negatives. It was not:
#: the false negatives came from dedup twins inheriting the representative's ok flag at the strictest
#: rung, fixed below by re-gating each twin, and ``tests/test_arena.py`` pins ``maxdiff == 0.0`` for the
#: strict winner. Relax this only against a build that demonstrably cannot reach it.
ARENA_ATOL: Dict[str, float] = {**flags.FP_ATOL, "strict-ieee": 0.0}

_CANDIDATE_COMPILERS = {"gcc": "gcc", "clang": "clang"}
# numpy dtype name -> ctypes scalar for the emitted kernel's ABI. ``bool`` is needed because a comparison
# materialises a boolean transient, which DaCe lowers to a 1-byte C ``bool``.
CTYPE = {
    "float64": ctypes.c_double,
    "float32": ctypes.c_float,
    "int64": ctypes.c_int64,
    "int32": ctypes.c_int32,
    "bool": ctypes.c_bool
}


def discover_compilers() -> Dict[str, str]:
    """Probe PATH for gcc/clang."""
    return {name: shutil.which(exe) for name, exe in _CANDIDATE_COMPILERS.items() if shutil.which(exe)}


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
@dataclass(slots=True)
class Cell:
    compiler: str
    fp_mode: str
    ok: bool
    maxdiff: float
    time_us: float
    compile_us: float = 0.0  # wall time of THIS candidate's compile (the post-optimization toolchain cost)
    so_path: Optional[str] = None
    symbol: Optional[str] = None
    #: The EMITTED signature's parameter order (harness.signature_order); use it verbatim. numpyto orders
    #: by param_order() (arrays sorted, then scalars), not the manifest's role order -- re-deriving it
    #: elsewhere silently swaps pointers.
    abi_order: Optional[List[str]] = None
    error: Optional[str] = None
    #: Cell whose measurement this one carries because both built the identical artifact (dedup.variant_key).
    #: Its timing is real, just not measured twice -- ``None`` means this cell was measured itself.
    same_as: Optional[str] = None


def scalar_ctype(sdfg: dace.SDFG, name: str) -> type[ctypes._SimpleCData]:
    """ctypes type of a by-value (non-array) kernel arg, matching the translator's signature.

    A float value scalar is ``double`` -> ``c_double``. EVERY integer symbol is emitted ``int64_t`` by the
    translator regardless of the SDFG's own int width, so it must be ``c_int64`` here -- a 32-bit
    ``c_int`` leaves the upper half of the register garbage and blows the loop bound out of range."""
    if name in sdfg.symbols and np.dtype(sdfg.symbols[name].type).kind == "f":
        return ctypes.c_double
    return ctypes.c_int64


def resolve_argtypes(order: List[str], boundary: Boundary) -> list:
    """ctypes argtypes for the emitted entry, in the order the EMITTED SIGNATURE declares.

    ``order`` must come from parsing the generated source (``harness.signature_order``), NOT the
    manifest's ``input_args``: numpyto emits ``param_order()`` (arrays sorted, then scalars), which
    coincides with the manifest's role order only by luck of the alphabet.
    """
    sdfg = boundary.standalone_sdfg
    types = []
    for arg in order:
        if arg in sdfg.arrays:
            dt = np.dtype(sdfg.arrays[arg].dtype.type).name
            types.append(ctypes.POINTER(CTYPE[dt]))
        else:
            types.append(scalar_ctype(sdfg, arg))
    return types


def accumulating_outputs(boundary: Boundary, buffers: Dict[str, np.ndarray]) -> List[str]:
    """Outputs the kernel both READS and WRITES -- the ones a timed rep loop must restore.

    Every timing path in the repo needs this same set, and getting it wrong is invisible: an in-place nest
    left un-restored feeds on its own output, so rep k computes ``a * b**k``, reaches denormals within a
    handful of reps, and the median times subnormal arithmetic instead of the kernel. ONE definition, so a
    caller cannot quietly disagree about which buffers decay -- four timing loops each rolled their own and
    three picked a different set.

    A fully-overwritten output is deliberately NOT in the set: nothing it holds survives into the next rep,
    so it cannot accumulate, and snapshotting it would double peak RSS at the profiling preset for nothing.

    Lives here rather than in ``perf.harness`` because the non-perf timing paths (:mod:`differential`,
    :mod:`whole_program`) need it too and already depend on this module."""
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


def validate_and_time(built, boundary: Boundary, inputs: Dict[str, np.ndarray], sizes: Dict[str, int],
                      oracle: Dict[str, np.ndarray], atol: float, reps: int) -> Dict[str, Any]:
    """Validate a built program against ``oracle``, then time it: the body BOTH whole-program lanes run
    inside their fork (:mod:`nestforge.whole_program`, :mod:`nestforge.differential`).

    Three invariants they were each carrying their own copy of, and which are the whole reason this is one
    function:

    * a boundary output MISSING from the run is a failure, not a smaller comparison. ``if o in vbuf`` would
      quietly narrow the comparison set, and a miscompile confined to that output would still read ok.
    * the ABSOLUTE difference is reported (that is what a reader of ``maxdiff`` expects) while the SCALED
      one gates -- an absolute atol is unreachable at reduction magnitudes (see :func:`relative_maxdiff`).
    * every read-write buffer is rewound before each rep, OUTSIDE the timed region: an in-place program
      otherwise feeds on its own output and the median times denormal arithmetic instead of the kernel.

    ``built`` is a :class:`nestforge.build.BuiltSDFG`; untyped here only to keep :mod:`nestforge.build.arena`
    free of the codegen import.
    """
    vbuf = {k: v.copy() for k, v in inputs.items()}
    built.run(vbuf, sizes)
    absent = [o for o in boundary.outputs if o not in vbuf]
    outs = {} if absent else {o: vbuf[o] for o in boundary.outputs}
    if absent:
        verdict = {"ok": False, "maxdiff": float("inf"), "error": f"outputs absent from the run: {sorted(absent)}"}
    elif outs:
        md, md_rel = diff_stats({o: oracle[o] for o in outs}, outs)
        verdict = {"ok": bool(md_rel <= atol), "maxdiff": float(md)}
    else:
        verdict = {"ok": False, "maxdiff": float("inf")}
    # init once, bind once, call the bare kernel in the rep loop (no per-rep marshaling)
    tbuf = {k: v.copy() for k, v in inputs.items()}
    built.init(sizes)
    try:
        fn, cargs = built.bind_program(tbuf, sizes)
        snapshot = rewind_snapshot(boundary, tbuf)
        rewind(snapshot)
        fn(*cargs)  # warm
        samples: List[float] = []
        for _ in range(reps):
            rewind(snapshot)
            t0 = time.perf_counter()
            fn(*cargs)
            samples.append((time.perf_counter() - t0) * 1e6)
    finally:
        built.close()
    return {**verdict, "median_us": float(np.median(samples))}


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

    ``order`` is the EMITTED-signature parameter order (see :func:`resolve_argtypes`); binding by the
    manifest's role order instead puts each buffer in the wrong parameter slot, which same-typed arrays
    make completely silent.

    An array that is both READ and WRITTEN is restored before every timed rep, OUTSIDE the timed region.
    Without it an in-place kernel (``a[:] = a[:] * b``) feeds on its own output: TSVC inputs are drawn from
    [0, 0.25), so by rep k the buffer holds ``a * b**k`` and reaches denormals within a handful of reps --
    the median then times subnormal arithmetic rather than the kernel, and the faster candidate is whichever
    decayed slower. Only the read-write intersection is snapshotted: a fully-overwritten output cannot
    accumulate, and at the profiling preset a blanket copy would double the child's peak RSS.

    :param copy_inputs: ``False`` runs on the CALLER's buffers, so a validating caller can read the results
        back out of them (what :func:`nestforge.perf.harness.call_c` needs).
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
    so a NaN-poisoned kernel would report 0.0 and win the arena.
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


def gate_atol(mode: str, outputs: Dict[str, np.ndarray]) -> float:
    """The relative gate for one cell: its FP rung, never tighter than what the output dtype can express."""
    return max(ARENA_ATOL[mode], dtype_floor(outputs))


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


@dataclass(slots=True)
class ArenaResult:
    name: str
    cells: List[Cell] = field(default_factory=list)
    winners: Dict[str, Cell] = field(default_factory=dict)  # fp_mode -> best correct cell
    #: wall time of the whole sweep (all candidates: compile + validate + time) -- the search cost.
    optimization_seconds: float = 0.0
    #: ``rep == dup, dup`` per collapsed group. Reported because a silent collapse reads exactly like a
    #: sweep that measured everything.
    collapsed: List[str] = field(default_factory=list)


def run_arena(prep: Prepared,
              boundary: Boundary,
              c_source: Path,
              out_dir: Path,
              sizes: Dict[str, int],
              reps: int = 100,
              seed: int = 0,
              given: Optional[Dict[str, np.ndarray]] = None) -> ArenaResult:
    """Sweep discovered compilers x FP modes; validate + time each; pick a winner per FP mode.

    Every cell is BUILT, then cells that produced the identical artifact are measured once and the rest
    carry that result with ``Cell.same_as`` set (:func:`dedup.variant_key`) -- a pure add compiles to one
    object for all four fp rungs, and timing it four times measures nothing new. What collapsed is on
    ``ArenaResult.collapsed``.

    ``given`` is forwarded to :func:`make_inputs`; this layer is corpus-agnostic, so a caller measuring a
    corpus kernel must pass ``tsvc.index_fills(...)`` -- without it an integer index array fills to
    all-zeros and the sweep times a degenerate gather while validating vacuously."""
    out_dir.mkdir(parents=True, exist_ok=True)
    compilers = discover_compilers()
    symbol = f"{prep.name}_fp64"
    # bind order comes from the REAL emitted signature, never the manifest (see resolve_argtypes).
    # Imported here because harness imports arena: a top-level import would cycle.
    from nestforge.build.harness import signature_order
    order = signature_order(c_source.read_text(), symbol)
    argtypes = resolve_argtypes(order, boundary)
    inputs = make_inputs(boundary, sizes, seed=seed, given=given)
    oracle = run_oracle(prep, boundary, inputs, sizes)

    def measure(so: Path, mode: str) -> Dict[str, Any]:
        outs, us = call_native(so, symbol, order, argtypes, boundary, inputs, sizes, reps)
        # report the ABSOLUTE difference, gate on the scaled one (see relative_maxdiff); one pass over
        # the diff computes both instead of two.
        md, md_rel = diff_stats(oracle, outs)
        # md_rel and the dtype floor cross the pipe too: every TWIN re-gates on them at its OWN rung.
        return {
            "ok": bool(md_rel <= gate_atol(mode, outs)),
            "maxdiff": float(md),
            "md_rel": float(md_rel),
            "time_us": float(us),
            "dtype_floor": dtype_floor(outs),
        }

    result = ArenaResult(name=prep.name)
    t_sweep = time.perf_counter()
    built: Dict[str, Tuple[Cell, Path]] = {}
    for cname, cpath in compilers.items():
        family = compiler_family(cpath)
        for mode in flags.FP_LEVELS:
            # family-aware: a local table would hand -ffast-math to nvc, and be a second fp vocabulary
            composed, reason = flags.lane_flags(family,
                                                mode,
                                                "default",
                                                parallel="none",
                                                lang="c",
                                                nthreads=1,
                                                compiler=cpath)
            if composed is None:
                result.cells.append(Cell(cname, mode, False, float("inf"), float("inf"), error=reason))
                continue
            so = out_dir / f"lib{prep.name}_{cname}_{mode}.so"
            cmd = [cpath, *composed, str(c_source), "-o", str(so)]
            t_c = time.perf_counter()
            comp = subprocess.run(cmd, capture_output=True, text=True)
            compile_us = (time.perf_counter() - t_c) * 1e6
            if comp.returncode != 0:
                result.cells.append(
                    Cell(cname,
                         mode,
                         False,
                         float("inf"),
                         float("inf"),
                         compile_us=compile_us,
                         error=comp.stderr[-400:]))
                continue
            cell = Cell(cname,
                        mode,
                        False,
                        float("inf"),
                        float("inf"),
                        compile_us=compile_us,
                        so_path=str(so),
                        symbol=symbol,
                        abi_order=list(order))
            result.cells.append(cell)
            built[f"{cname}:{mode}"] = (cell, so)

    # Group by the ARTIFACT, not by the axes that asked for it: cells whose code AND link are identical
    # are one measurement, and measuring costs reps x the kernel against one objdump to find out. An
    # artifact that cannot be keyed gets a unique id, so failing to inspect means measuring it.
    keys = {cid: (variant_key(so, symbol) or f"unkeyed:{cid}") for cid, (_, so) in built.items()}
    for members in collapse(keys).values():
        cell, so = built[members[0]]
        # Forked so a segfault/runaway kills only the child; only the summary crosses the pipe.
        res = run_isolated(functools.partial(measure, so, cell.fp_mode))
        if "error" in res:
            cell.error = res["error"]
        else:
            cell.ok, cell.maxdiff, cell.time_us = bool(res["ok"]), float(res["maxdiff"]), float(res["time_us"])
        for twin_id in members[1:]:
            twin = built[twin_id][0]
            twin.maxdiff, twin.time_us = cell.maxdiff, cell.time_us
            # Re-gate at the TWIN's own rung. FP_LEVELS is strictest-first and collapse keeps the first
            # member, so the representative is always the strictest cell: inheriting its ``ok`` failed
            # every looser twin of a build that is correct AT THAT TWIN'S TOLERANCE, and the report then
            # said "no correct build" for a rung that passed.
            twin.ok = (cell.ok if "error" in res else bool(
                res["md_rel"] <= max(ARENA_ATOL[twin.fp_mode], res["dtype_floor"])))
            twin.error, twin.same_as = cell.error, members[0]
    result.collapsed = representatives(keys)[1]

    result.optimization_seconds = time.perf_counter() - t_sweep
    for mode in flags.FP_LEVELS:
        correct = [c for c in result.cells if c.fp_mode == mode and c.ok]
        if correct:
            result.winners[mode] = min(correct, key=lambda c: c.time_us)
    return result


def run_tool(cmd: List[str], what: str) -> None:
    """Run one toolchain command with a deadline and CAPTURED stderr, so a failure raises with the compiler's
    actual diagnostic instead of dumping it to the console and raising a bare CalledProcessError."""
    try:
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=COMPILE_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"{what} exceeded {COMPILE_TIMEOUT_S}s: {' '.join(cmd)}") from None
    if done.returncode != 0:
        raise RuntimeError(f"{what} failed ({done.returncode}): {' '.join(cmd)}\n{done.stderr[-2000:]}")


def compile_object(cpath: str,
                   fp_mode: str,
                   c_source: Path,
                   name: str,
                   out_dir: Path,
                   veclib: Optional[str] = None,
                   cost_model: str = "default",
                   lang: str = "c") -> Path:
    """Compile one emitted source to a ``.o`` at ``(compiler, fp-mode, veclib, cost-model)``.

    ``fp_mode`` is a :data:`flags.FP_LEVELS` rung -- the ONE fp vocabulary; a second one here is what
    silently killed every E1 cell and :func:`build_winner_archive` call once the flags moved."""
    out_dir.mkdir(parents=True, exist_ok=True)
    obj = out_dir / f"{name}_nest.o"
    family = compiler_family(cpath)
    composed, reason = flags.lane_flags(family,
                                        fp_mode,
                                        cost_model,
                                        parallel="none",
                                        lang=lang,
                                        nthreads=1,
                                        compiler=cpath,
                                        veclib=veclib)
    if composed is None:
        raise ValueError(f"cannot compile {name} with {Path(cpath).name} at fp={fp_mode!r} "
                         f"veclib={veclib!r} cost={cost_model!r}: {reason}")
    cflags = [f for f in composed if f != "-shared"]  # link-only, and this step is -c (mirrors build.compile)
    run_tool([cpath, *cflags, "-fPIC", "-c", str(c_source), "-o", str(obj)], f"compiling {name} with {cpath}")
    return obj


def archive_objects(objs: List[Path], name: str, out_dir: Path) -> Path:
    """Bundle objects into ``lib<name>_nest.a`` (single-object offload today: one winning nest per archive).

    WARNING: never put several nests' objects in one archive -- DaCe SORTS the parent's link flags and can
    place the archive before the parent objects, so ld pulls no member and later references stay
    unresolved (``undefined symbol`` at ``dlopen``). The sort also scrambles the
    ``--whole-archive``/``--no-whole-archive`` pair. Use :func:`link_shared` for a multi-nest swap."""
    out_dir.mkdir(parents=True, exist_ok=True)
    archive = out_dir / f"lib{name}_nest.a"
    if archive.exists():
        archive.unlink()  # ar r APPENDS; start clean so a rebuild doesn't stack stale members
    run_tool(["ar", "rcs", str(archive), *[str(o) for o in objs]], f"archiving {name}")
    return archive


def link_shared(objs: List[Path], name: str, out_dir: Path, cpath: str, veclib: Optional[str] = None) -> Path:
    """Link objects into ``lib<name>_nest.so``. Resolved at ``dlopen`` time, hence order-independent, so it
    survives dace SORTING the parent's link flags (which leaves a static archive's members un-pulled). The
    parent links it with an rpath (see :meth:`ExternLibEnv.configure`).

    ``veclib`` MUST match what :func:`compile_object` used. A shared object is allowed to carry undefined
    symbols, so omitting it links cleanly and then dies at ``dlopen`` with ``undefined symbol: _ZGV...``
    -- the failure surfaces a whole phase away from its cause.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    so = out_dir / f"lib{name}_nest.so"
    vec_l, reason = flags.veclib_link_flags(cpath, veclib)
    if vec_l is None:
        raise ValueError(f"cannot link {name} with {Path(cpath).name} veclib={veclib!r}: {reason}")
    run_tool([cpath, "-shared", "-fPIC", *[str(o) for o in objs], *vec_l, "-o", str(so)], f"linking lib{name}_nest.so")
    return so


def build_winner_archive(win: Cell, c_source: Path, name: str, out_dir: Path) -> Path:
    """Materialize a winning cell as a static ``lib<name>_nest.a`` for STATIC offload into a parent SDFG.

    Recompiles the SAME source with the winner's ``(compiler, fp-mode)`` to an object and archives it. An
    archive carries objects only, no linked runtime, so the parent supplies the single libomp instead of
    every nest ``.so`` dragging its own."""
    obj = compile_object(discover_compilers()[win.compiler], win.fp_mode, c_source, name, out_dir)
    return archive_objects([obj], name, out_dir)
