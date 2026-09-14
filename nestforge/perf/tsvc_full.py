# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""TSVC full-matrix job: every kernel of tsvc2 + tsvc2_5, three lanes, MEDIAN-of-N timing at a
memory-bound profiling size.

Lanes
-----
1. **native baseline** -- the ``<key>_reference.cpp`` scalar reference at ``-O3 -march=native``:
   how well the compiler auto-vectorizes the reference.
2. **DaCe-cpp baseline** -- DaCe's own C++ codegen of the EXTRACTED-NEST standalone SDFG (owned
   direct-compile via ``build.build_sdfg``, no cmake), strict-ieee FP. Fanned over codegen impl
   (``experimental`` = default/denominator, ``legacy`` where available). This is the speedup
   denominator; it runs the standalone nest, not the whole kernel, so it stays apples-to-apples even
   for kernels peeled to an inner nest (loop-carried recurrences are flagged non-bit-exact in tables).
3. **nest-forge external-nest** -- the extracted nest translated by numpyto and compiled, swept over
   the axis matrix below.

Axis matrix (lane 3)
---------------------
  * **opt-mode**: ``simplify-parallel`` | ``canonicalize`` | ``auto-opt`` (changes emitted source).
  * **language**: ``c`` | ``c++`` | ``fortran``. numpyto has no C++ target, so C++ recompiles the
    emitted C via ``-x c++`` + a shim (:func:`nestforge.perf.flags.cxx_source_flags`).
  * **parallelization**: ``sequential`` | ``auto-par`` (gcc/nvc/icx auto-parallelizers; clang/flang
    have none -> unsupported).
  * **compiler**: every discovered family (gcc / clang / nvhpc / intel).
  * **cost-model**: ``default`` | ``cheap`` | ``no-vec``.
  * **FP mode**: ``default-fp`` | ``no-fast-errno`` for timing, plus a ``strict-ieee`` bit-exact gate
    cell (see :data:`flags.REDUCED_FP_MODES`).

Sizing
------
  * validate at ``M`` -- the .so is size-agnostic but the pure-Python oracle is slow at scale.
  * time at ``PROF`` (one fp64 array clears GH200 L3, ~114MB/socket); ``--profile-preset XL`` for the
    big confirmation run.

Timing
------
  * median of N reps (+ min/p25/p75/mean). Each cell compiles once; identical flag sets dedupe to one
    compile through a bounded ``--compile-jobs`` pool. A fast VALIDATE pass gates the TIMING pass.

Every kernel execution runs in a forked child (:func:`nestforge.isolation.run_isolated`) so a
segfault/OOM kills only the child. Kernels self-partition across ranks (SLURM/MPI) via
:func:`rank_and_size` + :func:`my_slice`. ``--tables-only`` merges per-kernel JSON into markdown.

Usage::

    python -m nestforge.perf.tsvc_full --corpora tsvc2 tsvc2_5 --profile-preset PROF \\
        --languages c c++ fortran --parallelism both --reps 11 --compilers auto
    python -m nestforge.perf.tsvc_full --tables-only --out perf_results/tsvc_full
"""
from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import shutil
import socket
import statistics
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import dace  # noqa: F401 -- ensure the real DaCe package is importable (not a cwd stub)

from nestforge import tsvc
from nestforge.device_profile import rank_veclibs
from nestforge import vectorize_variants as vv
from nestforge.arena import maxdiff, make_inputs, relative_maxdiff, rewind, rewind_snapshot, run_oracle
from nestforge.build import (BuildOptions, codegen_impls_available, compile_program, default_codegen_impl,
                             generate_program)
from nestforge.build import build_sdfg as dace_build_sdfg
from nestforge.dataset import WinnerRecord, nest_features, record_winner, store_nest
from nestforge.dedup import cpp_body_key, variant_key
from nestforge.extract import extract_nest_to_sdfg
from nestforge.isolation import run_isolated
from nestforge.perf import flags, pluto_lane, support_matrix
from nestforge.toolchain import Toolchain, discover_toolchains
from nestforge.perf.harness import (COMPILE_TIMEOUT_S, RUN_TIMEOUT_S, c_argtypes, c_call_args, call_c, finite, fmt_us,
                                    fortran_unmunge, geomean, jsonable, lang_compilers, load_results, my_slice,
                                    native_setup, native_symbol, rank_and_size, run_compile, signature_order)
from nestforge.strategies import empty_strategy_reason, get_strategy, is_parallel_nest
from nestforge.translate import emit_sources, prepare

#: numpyto emit target + suffix per language; C++ just recompiles the C target with a C++ frontend.
_EMIT = {"c": ("c", ".c"), "c++": ("c", ".c"), "fortran": ("fortran", ".f90")}
#: presets too large for the O(N) pure-Python oracle -> validate at ``M`` instead.
_VALIDATE_CAP = "M"


def validate_cap(profile_preset: str) -> str:
    """Preset to VALIDATE at: the timing preset if small (S/M), else M. Safe because the .so is
    size-agnostic, and necessary because the pure-Python oracle takes minutes at L/PROF/XL."""
    return profile_preset if profile_preset in ("S", _VALIDATE_CAP) else _VALIDATE_CAP


def physical_cores(cpus) -> int:
    """Physical cores the logical CPU ids ``cpus`` cover, collapsing SMT siblings: sizing an auto-par
    team by logical CPUs oversubscribes ~2x. An unreadable sibling list counts as its own core."""
    groups = set()
    for c in cpus:
        try:
            groups.add(Path(f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list").read_text().strip())
        except OSError:
            groups.add(str(c))
    return len(groups)


def default_threads() -> int:
    """Default auto-par thread count: physical cores this process may actually use, NOT
    ``os.cpu_count()``, which reports the whole node (a 288-CPU node's 4-rank job oversubscribes 4x).

    Priority: ``OMP_NUM_THREADS`` (nested list "72,8" -> first level is the team size) ->
    ``SLURM_CPUS_PER_TASK``/``_ON_NODE`` -> affinity mask collapsed to physical cores."""
    env = (os.environ.get("OMP_NUM_THREADS") or "").strip()
    if env:
        try:
            return max(1, int(env.split(",")[0]))
        except ValueError:
            pass  # unparseable -> fall through to the machine
    for var in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
        val = (os.environ.get(var) or "").strip()
        if val:
            try:
                return max(1, int(val))
            except ValueError:
                pass
    try:
        cpus = os.sched_getaffinity(0)  # honours cgroups/taskset/srun --cpu-bind; cpu_count() does not
    except (AttributeError, OSError):
        cpus = set(range(os.cpu_count() or 4))
    return max(1, physical_cores(cpus))


# --- median-of-N timing -----------------------------------------------------------------------------
def summarize_times(samples: List[float]) -> Dict[str, float]:
    """Robust summary of per-rep microsecond samples: median (the headline), min, linear-interpolated
    p25/p75 and mean. Empty input -> all ``inf``."""
    if not samples:
        return {
            "median_us": float("inf"),
            "min_us": float("inf"),
            "p25_us": float("inf"),
            "p75_us": float("inf"),
            "mean_us": float("inf")
        }
    s = sorted(samples)
    n = len(s)

    def percentile(p: float) -> float:
        if n == 1:
            return s[0]
        idx = p * (n - 1)
        lo = math.floor(idx)
        hi = math.ceil(idx)
        return s[lo] + (s[hi] - s[lo]) * (idx - lo)

    return {
        "median_us": float(statistics.median(s)),
        "min_us": float(s[0]),
        "p25_us": percentile(0.25),
        "p75_us": percentile(0.75),
        "mean_us": float(sum(s) / n),
    }


def collect_samples(fn, cargs, reps: int, snapshot=()) -> List[float]:
    """Warm once, then time ``reps`` individual calls on the reused args: ONE sample per rep, not one
    mean over the whole loop.

    :param snapshot: :func:`arena.rewind_snapshot` pairs, restored before the warm call and before every
        rep, outside the timed region. Empty only when the nest writes nothing it also reads. Skipping it
        does not merely add noise -- an in-place nest feeds on its own output, so rep k times ``a * b**k``,
        which reaches denormals within a handful of reps; the median then reports subnormal arithmetic and
        the winner is whichever candidate decayed slower. The reps are also no longer samples of one
        quantity, so the median of a decaying series is not an estimate of anything."""
    rewind(snapshot)
    fn(*cargs)  # warm
    samples: List[float] = []
    for _ in range(reps):
        rewind(snapshot)
        t0 = time.perf_counter()
        fn(*cargs)
        samples.append((time.perf_counter() - t0) * 1e6)
    return samples


# --- lane 3: nest cell validate / time (run inside a forked child) -----------------------------------
def nest_validate_work(so: Path,
                       symbol: str,
                       order: List[str],
                       argtypes,
                       boundary,
                       validate_sizes,
                       oracle,
                       atol: float,
                       given=None) -> Dict:
    """Correctness at the SMALL preset: bind, run once, maxdiff vs the oracle. Fast (small buffers)."""
    vin = make_inputs(boundary, validate_sizes, seed=0, given=given)
    vout, _ = call_c(so, symbol, order, argtypes, boundary, vin, validate_sizes, reps=1)
    md = float(maxdiff(oracle, vout))  # absolute is REPORTED, the scaled one is the gate
    return {"ok": bool(relative_maxdiff(oracle, vout) <= atol), "maxdiff": md}


def nest_timing_work(so: Path, symbol: str, order: List[str], argtypes, boundary, time_inputs, time_sizes,
                     reps: int) -> Dict:
    """Median-of-N timing at the PROFILING preset. ``time_inputs`` is allocated once per opt-mode and
    reused by every timing cell; the fork gets its own COW copy, so timing in place is safe.

    COW protects the PARENT across forks; it does nothing for a buffer across reps INSIDE one fork, which
    is why the rep loop still rewinds every read-write output."""
    fn = ctypes.CDLL(str(so))[symbol]
    fn.argtypes, fn.restype = argtypes, None
    cargs = c_call_args(order, argtypes, time_inputs, time_sizes)
    return summarize_times(collect_samples(fn, cargs, reps, rewind_snapshot(boundary, time_inputs)))


def native_validate_work(so, symbol, sig, kernel, boundary, validate_sizes, oracle, given=None) -> Dict:
    buffers = make_inputs(boundary, validate_sizes, seed=0, given=given)  # fresh + correct (validation runs in place)
    fn, cargs, ptr_names = native_setup(so, symbol, sig, kernel, buffers, validate_sizes)
    fn(*cargs)
    outs = {o: buffers[o] for o in boundary.outputs if o in ptr_names}
    if not outs:  # nothing to compare -> UNCHECKED; never report ok for an unvalidatable lane
        return {"ok": False, "maxdiff": float("inf"), "unchecked": True}
    ref = {k: oracle[k] for k in outs}
    md = float(maxdiff(ref, outs))  # absolute is REPORTED, the scaled one is the gate (as every other lane)
    return {"ok": bool(relative_maxdiff(ref, outs) <= flags.NATIVE_ATOL), "maxdiff": md}


def native_timing_work(so, symbol, sig, kernel, boundary, time_inputs, time_sizes, reps) -> Dict:
    fn, cargs, _ptr = native_setup(so, symbol, sig, kernel, time_inputs, time_sizes)  # COW copy in this fork
    # Same rewind as the nest lane: the native column is the speedup DENOMINATOR, so letting only one of
    # the two decay would bias every ratio in the table by an unknown factor.
    return summarize_times(collect_samples(fn, cargs, reps, rewind_snapshot(boundary, time_inputs)))


def measure_native_lane(cxx: str,
                        family: str,
                        kernel,
                        boundary,
                        validate_sizes,
                        time_inputs,
                        time_sizes,
                        oracle,
                        reps: int,
                        cxx_std: str,
                        workdir: Path,
                        validate_fills=None) -> Optional[Dict]:
    """Lane 1: compile ``_reference.cpp`` at ``-O3 -march=native``, validate@small + time@prof (median).
    ``None`` when the kernel ships no native source or the family has no C++ compiler."""
    cpp = kernel.native_cpp
    if cpp is None or cxx is None:
        return None
    nat_flags = flags.base_flags(family) + ["-std=" + cxx_std]
    text = cpp.read_text()
    try:
        symbol = native_symbol(text, kernel.native_symbol)
        sig = tsvc.native_signature(text, symbol)
    # ValueError too: a type the arena cannot bind is a NATIVE-COLUMN problem. Caught as LookupError
    # only, it escaped and took the whole kernel down as a crash rather than one skipped column.
    except (LookupError, ValueError) as e:
        return {"ok": False, "error": str(e), **summarize_times([])}
    so = workdir / f"{kernel.key}_{family}_native.so"
    ok, compile_us, err = run_compile([cxx, *nat_flags, str(cpp), "-o", str(so)])
    if not ok:
        return {"ok": False, "error": err, "compile_us": compile_us, **summarize_times([])}
    vres = run_isolated(
        lambda: native_validate_work(so, symbol, sig, kernel, boundary, validate_sizes, oracle, validate_fills))
    if "error" in vres:
        return {"ok": False, "error": vres["error"], "compile_us": compile_us, **summarize_times([])}
    tres = run_isolated(lambda: native_timing_work(so, symbol, sig, kernel, boundary, time_inputs, time_sizes, reps),
                        timeout=RUN_TIMEOUT_S)
    stats = summarize_times([]) if "error" in tres else tres
    return {
        "compiler": family,
        "ok": vres["ok"],
        "maxdiff": vres["maxdiff"],
        "unchecked": vres.get("unchecked", False),
        "compile_us": compile_us,
        "error": tres.get("error"),
        **stats
    }


# --- lane 2: DaCe-cpp baseline ----------------------------------------------------------------------
# Codegen of the EXTRACTED NEST, not the whole-kernel SDFG: a kernel peeled to an inner nest (s1115)
# would do ~LEN more work as a whole and inflate the speedup. Median time is reported even when
# validation fails: PROMOTED loop-carried state (s111/s112) is not bit-exact vs numpyto but still
# representative to time.
def dace_run_work(built,
                  boundary,
                  validate_sizes,
                  time_inputs,
                  time_sizes,
                  oracle,
                  atol: float,
                  reps: int,
                  given=None) -> Dict:
    """Validate@small then time@prof DaCe's own codegen, in the forked child. ``built``/``time_inputs``
    are shared COW, so a large time-size run OOM-kills only the child."""
    vbuf = make_inputs(boundary, validate_sizes, seed=0, given=given)  # fresh buffer (validation runs in place)
    built.run(vbuf, validate_sizes)  # init -> program -> close
    outs = {o: vbuf[o] for o in boundary.outputs if o in vbuf}
    if outs:
        ref = {o: oracle[o] for o in outs}
        md = float(maxdiff(ref, outs))  # absolute is REPORTED, scaled is the gate
        verdict = {"ok": bool(relative_maxdiff(ref, outs) <= atol), "maxdiff": md}
    else:  # nothing to compare -> UNCHECKED; never report ok for an unvalidatable lane
        verdict = {"ok": False, "maxdiff": float("inf"), "unchecked": True}
    tbuf = time_inputs
    built.init(time_sizes)
    try:
        # bind once, time the bare call: matches native/nest timing, no per-rep marshaling
        fn, cargs = built.bind_program(tbuf, time_sizes)
        samples = collect_samples(fn, cargs, reps, rewind_snapshot(boundary, tbuf))
    finally:
        built.close()
    return {**verdict, **summarize_times(samples)}


def measure_dace_cpp_lane(tc: Toolchain,
                          boundary,
                          validate_sizes,
                          time_inputs,
                          time_sizes,
                          oracle,
                          reps: int,
                          cxx_std: str,
                          workdir: Path,
                          codegen_impl: Optional[str] = None,
                          validate_fills=None) -> Dict:
    """Lane 2: DaCe's own C++ codegen of the extracted-nest standalone SDFG, ``-O3`` + strict-ieee.
    Builds into a per-kernel ``workdir``, so concurrent ranks never share a build dir.

    :param codegen_impl: ``experimental`` (default, the speedup denominator) or ``legacy``; ``None``
        takes :func:`build.default_codegen_impl`. Stamped into the result for the reporter to group on."""
    codegen_impl = codegen_impl or default_codegen_impl()
    if tc.cxx is None:
        return {
            "ok": False,
            "error": f"{tc.name}: no C++ compiler for the DaCe-cpp lane",
            "codegen_impl": codegen_impl,
            **summarize_times([])
        }
    fam = tc.fp_family
    dace_flags = flags.base_flags(fam) + ["-std=" + cxx_std] + flags.fp_flags(fam, "strict-ieee", "c")
    try:
        # use_ccache=False and the BUILD's own split clocks: a wall-clock stopwatch around build_sdfg
        # reported codegen+compile under a field every other lane fills with a raw compile, and ccache
        # defaults to AUTO, so the second run of a kernel reported a ~0s cache hit as its compile time.
        built = dace_build_sdfg(
            boundary.standalone_sdfg, workdir,
            BuildOptions(compiler=tc.cxx, flags=dace_flags, codegen_impl=codegen_impl, use_ccache=False))
        compile_us = built.compile_seconds * 1e6
        codegen_us = built.codegen_seconds * 1e6
    except Exception as e:  # a codegen/compile failure must not crash the kernel
        return {
            "ok": False,
            "error": f"dace build: {type(e).__name__}: {str(e)[:200]}",
            "codegen_impl": codegen_impl,
            **summarize_times([])
        }
    atol = flags.FP_ATOL["strict-ieee"]
    try:
        res = run_isolated(lambda: dace_run_work(built, boundary, validate_sizes, time_inputs, time_sizes, oracle, atol,
                                                 reps, validate_fills),
                           timeout=RUN_TIMEOUT_S)
    finally:
        built.unload()  # parent side: free the dlopen mapping
    if "error" in res:
        return {
            "ok": False,
            "error": res["error"],
            "compile_us": compile_us,
            "codegen_us": codegen_us,
            "codegen_impl": codegen_impl,
            **summarize_times([])
        }
    return {
        "compiler": tc.name,
        "codegen_impl": codegen_impl,
        "ok": res["ok"],
        "maxdiff": res["maxdiff"],
        "unchecked": res.get("unchecked", False),
        "compile_us": compile_us,
        "codegen_us": codegen_us,
        "error": None,
        "median_us": res["median_us"],
        "min_us": res["min_us"],
        "p25_us": res["p25_us"],
        "p75_us": res["p75_us"],
        "mean_us": res["mean_us"]
    }


def measure_dace_vectorized_lane(tc: Toolchain,
                                 boundary,
                                 validate_sizes,
                                 time_inputs,
                                 time_sizes,
                                 oracle,
                                 reps: int,
                                 cxx_std: str,
                                 workdir: Path,
                                 codegen_impl: Optional[str] = None,
                                 validate_fills=None) -> Dict:
    """DaCe lane with the multi-dim tile-op vectorizer: coordinate-descent over ``VectorizeConfig`` for
    the fastest config that still validates at ``contract-fma`` tolerance, so a mis-vectorization is
    caught rather than timed. Errors when no config validated. The winner's timing comes from the
    descent, not a re-measurement."""
    if tc.cxx is None:
        return {
            "ok": False,
            "error": f"{tc.name}: no C++ compiler for the vectorized DaCe lane",
            "vectorized": True,
            **summarize_times([])
        }
    fam = tc.fp_family
    #: The lane's FP rung, in ONE place: it picks the compile flags, the validation tolerance AND whether
    #: the tile path fuses its own multiply-adds. Denying the vectorizer a contraction the compiler is
    #: already granted would make this lane numerically stricter than the baseline it is divided by.
    fp_mode = "contract-fma"
    dace_flags = flags.base_flags(fam) + ["-std=" + cxx_std] + flags.fp_flags(fam, fp_mode, "c")
    atol = flags.FP_ATOL[fp_mode]
    results: Dict[tuple, Dict] = {}
    counter = {"i": 0}
    # Why each config was dropped. Every rejection used to return a bare None, so a lane where the compiler
    # ICE'd on all 40 configs and a lane whose nest genuinely cannot vectorize reported the SAME sentence --
    # and the first is a toolchain bug to fix, the second a result to keep.
    #: reason CLASS -> (count, one example). Keyed on the class, never on the detail: a key carrying a
    #: per-cell maxdiff or a path made every rejection unique, so the counter that exists to say
    #: "validation (x40)" reported forty entries of one.
    rejected: Dict[str, Tuple[int, str]] = {}
    #: dedup.variant_key -> the cell actually measured for that artifact.
    by_artifact: Dict[str, Dict] = {}
    #: dedup.cpp_body_key -> the cell actually measured for that emitted SOURCE. The cheaper of the two
    #: screens and the one that fires first: it costs a codegen, not a codegen + compile + run.
    by_codegen: Dict[str, Dict] = {}

    def reject(kind: str, detail: str = "") -> None:
        count, example = rejected.get(kind, (0, detail))
        rejected[kind] = (count + 1, example or detail)

    def twin_of(cfg, seen: Dict[str, Dict], key: Optional[str]) -> Optional[Dict]:
        """The already-measured cell this config would duplicate, recorded under its own name."""
        if key is None or key not in seen:
            return None
        twin = seen[key]
        results[vv.resolved_key(cfg)] = {**twin, "vec_variant": vv.variant_name(cfg), "same_as": twin["vec_variant"]}
        return twin

    def measure(cfg) -> Optional[float]:
        counter["i"] += 1
        try:
            opts = BuildOptions(compiler=tc.cxx, flags=dace_flags, codegen_impl=codegen_impl, vectorize=cfg)
            gen = generate_program(boundary.standalone_sdfg, workdir / f"vec{counter['i']}", opts)
        except Exception as e:  # noqa: BLE001 -- a config the vectorizer/codegen rejects is not a candidate
            reject(f"build: {type(e).__name__}", str(e)[:100])
            return None
        # Screen 1, on the SOURCE, before any compile. An inert nest -- one the vectorizer declines to
        # touch -- emits the same code for every cell, and a knob whose flip is byte-identical here cannot
        # change a timing. Catching it now costs one codegen instead of a codegen + compile + timed run.
        code_key = cpp_body_key(gen.source)
        twin = twin_of(cfg, by_codegen, code_key)
        if twin is not None:
            return twin["median_us"]
        try:
            built = compile_program(gen, opts)
        except Exception as e:  # noqa: BLE001 -- a config the compiler rejects is not a candidate either
            reject(f"build: {type(e).__name__}", str(e)[:100])
            return None
        # Screen 2, on the ARTIFACT, as arena.run_arena does on the flag axis: two DIFFERENT sources can
        # still compile to the same code, which only the object shows. One objdump against reps x kernel.
        # WHOLE object, never `__program_<name>`: that entry point is a three-instruction trampoline into
        # `__program_<name>_internal`, so it disassembles identically for every config and keying on it
        # collapsed the whole descent onto the first cell. __dace_init_*/__dace_exit_* stay IN the key --
        # init is where persistent storage is allocated, so a config that allocates differently really is a
        # different build. Over-separating costs one extra measurement; over-collapsing deletes the axis.
        key = variant_key(built.so_path)
        twin = twin_of(cfg, by_artifact, key)
        if twin is not None:
            built.unload()
            by_codegen[code_key] = twin  # else this source stays unrecorded and the cheap screen holds
            return twin["median_us"]  # one entry for the whole lane
        try:
            res = run_isolated(lambda: dace_run_work(built, boundary, validate_sizes, time_inputs, time_sizes, oracle,
                                                     atol, reps, validate_fills),
                               timeout=RUN_TIMEOUT_S)
        finally:
            built.unload()
        if "error" in res:
            reject("run", str(res["error"])[:100])
            return None
        if not res.get("ok"):
            reject("validation: maxdiff > atol", f"maxdiff {res.get('maxdiff')} > {atol}")
            return None
        if not finite(res.get("median_us", float("inf"))):
            reject("timing: non-finite median")
            return None
        cell = {**res, "vec_variant": vv.variant_name(cfg)}
        results[vv.resolved_key(cfg)] = cell
        by_codegen[code_key] = cell
        if key is not None:
            by_artifact[key] = cell
        return res["median_us"]

    # Prune the one axis this NEST may not be able to move before spending a compile on it: fp_factor is a
    # no-op without an integer conditional -- it would measure the same build under a second name (verified
    # identical codegen; see docs/VECTORIZATION_KNOBS.md).
    axes = vv.descent_axes(live=vv.live_axes(boundary.standalone_sdfg))
    best_cfg, best_t = vv.multistart_descent(vv.default_seeds(fp_mode=fp_mode), axes, measure)
    winner = results.get(vv.resolved_key(best_cfg)) if best_t is not None else None
    if winner is None:
        top = sorted(rejected.items(), key=lambda kv: -kv[1][0])[:3]
        why = "; ".join(f"{kind} (x{count}{': ' + example if example else ''})" for kind, (count, example) in top)
        return {
            "ok": False,
            "error": f"no vectorization config validated ({counter['i']} tried): {why or 'none attempted'}",
            "vectorized": True,
            "compiler": tc.name,
            **summarize_times([])
        }
    return {
        "compiler": tc.name,
        "vectorized": True,
        "vec_variant": winner["vec_variant"],
        "ok": winner["ok"],
        "maxdiff": winner["maxdiff"],
        "error": None,
        "median_us": winner["median_us"],
        "min_us": winner["min_us"],
        "p25_us": winner["p25_us"],
        "p75_us": winner["p75_us"],
        "mean_us": winner["mean_us"]
    }


# --- lane 3 cell model -------------------------------------------------------------------------------
@dataclass(slots=True)
class Cell:
    """One lane-3 sweep point: the full axis coordinates + correctness + median timing, or an error."""
    opt_mode: str
    language: str
    compiler: str  # family label (gcc | clang | nvhpc | intel)
    parallel: str
    cost_model: str
    fp_mode: str
    role: str  # "timing" | "gate"
    nest: int = 0  # 0-based extracted nest; -1 = whole-kernel (native lane)
    veclib: str = "none"  # 'none' | the per-device characterized winner
    ok: bool = False
    maxdiff: float = float("inf")
    median_us: float = float("inf")
    min_us: float = float("inf")
    p25_us: float = float("inf")
    p75_us: float = float("inf")
    mean_us: float = float("inf")
    compile_us: float = float("inf")
    error: Optional[str] = None


@dataclass(slots=True)
class Pending:
    """A cell not yet compiled/run, plus the context it needs (kept off the serialized :class:`Cell`)."""
    cell: Cell
    compile_key: Optional[Tuple] = None  # (exe, flags, src) -> dedup identical compiles
    atol: float = 0.0
    symbol: str = ""
    order: List[str] = field(default_factory=list)
    argtypes: list = field(default_factory=list)
    opt_ctx_key: Tuple = ()  # (opt_mode, nest_idx, lang) -> shared per-nest boundary/oracle/sizes


def compiler_for(lang: str, tc: Toolchain, fortran_by_family: Dict[str, str]) -> Optional[str]:
    """The compiler executable for a (language, family), or ``None`` if absent."""
    if lang == "c":
        return tc.cc
    if lang == "c++":
        return tc.cxx
    return fortran_by_family.get(tc.name)


def emit_lang_sources(prep,
                      boundary,
                      workdir: Path,
                      languages: List[str],
                      validate_sizes: Dict[str, int],
                      name: str,
                      parallel: bool = False) -> Dict[str, Tuple[Path, List[str], list]]:
    """Emit the numpyto sources for the requested languages and parse each signature.

    C and C++ share one emitted ``.c``; the C++ lane wraps it in a generated ``.cpp`` with
    ``extern "C" {}``, else the frontend name-mangles the symbol and ctypes can't bind it.

    :param name: per-nest base name; the emitted symbol is ``<name>_fp64``.
    :param parallel: emit the OpenMP variant (same symbol/signature). numpyto raises for a nest with no
        sound parallel form, which the caller reads as "no OpenMP lane".
    :returns: ``{lang: (src_path, arg_order, argtypes)}``. An emit that fails RAISES -- a silently absent
        language would read downstream as "this nest has no C lane" (see build_opt_context)."""
    symbol = f"{name}_fp64"
    names = list(boundary.standalone_sdfg.arrays) + list(validate_sizes)
    c_target = "c_omp" if parallel else "c"
    f_target = "fortran_omp" if parallel else "fortran"
    out: Dict[str, Tuple[Path, List[str], list]] = {}
    want_c = any(lg in ("c", "c++") for lg in languages)
    c_info: Optional[Tuple[Path, List[str], list]] = None
    if want_c:
        src = next(s for s in emit_sources(prep, workdir, target=c_target)
                   if s.suffix == ".c" and "pluto" not in s.name)
        order = signature_order(src.read_text(), symbol, "c")
        c_info = (src, order, c_argtypes(order, boundary))
    for lang in languages:
        if lang == "c":
            if c_info is not None:
                out[lang] = c_info
        elif lang == "c++":
            if c_info is not None:
                c_src, order, argtypes = c_info
                wrapper = workdir / f"{name}_cxxwrap.cpp"
                wrapper.write_text('extern "C" {\n#include "%s"\n}\n' % c_src.resolve())
                out[lang] = (wrapper, order, argtypes)
        else:  # fortran
            fsrc = next(s for s in emit_sources(prep, workdir, target=f_target)
                        if s.suffix == ".f90" and "pluto" not in s.name)
            forder = fortran_unmunge(signature_order(fsrc.read_text(), symbol, "fortran"), names)
            out[lang] = (fsrc, forder, c_argtypes(forder, boundary))
    return out


# --- per-kernel run ----------------------------------------------------------------------------------
def build_opt_context(kernel, opt_mode: str, strategy: str, profile_preset: str, languages: List[str],
                      workdir: Path) -> List[Dict]:
    """Everything a lane-3 sweep needs for ONE opt-mode: every extracted nest with its own sizes, oracle
    and per-language sources. Raises on a skip condition (no nest, emit failure).

    :func:`extract_nest_to_sdfg` mutates the parent SDFG in place, so a captured ``(parent, node)`` goes
    stale after the first extraction: rebuild a fresh SDFG + refs per nest (deterministic, so ``refs_i``
    aligns with the initial ``refs``)."""
    sdfg = tsvc.build_sdfg(kernel, opt_mode=opt_mode)
    refs = get_strategy(strategy)(sdfg)
    if not refs:
        raise RuntimeError(empty_strategy_reason(sdfg))
    ctxs: List[Dict] = []
    for idx in range(len(refs)):
        sdfg_i = tsvc.build_sdfg(kernel, opt_mode=opt_mode)
        refs_i = get_strategy(strategy)(sdfg_i)
        parent, node = refs_i[idx]
        parallel = is_parallel_nest(node)  # read the schedule BEFORE extraction mutates the graph
        name = f"{kernel.key}_n{idx}"
        boundary = extract_nest_to_sdfg(parent, node, name=name)
        nest_dir = workdir / f"n{idx}"
        time_sizes = tsvc.sample_sizes(kernel, boundary, preset=profile_preset)
        validate_sizes = tsvc.sample_sizes(kernel, boundary, preset=validate_cap(profile_preset))
        # index-array fills, seeded once per nest so the oracle and every validating cell see the same subscripts.
        validate_fills = tsvc.index_fills(kernel, boundary, validate_sizes)
        time_fills = tsvc.index_fills(kernel, boundary, time_sizes)
        prep = prepare(boundary, name, nest_dir, sizes=validate_sizes)
        oracle = run_oracle(prep, boundary, make_inputs(boundary, validate_sizes, seed=0, given=validate_fills),
                            validate_sizes)
        lang_src = emit_lang_sources(prep, boundary, nest_dir, languages, validate_sizes, name)
        # A DaCe-parallel nest also gets the OpenMP omp-emit sources; numpyto refuses an unsound-to-parallelize
        # nest (RuntimeError, not fatal -- just no omp-emit lane). Own subdir so _omp.c never shadows the .c glob.
        omp_src: Dict[str, Tuple[Path, List[str], list]] = {}
        if parallel:
            try:
                omp_src = emit_lang_sources(prep,
                                            boundary,
                                            nest_dir / "omp",
                                            languages,
                                            validate_sizes,
                                            name,
                                            parallel=True)
            except (RuntimeError, StopIteration) as e:
                print(f"[tsvc-full] {name}: no OpenMP variant ({type(e).__name__}: {str(e)[:120]})")
        # Precompute the veclib gate per lang ONCE, here where the sources exist (enumeration stays I/O-free).
        has_math: Dict[str, bool] = {
            lang: source_has_math(src.read_text())
            for lang, (src, _o, _a) in lang_src.items() if lang != "c++"
        }
        if "c++" in lang_src:
            # C++ compiles the same emitted .c (its own file is just an extern "C" wrapper), so its math
            # gate reads that .c from disk rather than has_math["c"] -- "c" may not have been requested.
            c_src = next(s for s in nest_dir.glob("*.c") if "pluto" not in s.name)
            has_math["c++"] = source_has_math(c_src.read_text())
        ctxs.append({
            "nest_idx": idx,
            "name": name,
            "symbol": f"{name}_fp64",
            "boundary": boundary,
            "time_sizes": time_sizes,
            # allocated ONCE per nest; every timing cell reuses it via COW-fork isolation (see nest_timing_work).
            "time_inputs": make_inputs(boundary, time_sizes, seed=0, given=time_fills),
            "validate_sizes": validate_sizes,
            "validate_fills": validate_fills,
            "oracle": oracle,
            "lang_src": lang_src,
            # the source every language variant is translated FROM -- the nest's identity in the dataset
            "numpy_source": prep.numpy_source,
            "has_math": has_math,
            "omp_src": omp_src,
            "parallel": parallel,
        })
    return ctxs


def cost_models_for(parallel: str, cost_models: List[str], matrix_preset: str) -> List[str]:
    """Vectorizer cost models to sweep for a ``parallel`` mode under a ``matrix_preset``. The cost axis
    is a single-core question (scalar floor vs vectorizer); once threaded, cost points add little value.
    ``full`` sweeps every cost model on every parallel mode; ``lean`` sweeps the full axis only on
    ``sequential`` and collapses ``auto-par``/``omp-emit`` to the compiler default (roughly halves timing
    cells, keeps all single-core vec data). Unknown preset -> behaves like ``full`` (fail open)."""
    if matrix_preset == "lean" and parallel != "sequential":
        return ["default"] if "default" in cost_models else list(cost_models[:1])
    return list(cost_models)


#: libm transcendentals a vector-math library (``-fveclib``) can accelerate; a nest calling none of these
#: gets no veclib axis (the library would be inert for it).
_MATH_CALLS = ("sin", "cos", "exp", "log", "pow", "tan", "asin", "acos", "atan", "atan2", "hypot", "sinh", "cosh",
               "tanh", "cbrt", "expm1", "log1p")


def source_has_math(text: str) -> bool:
    """True if the emitted source calls a libm transcendental a veclib could vectorize (plain ``name(``
    scan). Cheap gate so a math-free nest skips the veclib axis."""
    return any(f"{fn}(" in text for fn in _MATH_CALLS)


def veclibs_for(has_math: bool, veclibs: Sequence[str], compiler: Optional[str]) -> Tuple[str, ...]:
    """Veclib values to sweep for one (nest-lang, compiler): always ``none``, plus a candidate only when
    the nest has a libm call and the veclib is compatible with the compiler (incompatible ones are
    dropped here, never turned into an error cell). Takes the precomputed ``has_math`` flag so
    :func:`enumerate_cells` stays I/O-free."""
    out = ["none"]
    if has_math:
        for vl in veclibs:
            if vl == "none":
                continue
            resolved, _ = flags.veclib_flags(compiler, vl)
            if resolved:  # compatible and contributes at least one flag
                out.append(vl)
    return tuple(out)


def enumerate_cells(opt_ctx: Dict, toolchains: List[Toolchain], fortran_by_family: Dict[str, str], axes: Dict,
                    nthreads: int, cxx_std: str, workdir: Path) -> Tuple[List[Pending], Dict[Tuple, Dict]]:
    """Build every lane-3 :class:`Pending` cell for one opt-mode + the deduped compile jobs.

    Compile-job key is ``(exe, flags-tuple, src)`` so two axis points with identical flags compile once.
    An unsupported combo (clang auto-par) becomes a finished error :class:`Cell` with no compile job."""
    pendings: List[Pending] = []
    jobs: Dict[Tuple, Dict] = {}
    opt_mode = axes["opt_mode"]
    nest_idx = opt_ctx.get("nest_idx", 0)  # dict.get: a synthetic ctx may omit it -> nest 0
    # Dedup identical TIMING cells: axis points whose (exe, flags, src) collapse to the same compile (e.g.
    # clang/icx/nvc `cheap` == `default`) are the same measurement -- keep the first, skip the rest. Gate
    # cells are exempt (different FP flags, never collide) so the bit-exact gate always runs.
    seen_timing: set = set()

    def add(cell: Cell, exe: str, cflags: List[str], src: Path, atol: float, symbol, order, argtypes, ctx_key):
        key = (exe, tuple(cflags), str(src))
        if cell.role == "timing":
            if key in seen_timing:
                return
            seen_timing.add(key)
        so = workdir / (f"{opt_mode}_n{cell.nest}_{cell.language.replace('+', 'x')}_{cell.compiler}_{cell.parallel}_"
                        f"{cell.cost_model}_{cell.fp_mode}_{cell.veclib}_{len(jobs)}.so")
        jobs.setdefault(key, {"exe": exe, "flags": cflags, "src": src, "so": so})
        pendings.append(
            Pending(cell=cell,
                    compile_key=key,
                    atol=atol,
                    symbol=symbol,
                    order=order,
                    argtypes=argtypes,
                    opt_ctx_key=ctx_key))

    # Read once: cached_default_runtime hits a JSON cache file, so it must not be called in the loops below.
    machine_runtime = support_matrix.cached_default_runtime()

    for lang, (src, order, argtypes) in opt_ctx["lang_src"].items():
        symbol = opt_ctx["symbol"]
        ctx_key = (opt_mode, nest_idx, lang)
        # veclib gate is a PRECOMPUTED per-lang fact (build_opt_context read the source once); enumeration
        # itself does no source I/O, so it runs on synthetic ctxs with dummy paths. Absent -> math-free.
        has_math = opt_ctx.get("has_math", {}).get(lang, False)
        for tc in toolchains:
            exe = compiler_for(lang, tc, fortran_by_family)
            fam = tc.fp_family
            if exe is None:
                pendings.append(
                    Pending(cell=Cell(opt_mode,
                                      lang,
                                      tc.name,
                                      "sequential",
                                      "default",
                                      "-",
                                      "timing",
                                      nest=nest_idx,
                                      error=f"no {lang} compiler for family {tc.name}")))
                continue
            # timing cells: parallel x cost x reduced-FP
            # TODO(machine-compat prune): skip cells MachineCompat.is_supported rejects, keyed on
            # compiler_family(exe) not tc.fp_family (icx: 'llvm' vs 'intel' -- must not conflate).
            for parallel in axes["parallelism"]:
                # omp-emit uses our #pragma omp source; exists only for a DaCe-parallel, soundly-parallelizable nest.
                psrc = src
                if parallel == "omp-emit":
                    omp = opt_ctx.get("omp_src", {}).get(lang)
                    if omp is None:
                        continue
                    psrc = omp[0]
                for cost in cost_models_for(parallel, axes["cost_models"], axes.get("matrix_preset", "full")):
                    for fp in axes["fp_modes"]:
                        for veclib in veclibs_for(has_math, axes.get("veclibs", ("none", )), exe):
                            cflags, reason = flags.lane_flags(fam,
                                                              fp,
                                                              cost,
                                                              parallel,
                                                              lang,
                                                              nthreads,
                                                              cxx_std,
                                                              compiler=exe,
                                                              veclib=veclib,
                                                              openmp=machine_runtime)
                            cell = Cell(opt_mode,
                                        lang,
                                        tc.name,
                                        parallel,
                                        cost,
                                        fp,
                                        "timing",
                                        nest=nest_idx,
                                        veclib=veclib)
                            if cflags is None:
                                if veclib == "none":  # base combo unsupported (e.g. clang auto-par): record once
                                    cell.error = f"unsupported: {reason}"
                                    pendings.append(Pending(cell=cell))
                                continue
                            add(cell, exe, cflags, psrc, flags.REDUCED_FP_ATOL[fp], symbol, order, argtypes, ctx_key)
            # correctness GATE cell: strict-ieee, sequential, default cost (bit-exact vs the oracle)
            if axes["gate"]:
                cflags, _ = flags.lane_flags(fam,
                                             "strict-ieee",
                                             "default",
                                             "sequential",
                                             lang,
                                             nthreads,
                                             cxx_std,
                                             compiler=exe)
                gate = Cell(opt_mode, lang, tc.name, "sequential", "default", "strict-ieee", "gate", nest=nest_idx)
                add(gate, exe, cflags, src, flags.FP_ATOL["strict-ieee"], symbol, order, argtypes, ctx_key)
    return pendings, jobs


# --- Pluto polyhedral lane (opt-in --pluto): polycc transform of the emitted scop -------------------
# Distinct polyhedral TOOLCHAIN, not a flag: polycc tiles + auto-parallelizes the emitted scop offline into
# a new C source. Its VLA-parameter signature (size symbols first) can't be parsed by signature_order, so
# the ABI comes from ``_pluto_binding.json`` instead (see nestforge.perf.pluto_lane). polycc is almost
# always absent off the optarena container -> the lane records a skip reason, never a crash.
def pluto_validate_work(so: Path,
                        symbol: str,
                        order: List[str],
                        argtypes,
                        boundary,
                        validate_sizes,
                        oracle,
                        given=None) -> Dict:
    """Fresh-buffer correctness run of the compiled Pluto ``.so`` vs the numpy oracle, in the forked child.
    Marshals with the Pluto (size-symbols-first) ``order``; the VLA params decay to pointers, so the same
    ``call_c`` the C lane uses binds them correctly once the order is right."""
    inputs = make_inputs(boundary, validate_sizes, seed=0, given=given)  # fresh buffer (the call runs in place)
    outputs, _ = call_c(so, symbol, order, argtypes, boundary, inputs, validate_sizes, 1)
    outs = {o: outputs[o] for o in boundary.outputs if o in outputs}
    if not outs:  # nothing to compare -> UNCHECKED, never report ok for an unvalidatable lane
        return {"ok": False, "maxdiff": float("inf"), "unchecked": True}
    ref = {o: oracle[o] for o in outs}
    md = float(maxdiff(ref, outs))  # absolute is REPORTED, the scaled one is the gate (as every other lane)
    return {"ok": bool(relative_maxdiff(ref, outs) <= flags.NATIVE_ATOL), "maxdiff": md}


def measure_pluto_lane(nc: Dict, cc: Optional[str], reps: int, workdir: Path) -> Dict:
    """Pluto lane for ONE nest: locate the emitted scop, gate it (no-scop / polycc-absent / non-affine),
    run ``polycc``, compile the transform, validate@small + time@prof. Returns a status dict with a
    ``skip`` key for a recorded skip, or the usual ``ok``/``maxdiff``/timing shape. ``cc`` is a GNU C
    compiler; when it or the scop is missing the lane skips cleanly."""
    workdir.mkdir(parents=True, exist_ok=True)
    label = {"compiler": "pluto", "nest": nc["nest_idx"]}
    src_dir = next(iter(nc["lang_src"].values()))[0].parent  # every nest source sits in the nest dir
    pluto_input = pluto_lane.find_pluto_input(sorted(src_dir.glob("*.c")))
    reason = pluto_lane.pluto_gate_reason(pluto_input)
    if reason is not None:
        return {**label, "skip": reason, **summarize_times([])}
    if cc is None:
        return {**label, "skip": "skip:not-installed:no-gnu-cc", **summarize_times([])}
    binding = pluto_lane.read_pluto_binding(pluto_input)
    if binding is None:
        return {**label, "skip": "skip:unsupported:no-binding", **summarize_times([])}
    symbol, order = pluto_lane.binding_symbol_and_order(binding)
    boundary = nc["boundary"]
    # ABI guard: every binding arg must be a real boundary array/size symbol, else reordered VLA
    # marshaling would land a size in a pointer slot.
    known = set(boundary.standalone_sdfg.arrays) | set(nc["validate_sizes"])
    unknown = [a for a in order if a not in known]
    if unknown:
        return {**label, "ok": False, "error": f"pluto binding args not in boundary: {unknown}", **summarize_times([])}
    out_c = pluto_lane.pluto_output_path(pluto_input)
    ok, pre = pluto_lane.run_polycc(pluto_input, out_c, COMPILE_TIMEOUT_S)
    if not ok:
        return {**label, "skip": pre, **summarize_times([])}
    so = workdir / f"pluto_n{nc['nest_idx']}.so"
    # PLUTO_EXTRA_FLAGS carries -fopenmp; pin the machine's discovered runtime so it doesn't link a second
    # (gcc default) libgomp beside the one every other lane uses.
    omp_rt, omp_reason = flags.openmp_runtime_flags(cc, "gnu", support_matrix.cached_default_runtime())
    if omp_rt is None:
        return {**label, "skip": omp_reason, **summarize_times([])}
    cflags = list(flags.base_flags("gnu")) + list(pluto_lane.PLUTO_EXTRA_FLAGS) + omp_rt
    cok, compile_us, cerr = run_compile([cc, *cflags, str(out_c), "-o", str(so)])
    if not cok:
        return {**label, "ok": False, "error": cerr, "compile_us": compile_us, **summarize_times([])}
    argtypes = c_argtypes(order, boundary)
    vres = run_isolated(lambda: pluto_validate_work(so, symbol, order, argtypes, boundary, nc["validate_sizes"], nc[
        "oracle"], nc["validate_fills"]))
    if "error" in vres:
        return {**label, "ok": False, "error": vres["error"], "compile_us": compile_us, **summarize_times([])}
    tres = run_isolated(
        lambda: nest_timing_work(so, symbol, order, argtypes, boundary, nc["time_inputs"], nc["time_sizes"], reps),
        timeout=RUN_TIMEOUT_S)
    stats = summarize_times([]) if "error" in tres else tres
    return {
        **label, "ok": vres["ok"],
        "maxdiff": vres["maxdiff"],
        "unchecked": vres.get("unchecked", False),
        "compile_us": compile_us,
        "error": tres.get("error"),
        **stats
    }


def record_nest_winner(kernel, nc: Dict, axis: str, cell: Dict, baseline_cells: List[Dict], compiler: str,
                       dataset_dir: Optional[Path]) -> None:
    """Append this nest's winning config to the dataset, so a sweep leaves behind what it learned and not
    only what it printed. Never raises: the tables are the job's product, the dataset is a by-product."""
    if not cell.get("ok") or not finite(cell.get("median_us", float("inf"))):
        return  # a failed lane has no winner to learn from; the skip reason is already on the cell
    base = next((b for b in baseline_cells if b.get("nest") == nc["nest_idx"] and b.get("ok")), None)
    base_us = base.get("median_us") if base else None
    base_us = base_us if finite(base_us if base_us is not None else float("inf")) else None
    try:
        # The NUMPY rendering is the nest's identity: every emitted C/C++/Fortran variant is a translation
        # of it, so it is the one form upstream of the language axis. Stored content-addressed, so a nest
        # a hundred jobs measured is one file on disk.
        record_winner(
            WinnerRecord(kernel=kernel.key,
                         corpus=kernel.corpus,
                         nest=nc["nest_idx"],
                         fingerprint=store_nest(nc["numpy_source"], dataset_dir),
                         axis=axis,
                         winner={
                             k: v
                             for k, v in cell.items() if k in ("vec_variant", "compiler", "codegen_impl")
                         },
                         median_us=float(cell["median_us"]),
                         baseline_us=base_us,
                         speedup=(base_us / cell["median_us"]) if base_us else None,
                         features=nest_features(nc["boundary"].standalone_sdfg),
                         compiler=compiler), dataset_dir)
    except Exception as e:  # noqa: BLE001 -- a by-product must never take the sweep down
        print(f"[tsvc-full] dataset record failed for {kernel.key} n{nc['nest_idx']}: {type(e).__name__}: {e}")


def run_kernel(kernel,
               toolchains: List[Toolchain],
               fortran_by_family: Dict[str, str],
               strategy: str,
               axes: Dict,
               reps: int,
               profile_preset: str,
               nthreads: int,
               cxx_std: str,
               compile_jobs: int,
               workdir: Path,
               dataset_dir: Optional[Path] = None) -> Dict:
    """Run all three lanes + the full lane-3 sweep for one kernel; return the JSON-able result."""
    result = {
        "key": kernel.key,
        "corpus": kernel.corpus,
        "regime": kernel.regime,
        "host": socket.gethostname(),
        "profile_preset": profile_preset
    }

    # per-opt-mode context: every extracted nest as a list of per-nest ctxs. A failing opt-mode is noted, not fatal.
    contexts: Dict[str, List[Dict]] = {}
    opt_notes: Dict[str, str] = {}
    for opt_mode in axes["opt_modes"]:
        try:
            contexts[opt_mode] = build_opt_context(kernel, opt_mode, strategy, profile_preset, axes["languages"],
                                                   workdir / opt_mode)
        except Exception as e:
            opt_notes[opt_mode] = f"{type(e).__name__}: {str(e)[:160]}"
    if not contexts:
        return {**result, "skipped": "; ".join(f"{m}: {n}" for m, n in opt_notes.items()) or "no opt-mode built"}

    base_key = "simplify-parallel" if "simplify-parallel" in contexts else next(iter(contexts))
    base_ctxs = contexts[base_key]  # the base opt-mode's per-nest ctxs (what lanes 1/2 measure against)
    result["baseline_opt"] = base_key  # which opt-mode's nests lanes 1/2 used (may not be 'simplify-parallel')
    # lane 1 (native): one whole-kernel measurement (nest=-1), borrows the base opt-mode's first nest for
    # buffer sizing since it's compared against the SUM over nests. lane 2 (DaCe-cpp): runs per nest,
    # denominator is the sum. One C++ toolchain (first with cxx) drives both.
    cxx_tc = next((tc for tc in toolchains if tc.cxx is not None), toolchains[0] if toolchains else None)
    if cxx_tc is not None:
        nat0 = base_ctxs[0]
        native = measure_native_lane(cxx_tc.cxx, cxx_tc.fp_family, kernel, nat0["boundary"], nat0["validate_sizes"],
                                     nat0["time_inputs"], nat0["time_sizes"], nat0["oracle"], reps, cxx_std, workdir,
                                     nat0["validate_fills"])
        if native is not None:
            native["nest"] = -1  # whole-kernel sentinel: not a single-nest measurement
        result["native"] = native
        # DaCe lane fans out over codegen impl: 'legacy' + 'experimental' where available. Each cell carries
        # nest idx + codegen_impl; the reporter sums legacy for the denominator, geomeans experimental against it.
        dace_cpp_cells: List[Dict] = []
        for nc in base_ctxs:
            for impl in codegen_impls_available():
                dcell = measure_dace_cpp_lane(cxx_tc,
                                              nc["boundary"],
                                              nc["validate_sizes"],
                                              nc["time_inputs"],
                                              nc["time_sizes"],
                                              nc["oracle"],
                                              reps,
                                              cxx_std,
                                              workdir / "dace_cpp" / f"n{nc['nest_idx']}_{impl}",
                                              codegen_impl=impl,
                                              validate_fills=nc["validate_fills"])
                dcell["nest"] = nc["nest_idx"]
                dace_cpp_cells.append(dcell)
        result["dace_cpp"] = dace_cpp_cells
        # Optional vectorized DaCe lane: the multi-dim tile-op vectorizer, coordinate-descent per nest for
        # the fastest validating VectorizeConfig. Opt-in (--vectorize) since it compiles ~15-30 cells/nest.
        if axes.get("vectorize"):
            dace_vec_cells: List[Dict] = []
            for nc in base_ctxs:
                vcell = measure_dace_vectorized_lane(cxx_tc,
                                                     nc["boundary"],
                                                     nc["validate_sizes"],
                                                     nc["time_inputs"],
                                                     nc["time_sizes"],
                                                     nc["oracle"],
                                                     reps,
                                                     cxx_std,
                                                     workdir / "dace_vec" / f"n{nc['nest_idx']}",
                                                     codegen_impl=default_codegen_impl(),
                                                     validate_fills=nc["validate_fills"])
                vcell["nest"] = nc["nest_idx"]
                dace_vec_cells.append(vcell)
                record_nest_winner(kernel, nc, "vectorize", vcell, dace_cpp_cells, cxx_tc.name, dataset_dir)
            result["dace_cpp_vec"] = dace_vec_cells

    # Optional Pluto polyhedral lane: polycc transform of the emitted scop, one cell per nest. Opt-in
    # (--pluto); polycc almost always absent off the optarena container -> each nest records a skip reason.
    if axes.get("pluto"):
        gnu_cc = next((tc.cc for tc in toolchains if tc.fp_family == "gnu" and tc.cc is not None), None)
        pluto_cells: List[Dict] = []
        for nc in base_ctxs:
            pcell = measure_pluto_lane(nc, gnu_cc, reps, workdir / "pluto" / f"n{nc['nest_idx']}")
            pluto_cells.append(pcell)
        result["pluto"] = pluto_cells

    # lane 3: enumerate + dedup-compile + validate + time. Each nest is enumerated independently.
    all_pending: List[Pending] = []
    all_jobs: Dict[Tuple, Dict] = {}
    ctx_by_key: Dict[Tuple, Dict] = {}
    for opt_mode, nest_ctxs in contexts.items():
        axes_om = {**axes, "opt_mode": opt_mode}
        for nc in nest_ctxs:
            pend, jobs = enumerate_cells(nc, toolchains, fortran_by_family, axes_om, nthreads, cxx_std,
                                         workdir / opt_mode)
            all_pending.extend(pend)
            all_jobs.update(jobs)
            for lang in nc["lang_src"]:
                ctx_by_key[(opt_mode, nc["nest_idx"], lang)] = nc

    # (a) parallel compile of the unique flag sets (compilation is the bottleneck; the timed runs are not).
    def do_compile(job: Dict) -> Tuple[bool, float, Optional[str]]:
        return run_compile([job["exe"], *job["flags"], str(job["src"]), "-o", str(job["so"])])

    if all_jobs:
        with ThreadPoolExecutor(max_workers=max(1, compile_jobs)) as pool:
            for key, res in zip(all_jobs, pool.map(do_compile, all_jobs.values())):
                all_jobs[key]["result"] = res

    # (b) validate every compiled cell (fast, small preset); (c) time only the ones that validate.
    cells_out: List[Dict] = []
    for p in all_pending:
        cell = p.cell
        if p.compile_key is None:  # unsupported combo or missing compiler -> already an error cell
            cells_out.append(asdict(cell))
            continue
        job = all_jobs[p.compile_key]
        ok, compile_us, cerr = job["result"]
        cell.compile_us = compile_us
        if not ok:
            cell.error = cerr
            cells_out.append(asdict(cell))
            continue
        ctx = ctx_by_key[p.opt_ctx_key]
        so = job["so"]
        vres = run_isolated(lambda: nest_validate_work(so, p.symbol, p.order, p.argtypes, ctx["boundary"], ctx[
            "validate_sizes"], ctx["oracle"], p.atol, ctx["validate_fills"]))
        if "error" in vres:
            cell.error = vres["error"]
            cells_out.append(asdict(cell))
            continue
        cell.ok, cell.maxdiff = vres["ok"], vres["maxdiff"]
        if cell.role == "timing" and cell.ok:  # only VALIDATED timing cells are (expensively) timed
            tres = run_isolated(lambda: nest_timing_work(so, p.symbol, p.order, p.argtypes, ctx["boundary"], ctx[
                "time_inputs"], ctx["time_sizes"], reps),
                                timeout=RUN_TIMEOUT_S)
            if "error" in tres:
                cell.error = f"timing: {tres['error']}"
            else:
                cell.median_us, cell.min_us = tres["median_us"], tres["min_us"]
                cell.p25_us, cell.p75_us, cell.mean_us = tres["p25_us"], tres["p75_us"], tres["mean_us"]
        cells_out.append(asdict(cell))

    result["cells"] = cells_out
    # per-nest sizes of the base opt-mode, keyed by nest idx (as a str for JSON object keys).
    result["sizes"] = {
        str(nc["nest_idx"]): {
            "validate": {
                k: int(v)
                for k, v in nc["validate_sizes"].items()
            },
            "time": {
                k: int(v)
                for k, v in nc["time_sizes"].items()
            },
        }
        for nc in base_ctxs
    }
    if opt_notes:
        result["opt_notes"] = opt_notes
    return result


# --- tables ------------------------------------------------------------------------------------------
def kernel_winner(cells: List[Dict],
                  opt_mode: str,
                  lang: str,
                  compiler: str,
                  nest: Optional[int] = None) -> Optional[Dict]:
    """Fastest validated timing cell (min median) for one (opt, lang, compiler[, nest]) group, or None.

    ``nest=None`` matches any nest; :func:`render_tables` passes an explicit nest and sums per-nest
    winners for the kernel's offloaded time. ``c.get("nest", 0)`` tolerates a cell predating the field."""
    grp = [
        c for c in cells if c["role"] == "timing" and c["ok"] and c["opt_mode"] == opt_mode and c["language"] == lang
        and c["compiler"] == compiler and (nest is None or c.get("nest", 0) == nest) and finite(c["median_us"])
    ]
    return min(grp, key=lambda c: c["median_us"]) if grp else None


def render_tables(out: Path) -> str:
    kernels = load_results(out)
    done = [k for k in kernels if "cells" in k]
    skipped = [k for k in kernels if "skipped" in k]

    lines = [
        "# TSVC full-matrix job (native cpp / DaCe-cpp baseline / nest-forge sweep)",
        "",
        f"{len(done)} kernels measured, {len(skipped)} skipped. Timing is MEDIAN of N reps at the profiling "
        "preset; the speedup baseline is the DaCe-cpp (strict-ieee) lane.",
        "",
    ]

    # strict-ieee gate must validate within the strict tolerance, not maxdiff==0 (reductions/transcendentals
    # differ ~1e-15 from numpy's pairwise sum), so it agrees with the cell's own FP_ATOL["strict-ieee"] verdict.
    gate_fail = []
    for k in done:
        for c in k["cells"]:
            if c["role"] == "gate" and c["error"] is None and not c["ok"]:
                gate_fail.append((k["key"], k["corpus"], c["opt_mode"], c["language"], c["compiler"], c["maxdiff"]))
    gate_status = "PASS" if not gate_fail else f"{len(gate_fail)} FAILURES"
    lines += [
        f"**strict-ieee gate:** {gate_status} "
        "(validates within the strict tolerance vs the numpy oracle).", ""
    ]
    if gate_fail:
        lines += ["| kernel | corpus | opt | lang | compiler | maxdiff |", "|" + "---|" * 6]
        for a, b, c, d, e, f in gate_fail:
            mds = "—" if f is None else f"{f:g}"
            lines.append(f"| {a} | {b} | {c} | {d} | {e} | {mds} |")
        lines.append("")

    lines += [
        "## per (kernel, opt-mode, language, compiler): best offloaded nests vs the DaCe-cpp baseline", "",
        "A kernel may split into several loop-nests; the offloaded time is the SUM over nests of each nest's "
        "own best (min validated median) cell, and the DaCe-cpp baseline is the SUM over nests of the "
        "per-nest DaCe-cpp lane -- an apples-to-apples denominator. The offloaded sum is blank unless EVERY "
        "nest has a validated cell for that (lang, compiler). A `†` marks a baseline where some nest did not "
        "bit-match the oracle (loop-carried-state kernel -- DaCe vs numpyto boundary-semantics divergence). "
        "``native`` is the single whole-kernel measurement (not per-nest).", "",
        "| kernel | corpus | opt | lang | compiler | native (us) | DaCe-cpp (us) | best nests (us) | best config "
        "| maxdiff | nest/DaCe |", "|" + "---|" * 11
    ]
    speedups: List[float] = []
    codegen_speedups: List[float] = []  # per-kernel legacy/experimental DaCe-codegen ratios (the codegen axis)
    vec_speedups: List[float] = []  # per-kernel plain-DaCe / vectorized-DaCe ratios (the vectorization axis)
    pluto_speedups: List[float] = []  # per-kernel DaCe-cpp / Pluto ratios (the polyhedral lane), where it ran
    pluto_skips: Dict[str, int] = {}  # skip-reason -> count, so an absent polycc is reported, not hidden
    novalidate = 0
    # (corpus, axis) -> ratios. The headline geomeans below pool every row, which describes NEITHER corpus
    # when a sweep covers two: tsvc2 (151 kernels) and tsvc2_5 (65) are DISJOINT key sets, so their rows are
    # not samples of one population and comparing the two corpora is the only reason to sweep both.
    per_corpus: Dict[Tuple[str, str], List[float]] = {}

    def note(corpus: str, axis: str, ratio: float) -> None:
        per_corpus.setdefault((corpus, axis), []).append(ratio)

    for k in sorted(done, key=lambda x: (x["corpus"], x["key"])):
        # dace_cpp is a list of per-nest cells (older JSON: single-dict/empty tolerated), fanned over codegen
        # impl; a cell with no codegen_impl predates the axis -> treated as legacy.
        dcpp_raw = k.get("dace_cpp")
        dcpp_list = dcpp_raw if isinstance(dcpp_raw, list) else ([dcpp_raw] if dcpp_raw else [])
        legacy_cells = [d for d in dcpp_list if d.get("codegen_impl", "legacy") == "legacy"]
        exp_cells = [d for d in dcpp_list if d.get("codegen_impl") == "experimental"]
        # Denominator = experimental when produced (nest-forge's default), else legacy.
        primary_cells = exp_cells if exp_cells else legacy_cells
        dace_meds = [d.get("median_us") for d in primary_cells]
        dace_us = sum(dace_meds) if (dace_meds and all(finite(m) for m in dace_meds)) else None
        dace_flag = "" if (dace_us is None or all(d.get("ok") for d in primary_cells)) else "†"
        if dace_flag:
            novalidate += 1
        # codegen axis: legacy total / experimental total for this kernel, when BOTH impls fully timed.
        legacy_meds = [d.get("median_us") for d in legacy_cells]
        exp_meds = [d.get("median_us") for d in exp_cells]
        legacy_us = sum(legacy_meds) if (legacy_meds and all(finite(m) for m in legacy_meds)) else None
        exp_us = sum(exp_meds) if (exp_meds and all(finite(m) for m in exp_meds)) else None
        if legacy_us and exp_us:
            codegen_speedups.append(legacy_us / exp_us)
            note(k["corpus"], "legacy/experimental codegen", legacy_us / exp_us)
        # vectorization axis: plain-DaCe (denominator) total / vectorized-DaCe total, when both fully timed.
        vec_raw = k.get("dace_cpp_vec")
        vec_list = vec_raw if isinstance(vec_raw, list) else ([vec_raw] if vec_raw else [])
        vec_meds = [d.get("median_us") for d in vec_list if d.get("ok")]
        vec_us = sum(vec_meds) if (vec_list and len(vec_meds) == len(vec_list) and all(finite(m)
                                                                                       for m in vec_meds)) else None
        if dace_us is not None and vec_us:
            vec_speedups.append(dace_us / vec_us)
            note(k["corpus"], "DaCe/tile-op-vectorized", dace_us / vec_us)
        # Pluto polyhedral lane: DaCe-cpp total / Pluto total, only when EVERY nest's Pluto cell timed
        # (a skip on any nest -> the kernel is not comparable). Skips are tallied by reason instead.
        pluto_list = k.get("pluto") or []
        for pc in pluto_list:
            if pc.get("skip"):
                pluto_skips[pc["skip"]] = pluto_skips.get(pc["skip"], 0) + 1
        pluto_meds = [d.get("median_us") for d in pluto_list if d.get("ok") and not d.get("skip")]
        pluto_us = sum(pluto_meds) if (pluto_list and len(pluto_meds) == len(pluto_list)
                                       and all(finite(m) for m in pluto_meds)) else None
        if dace_us is not None and pluto_us:
            pluto_speedups.append(dace_us / pluto_us)
            note(k["corpus"], "DaCe/Pluto", dace_us / pluto_us)
        nat = k.get("native") or {}
        nat_med = nat.get("median_us")
        nat_us = nat_med if (nat.get("ok") and finite(nat_med)) else None
        groups = sorted({(c["opt_mode"], c["language"], c["compiler"]) for c in k["cells"] if c["role"] == "timing"})
        for opt_mode, lang, comp in groups:
            nests = sorted(
                {c.get("nest", 0)
                 for c in k["cells"] if c["role"] == "timing" and c["opt_mode"] == opt_mode})
            wins = [kernel_winner(k["cells"], opt_mode, lang, comp, n) for n in nests]
            if wins and all(w is not None for w in wins):
                win_us = sum(w["median_us"] for w in wins)
                cfg = "+".join(f"{w['parallel']}/{w['cost_model']}/{w['fp_mode']}"
                               f"{'/' + w['veclib'] if w.get('veclib', 'none') != 'none' else ''}" for w in wins)
                md = f"{max(w['maxdiff'] for w in wins):g}"
            else:
                win_us, cfg, md = None, "—", "—"
            sp = (dace_us / win_us) if (dace_us is not None and win_us) else None
            if sp is not None and math.isfinite(sp):
                speedups.append(sp)
                note(k["corpus"], "best-nest/DaCe", sp)
            lines.append(f"| {k['key']} | {k['corpus']} | {opt_mode} | {lang} | {comp} | {fmt_us(nat_us)} "
                         f"| {fmt_us(dace_us)}{dace_flag} | {fmt_us(win_us)} | {cfg} | {md} | "
                         f"{'—' if sp is None else f'{sp:.2f}x'} |")

    if speedups:
        geo = math.exp(sum(math.log(s) for s in speedups) / len(speedups))
        lines += [
            "", f"**Geomean best-nest / DaCe-cpp speedup:** {geo:.3f}x over {len(speedups)} "
            "(kernel x opt x lang x compiler) rows where both timed."
        ]
    if codegen_speedups:
        cgeo = math.exp(sum(math.log(s) for s in codegen_speedups) / len(codegen_speedups))
        lines += [
            "", f"**Geomean DaCe legacy / experimental codegen speedup:** {cgeo:.3f}x over "
            f"{len(codegen_speedups)} kernels where both codegen impls timed "
            "(>1 = experimental faster; the codegen-implementation axis)."
        ]
    if vec_speedups:
        vgeo = math.exp(sum(math.log(s) for s in vec_speedups) / len(vec_speedups))
        lines += [
            "", f"**Geomean DaCe plain / tile-op-vectorized speedup:** {vgeo:.3f}x over "
            f"{len(vec_speedups)} kernels where the vectorized lane's best config timed "
            "(>1 = the multi-dim tile-op vectorizer faster; the vectorization axis)."
        ]
    if pluto_speedups:
        pgeo = math.exp(sum(math.log(s) for s in pluto_speedups) / len(pluto_speedups))
        lines += [
            "", f"**Geomean DaCe-cpp / Pluto speedup:** {pgeo:.3f}x over {len(pluto_speedups)} kernels where "
            "every nest's Pluto lane timed (>1 = the polyhedral Pluto transform faster; the Pluto lane)."
        ]
    if pluto_skips:
        summary = ", ".join(f"{n}x {reason}" for reason, n in sorted(pluto_skips.items()))
        lines += [
            "", f"**Pluto lane skips (per nest):** {summary}. A skip is recorded, never a silent drop "
            "(polycc is a separate polyhedral toolchain, absent off the optarena container)."
        ]
    corpora = {corpus for corpus, _ in per_corpus}
    if len(corpora) > 1:
        lines += [
            "", "### per corpus", "",
            "Every geomean above pools the whole sweep. tsvc2 and tsvc2_5 share NO kernel key, so a pooled "
            "number is not an estimate of either corpus -- read this table instead whenever a run covers "
            "more than one. Comparing the corpora is a comparison of two kernel POPULATIONS (tsvc2_5 is the "
            "irregular/conditional extension set), never a per-kernel A/B.", "", "| corpus | axis | geomean | rows |",
            "|" + "---|" * 4
        ]
        for (corpus, axis), ratios in sorted(per_corpus.items()):
            g = geomean(ratios)
            lines.append(f"| {corpus} | {axis} | {'—' if g is None else f'{g:.3f}x'} | {len(ratios)} |")
    if novalidate:
        lines += [
            "", f"`†` {novalidate} kernels' DaCe-cpp baseline did not bit-match the numpy oracle (DaCe vs "
            "numpyto boundary semantics for loop-carried state); their timing baseline is still "
            "representative (identical iteration space)."
        ]

    # unsupported/error summary (auto-par on clang, missing Fortran compilers, ...)
    reasons: Dict[str, int] = {}
    for k in done:
        for c in k["cells"]:
            if c["error"] and ("unsupported" in c["error"] or "no " in c["error"][:4]):
                reasons[c["error"][:80]] = reasons.get(c["error"][:80], 0) + 1
    if reasons:
        lines += ["", "## unsupported / skipped cells (recorded, not silently dropped)", ""]
        lines += [f"- {n}x — {r}" for r, n in sorted(reasons.items(), key=lambda kv: -kv[1])]

    # real compile/validation failures, not the expected unsupported/missing-compiler cells above.
    failures: Dict[str, int] = {}
    for k in done:
        for c in k["cells"]:
            err = c["error"]
            if err and not ("unsupported" in err or "no " in err[:4]):
                failures[err[:60]] = failures.get(err[:60], 0) + 1
    if failures:
        lines += ["", "## compile / validation failures (recorded)", ""]
        lines += [f"- {n}x  {r}" for r, n in sorted(failures.items(), key=lambda kv: -kv[1])[:12]]
    if skipped:
        lines += ["", "## skipped kernels", ""]
        lines += [
            f"- `{k['key']}` ({k.get('corpus', '?')}) — {k['skipped']}" for k in sorted(skipped, key=lambda x: x["key"])
        ]
    report = "\n".join(lines) + "\n"
    (out / "tables.md").write_text(report)
    return report


# --- CLI ---------------------------------------------------------------------------------------------
def resolve_veclibs(spec: List[str], compilers: Sequence[str] = ("gcc", )) -> Tuple[str, ...]:
    """Resolve ``--veclibs`` to axis values. ``'auto'`` -> none + each compiler's characterized winner
    (compiles tiny probes once per compiler); ``'none'`` -> just none; an explicit list is used verbatim
    with none ensured present. No installed veclib -> resolves to ``('none',)`` rather than erroring.

    Ranked PER COMPILER, not once: the libraries are not interchangeable across families -- SVML is
    clang/icx-only, so a gcc-only ranking silently denies llvm cells their best candidate.
    :func:`veclibs_for` then drops each winner from the cells whose compiler cannot use it.
    """
    if list(spec) == ["auto"]:
        winners = []
        for compiler in compilers:
            ranked = [p for p in rank_veclibs(compiler) if p.ok]
            if ranked:
                winners.append(ranked[0].name)
        return tuple(dict.fromkeys(["none", *winners]))
    out = list(dict.fromkeys(spec))
    if "none" not in out:
        out = ["none"] + out
    return tuple(out)


def resolved_axes(args, compilers: Sequence[str] = ("gcc", )) -> Dict:
    parallelism = list(flags.PARALLEL_MODES) if args.parallelism == "both" else [args.parallelism]
    return {
        "opt_modes": args.opt_modes,
        "languages": args.languages,
        "parallelism": parallelism,
        "cost_models": args.cost_models,
        "fp_modes": args.fp_modes,
        "gate": not args.no_gate,
        "matrix_preset": args.matrix_preset,
        "veclibs": resolve_veclibs(args.veclibs, compilers),
        "vectorize": args.vectorize,
        "pluto": args.pluto,
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="TSVC full-matrix job (native / DaCe-cpp / nest-forge sweep)")
    # `foundation` is the hpcagent_bench track and the default: mostly tsvc2 (as `tsvc_2_<key>`)
    # + 56 tsvc2_5 (bare) + 29 that exist nowhere else. NOT a superset -- 16 tsvc2 and 9 tsvc2_5 kernels
    # ship only in the standalone corpus modules -- so those stay selectable and the run reports the gap
    # rather than letting a narrower sweep read as full coverage.
    ap.add_argument("--corpora", nargs="*", default=["foundation"], choices=["tsvc2", "tsvc2_5", "foundation"])
    ap.add_argument("--languages", nargs="*", default=["c", "c++", "fortran"], choices=list(_EMIT))
    ap.add_argument("--opt-modes",
                    nargs="*",
                    default=list(tsvc.OPT_MODES),
                    choices=list(tsvc.OPT_MODES),
                    dest="opt_modes")
    ap.add_argument("--parallelism",
                    default="both",
                    choices=["sequential", "auto-par", "omp-emit", "both"],
                    help="'both' = all of sequential/auto-par/omp-emit; omp-emit uses OUR "
                    "``#pragma omp`` source (only for DaCe-parallel nests).")
    ap.add_argument("--cost-models",
                    nargs="*",
                    default=list(flags.COST_MODELS),
                    choices=list(flags.COST_MODELS),
                    dest="cost_models")
    ap.add_argument("--fp-modes",
                    nargs="*",
                    default=list(flags.REDUCED_FP_MODES),
                    choices=list(flags.REDUCED_FP_MODES),
                    dest="fp_modes")
    ap.add_argument("--matrix-preset",
                    default="full",
                    choices=["full", "lean"],
                    dest="matrix_preset",
                    help="'full' = sweep every cost model on every parallel mode; 'lean' = sweep the cost "
                    "(vectorizer) axis only on the sequential single-core lane and collapse auto-par/"
                    "omp-emit to the compiler default (roughly halves the timing cells, keeps all "
                    "single-core vectorization data). Identical-flag cost cells are always deduped.")
    ap.add_argument("--no-gate", action="store_true", help="skip the strict-ieee bit-exact correctness gate cells")
    ap.add_argument("--veclibs",
                    nargs="*",
                    default=["auto"],
                    help="vector-math library axis: 'auto' (none + the per-device characterized winner), "
                    "'none', or an explicit list from {none, sleef, libmvec, svml}. Only nests whose source "
                    "calls a libm transcendental get the non-none cells.")
    ap.add_argument("--vectorize",
                    action="store_true",
                    help="add the vectorized DaCe lane: coordinate-descent over the multi-dim tile-op "
                    "VectorizeConfig per nest (compiles ~15-30 cells/nest; off by default).")
    ap.add_argument("--pluto",
                    action="store_true",
                    help="add the Pluto polyhedral lane: polycc-transform the emitted scop per nest and "
                    "time it (needs polycc on PATH; each nest records a skip reason when absent).")
    ap.add_argument("--profile-preset",
                    default="PROF",
                    choices=["S", "M", "L", "PROF", "XL"],
                    dest="profile_preset",
                    help="timing size (PROF = >L3 memory-bound; XL = the big confirmation run)")
    ap.add_argument("--compilers", default="auto", help="'auto' or a whitespace list (gcc clang nvc++ icx)")
    ap.add_argument("--strategy", default="skip-taskloops")
    ap.add_argument("--reps", type=int, default=11, help="median-of-N timing reps")
    ap.add_argument("--threads",
                    type=int,
                    default=default_threads(),
                    help="thread count baked into auto-par (gcc -ftree-parallelize-loops=N)")
    ap.add_argument("--compile-jobs",
                    type=int,
                    default=max(1, (os.cpu_count() or 8) // 2),
                    dest="compile_jobs",
                    help="bounded pool of concurrent per-cell compiles within this rank")
    ap.add_argument("--cxx-std", default=os.environ.get("DACE_PERF_CXX_STD", flags.CXX_STD), dest="cxx_std")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="perf_results/tsvc_full")
    ap.add_argument("--tables-only", action="store_true")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.tables_only:
        print(render_tables(out))
        return 0

    toolchains = discover_toolchains(args.compilers)
    if not toolchains:
        print("[tsvc-full] no toolchains discovered (checked PATH + spack); nothing to run")
        return 1
    fortran_by_family = lang_compilers(["fortran"], toolchains).get("fortran", {})
    axes = resolved_axes(args, tuple(dict.fromkeys(t.cc for t in toolchains)))
    print("[tsvc-full] toolchains: " +
          ", ".join(f"{t.name}(cc={Path(t.cc).name},cxx={Path(t.cxx).name if t.cxx else '-'},"
                    f"ftn={Path(fortran_by_family[t.name]).name if t.name in fortran_by_family else '-'})"
                    for t in toolchains))
    print(f"[tsvc-full] axes: opt={axes['opt_modes']} lang={axes['languages']} par={axes['parallelism']} "
          f"cost={axes['cost_models']} fp={axes['fp_modes']} gate={axes['gate']} preset={axes['matrix_preset']} "
          f"profile={args.profile_preset}")

    kernels = [k for corpus in args.corpora for k in tsvc.iter_tsvc_kernels(only=args.only, corpus=corpus)]
    if args.corpora == ["foundation"] and not args.only:
        # say what a foundation-only sweep leaves out, so it cannot be read as full corpus coverage
        gap = tsvc.foundation_coverage_gap()
        for corpus, keys in gap.items():
            if keys:
                print(f"[tsvc-full] NOT covered by foundation: {len(keys)} {corpus} kernel(s) "
                      f"(add --corpora foundation {corpus} to sweep them): {' '.join(keys)}")
    procid, ntasks = rank_and_size()
    mine = my_slice(kernels, procid, ntasks)
    if args.limit:
        mine = mine[:args.limit]
    print(f"[tsvc-full] rank {procid}/{ntasks}: {len(mine)} of {len(kernels)} kernels -> {out}")

    for i, kernel in enumerate(mine):
        workdir = Path(tempfile.mkdtemp(prefix=f"nf_full_{kernel.corpus}_{kernel.key}_"))
        try:
            res = run_kernel(kernel,
                             toolchains,
                             fortran_by_family,
                             args.strategy, {
                                 **axes, "opt_mode": None
                             },
                             args.reps,
                             args.profile_preset,
                             args.threads,
                             args.cxx_std,
                             args.compile_jobs,
                             workdir,
                             dataset_dir=out)
        except Exception as e:  # pragma: no cover - a kernel must never crash the whole rank
            res = {"key": kernel.key, "corpus": kernel.corpus, "skipped": f"crash: {type(e).__name__}: {str(e)[:160]}"}
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
        (out / f"{kernel.corpus}_{kernel.key}.json").write_text(json.dumps(jsonable(res), indent=1))
        print(f"[tsvc-full] ({i + 1}/{len(mine)}) {kernel.corpus}/{kernel.key}: {res.get('skipped', 'ok')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
