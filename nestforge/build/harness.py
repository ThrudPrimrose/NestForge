# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Shared arena infrastructure: rank partitioning, bounded compiles, ABI binding, and result IO.

Used by every perf driver and every ``perf/plot_*.py`` script.
"""
from __future__ import annotations

import ctypes
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from nestforge.build.arena import CTYPE, call_native, scalar_ctype
from nestforge.build.toolchain import COMPILE_TIMEOUT_S, Toolchain, raw_signature

#: Per-kernel *execution* ceiling (s); a runaway kernel would otherwise hold the fork open for the whole
#: job. Override with ``NF_RUN_TIMEOUT``.
RUN_TIMEOUT_S: float = float(os.environ.get("NF_RUN_TIMEOUT", "1800"))

#: base C-type name -> ctypes type, for binding a native baseline signature.
C_BASE = {"double": ctypes.c_double, "float": ctypes.c_float, "int64_t": ctypes.c_int64, "int": ctypes.c_int}

#: rank / size env vars, most specific launcher first (SLURM, OpenMPI, MPICH/Hydra).
RANK_VARS = ("SLURM_PROCID", "OMPI_COMM_WORLD_RANK", "PMI_RANK", "PMIX_RANK")
SIZE_VARS = ("SLURM_NTASKS", "OMPI_COMM_WORLD_SIZE", "PMI_SIZE")


# --- rank self-partition ----------------------------------------------------------------------------
def my_slice(items: List, procid: int, ntasks: int) -> List:
    """This rank's disjoint stride of the kernel list (round-robin balances long/short kernels)."""
    return items[procid::ntasks] if ntasks > 1 else items


def rank_and_size() -> Tuple[int, int]:
    """``(rank, nranks)`` from the launcher env, defaulting to ``(0, 1)`` for a single process.

    :raises RuntimeError: on an asymmetric environment (only one of rank/size set) -- defaulting the
        other silently duplicates or drops work."""
    rank_var = next((v for v in RANK_VARS if os.environ.get(v)), None)
    size_var = next((v for v in SIZE_VARS if os.environ.get(v)), None)
    if rank_var is not None and size_var is None:
        raise RuntimeError(f"launcher set a rank ({rank_var}={os.environ[rank_var]}) but no recognized size variable "
                           f"({list(SIZE_VARS)}); cannot partition safely -- every rank would run the whole list. Set "
                           "the matching size env var, or run without a launcher for a single process.")
    if size_var is not None and rank_var is None:
        # `sbatch --ntasks=4` without srun sets size but not rank; rank=0 would measure 1/N of the corpus.
        raise RuntimeError(f"launcher set a size ({size_var}={os.environ[size_var]}) but no recognized rank variable "
                           f"({list(RANK_VARS)}); cannot partition safely -- this process would measure only "
                           f"1/{os.environ[size_var]} of the list and report it as the whole. Set the matching rank "
                           "env var (srun/mpirun set both), or run without a launcher for a single process.")
    rank = int(os.environ[rank_var]) if rank_var else 0
    size = int(os.environ[size_var]) if size_var else 1
    return rank, max(size, 1)


# --- bounded compile --------------------------------------------------------------------------------
def run_compile(cmd: List[str]) -> Tuple[bool, float, Optional[str]]:
    """Run one compile under :data:`COMPILE_TIMEOUT_S`; ``(ok, microseconds, stderr_tail)``."""
    t0 = time.perf_counter()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=COMPILE_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        dt = (time.perf_counter() - t0) * 1e6
        return False, dt, f"compile timed out after {COMPILE_TIMEOUT_S:.0f}s (NF_COMPILE_TIMEOUT)"
    dt = (time.perf_counter() - t0) * 1e6
    return (p.returncode == 0), dt, (None if p.returncode == 0 else p.stderr[-400:])


# --- ABI binding ------------------------------------------------------------------------------------
def signature_order(text: str, symbol: str, lang: str = "c") -> List[str]:
    """Parameter names of the kernel entry, in declaration order, for the language's syntax.

    The emitted C order (sorted arrays, then symbols) is NOT the manifest ``input_args`` order, so args
    must bind to this or a size lands in a pointer slot. Fortran ``&`` continuations are stripped first."""
    params = raw_signature(text, symbol, lang)
    if lang == "fortran":
        return [a.strip() for a in params.replace("&", " ").split(",") if a.strip()]
    return [p.strip().split()[-1].lstrip("*") for p in params.split(",") if p.strip() and p.strip() != "void"]


def c_argtypes(order: List[str], boundary) -> list:
    """ctypes type per C parameter: array name -> pointer-to-dtype, size/index symbol -> int64, value
    scalar -> its SDFG dtype, matching the translator's signature."""
    sdfg = boundary.standalone_sdfg
    return [
        ctypes.POINTER(CTYPE[np.dtype(sdfg.arrays[a].dtype.type).name]) if a in sdfg.arrays else scalar_ctype(sdfg, a)
        for a in order
    ]


def c_call_args(order: List[str], argtypes: list, work: Dict[str, np.ndarray], sizes: Dict[str, int]) -> list:
    """ctypes arguments for the emitted kernel, in C-signature order: array -> buffer pointer, size
    symbol -> scalar by value. Uses each arg's own type, since a leaked FLOAT scalar is typed
    ``c_double`` by :func:`c_argtypes` and a hardcoded int64 would raise."""
    return [work[a].ctypes.data_as(t) if a in work else t(sizes[a]) for a, t in zip(order, argtypes)]


def call_c(so: Path,
           symbol: str,
           order: List[str],
           argtypes: list,
           boundary,
           inputs,
           sizes,
           reps: int,
           copy_outputs: bool = True):
    """Bind by the C signature order, run once for correctness, then time ``reps`` calls, mutating
    ``inputs`` in place. Callers must run this in a forked child.

    A thin re-export of :func:`nestforge.arena.call_native`, which is the one implementation of
    bind-once / snapshot / rewind-per-rep. They were the same 30 lines twice and drifted: this path was
    missing the per-rep restore entirely, so every in-place kernel it timed was decaying toward denormals.
    ``copy_inputs=False`` is the only difference that remains -- a validating caller reads the results back
    out of the buffers it passed in.
    """
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


def native_symbol(text: str, expected: str) -> str:
    """The first ``extern "C"`` kernel symbol in the baseline source (``expected`` is the ``<key>_d``
    convention, used only as the search hint / fallback)."""
    if re.search(rf"\b{re.escape(expected)}\s*\(", text):
        return expected
    m = re.search(r"\bvoid\s+(\w+)\s*\(", text)
    if not m:
        raise LookupError("no kernel function found in the native baseline source")
    return m.group(1)


def native_setup(so: Path, symbol: str, sig, kernel, buffers: Dict[str, np.ndarray], sizes: Dict[str, int]):
    """Bind ``_reference.cpp`` on ``buffers``; return ``(fn, cargs, ptr_names)``.

    The baselines are pure kernels -- hpcagent_bench's ``scripts/collect_reference_sources.py`` strips the
    upstream ``time_ns`` self-timing param on import -- so every pointer arg resolves to a data buffer and the
    arena times the call itself. Native bounds are independent of nest buffers, so an OOB here is real -- run via
    :func:`nestforge.isolation.run_isolated`.
    :raises KeyError: on an unresolved pointer or scalar arg."""
    pool = {"iterations": 1, "vlen": 8}
    pool.update({s.lower(): int(v) for s, v in sizes.items()})
    pool.update({k.lower(): int(v) for k, v in kernel.params.items()})
    argtypes, ptr_names, cargs = [], [], []
    for name, base, is_ptr in sig:
        ct = C_BASE[base]
        if is_ptr:
            if name not in buffers:
                raise KeyError(f"native pointer arg {name!r} has no matching array buffer")
            argtypes.append(ctypes.POINTER(ct))
            ptr_names.append(name)
            cargs.append(buffers[name].ctypes.data_as(ctypes.POINTER(ct)))
        else:
            if name.lower() not in pool:
                raise KeyError(f"native scalar arg {name!r} unresolved")
            argtypes.append(ct)
            cargs.append(ct(pool[name.lower()]))
    fn = ctypes.CDLL(str(so))[symbol]
    fn.argtypes, fn.restype = argtypes, None
    return fn, cargs, ptr_names


#: language -> (numpyto target, suffix, compiler-exe candidates per family). C and Fortran both emit
#: the same C-ABI ``<key>_fp64`` symbol (Fortran via ``bind(c)``), so ctypes calls are uniform.
LANG_EXES = {
    "c": {
        "target": "c",
        "suffix": ".c",
        "exes": {
            "gcc": ["gcc"],
            "clang": ["clang"],
            "nvhpc": ["nvc"],
            "intel": ["icx"]
        }
    },
    "fortran": {
        "target": "fortran",
        "suffix": ".f90",
        "exes": {
            "gcc": ["gfortran"],
            "clang": ["flang-new", "flang"],
            "nvhpc": ["nvfortran"],
            "intel": ["ifx"]
        }
    },
}


def lang_compilers(languages: List[str], toolchains: List[Toolchain]) -> Dict[str, Dict[str, str]]:
    """``{language: {family: compiler_path}}`` for each discovered family x requested language (e.g.
    gcc compiles C with ``gcc``, Fortran with ``gfortran``). A family missing a compiler is absent."""
    out: Dict[str, Dict[str, str]] = {}
    for lang in languages:
        spec = LANG_EXES[lang]
        per_family: Dict[str, str] = {}
        for tc in toolchains:
            for exe in spec["exes"].get(tc.name, []):  # keyed by family label (gcc/clang/nvhpc)
                path = shutil.which(exe)
                if path:
                    per_family[tc.name] = path
                    break
        out[lang] = per_family
    return out


def fortran_unmunge(order: List[str], names: List[str]) -> List[str]:
    """Map Fortran arg names back to SDFG/size names. Fortran forbids a leading underscore, so the
    translator rewrites it to ``x`` (``__sym_out_i`` -> ``x_sym_out_i``); reverse via the munge map."""
    munge = {("x" + n[1:] if n.startswith("_") else n): n for n in names}
    return [munge.get(a, a) for a in order]


def family_of(name: str) -> str:
    """Toolchain family LABEL (gcc/clang/nvhpc/intel) -> the flag-matrix FP family (gnu/llvm/nvidia/intel).

    Deliberately NOT :func:`nestforge.toolchain.compiler_family`, which looks like the same function and is not:
    that one classifies a compiler EXECUTABLE for its OpenMP ABI and linker, and speaks a different
    vocabulary on purpose -- ``icc`` is ``intel-classic`` there and ``icx`` is ``llvm``, neither of which is
    a key of the FP tables. Feeding one's answer to the other's consumer reads a flag table under the wrong
    family. `tests/test_perf_units.py` pins this function's codomain to the FP tables' actual keys."""
    return {"gcc": "gnu", "clang": "llvm", "nvhpc": "nvidia", "intel": "intel"}.get(name, "gnu")


# --- numbers + result IO ----------------------------------------------------------------------------
def finite(x) -> bool:
    """True only for a real, usable numeric time: a finite int/float. Rejects ``None``, ``inf``, ``nan``."""
    return isinstance(x, (int, float)) and math.isfinite(x)


def median(xs: List[float]) -> float:
    """Median of a non-empty sample (interpolates the two central values on an even-length list)."""
    return float(statistics.median(xs))


def geomean(xs: List[float]) -> Optional[float]:
    """Geometric mean of the finite positive values; ``None`` when there are none."""
    vals = [x for x in xs if finite(x) and x > 0.0]
    if not vals:
        return None
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def fmt_us(x) -> str:
    """A microsecond time for a markdown cell; an em dash for missing / infinite."""
    return "—" if x is None or x == float("inf") else f"{x:.2f}"


def jsonable(obj):
    """Recursively map non-finite floats to ``None`` so ``json.dumps`` emits standard JSON."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [jsonable(v) for v in obj]
    return obj


def load_results(results_dir: Path) -> List[dict]:
    """Every per-kernel JSON in ``results_dir``.

    A driver writes one file per kernel as it goes, so a results directory routinely contains a file that
    is truncated (the job was killed) or still being written (a concurrent rank). Reading them with a bare
    ``json.loads`` takes the whole report down over one bad file, losing every kernel that DID finish --
    which is the opposite of what per-kernel files are for.

    An unreadable file is skipped, but WARNED about, never passed over in silence: a table quietly missing
    three kernels reads exactly like a sweep that only ran the others."""
    rows: List[dict] = []
    unreadable: List[str] = []
    for path in sorted(results_dir.glob("*.json")):
        try:
            rows.append(json.loads(path.read_text()))
        except (json.JSONDecodeError, OSError, ValueError) as e:
            unreadable.append(f"{path.name}: {type(e).__name__}")
    if unreadable:
        warnings.warn(f"{results_dir}: skipped {len(unreadable)} unreadable result file(s) -- the tables below "
                      f"are missing their kernels: {', '.join(sorted(unreadable))}")
    return rows
