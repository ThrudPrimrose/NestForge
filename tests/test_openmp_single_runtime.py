# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""ONE OpenMP runtime, globally, across every compiler and every lane -- the PARALLEL.md contract.

Every node library and the driver must link the SAME OpenMP runtime, so that libraries built by
DIFFERENT compilers share one runtime and ONE thread pool. Left to a bare ``-fopenmp`` each family
links its own default (gcc -> libgomp, clang -> libomp, icx -> libiomp5), which silently puts TWO
runtimes with two thread pools in one process as soon as a sweep spans gcc and clang.

These tests assert on the ARTIFACT, not on the flag list: ``readelf -d`` over a real ``.so`` says which
runtime actually landed in DT_NEEDED. That distinction is the whole point -- the violation these tests
were written for was invisible to every flag-level assertion, because each family's flags looked
perfectly correct in isolation. Only linking a cell from each compiler and comparing the results shows
the two runtimes.

Most tests here only LINK -- linking is what selects a runtime, so compiling is the whole experiment,
and it keeps this process free of the libgomp fork hazard (an OpenMP region here would poison every
later ``run_isolated`` fork). The one end-to-end test that must RUN both nests does so in a fresh
interpreter via exec, for the same reason plus a second: a forked child inherits every mapping the
worker already had, so it could not tell which runtime the two node libraries themselves pulled in.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from nestforge.build.toolchain import OPENMP_RUNTIMES, compiler_family
from nestforge.build import flags

#: A minimal nest with an OpenMP region: enough to make the compiler link a runtime, which is all that
#: is under test. ``omp-emit`` compiles the pragma as written; ``auto-par`` re-derives it.
OMP_SRC = """#include <omp.h>
void kern(double *a, int n) {
  #pragma omp parallel for
  for (int i = 0; i < n; i++) a[i] += 1.0;
}
"""

#: A SECOND, differently-shaped nest. A single kernel could link one runtime by luck of its shape; the
#: contract is about every node library in the program, so the matrix runs two unrelated nests (an
#: elementwise map and a reduction, which lower to different OpenMP constructs -- ``parallel for`` vs
#: ``parallel for reduction``).
OMP_SRC_REDUCE = """#include <omp.h>
double kern2(const double *a, int n) {
  double s = 0.0;
  #pragma omp parallel for reduction(+:s)
  for (int i = 0; i < n; i++) s += a[i] * 2.0;
  return s;
}
"""

#: The OpenMP runtimes a linked object can name, by DT_NEEDED soname stem.
OMP_SONAMES = ("libgomp", "libomp", "libiomp5", "libnvomp")

#: (family label used by flags.py, C compiler exe). The families nest-forge sweeps.
FAMILIES = (("gnu", "gcc"), ("llvm", "clang"), ("intel", "icx"), ("nvidia", "nvc"))


def linked_openmp_runtimes(so):
    """The OpenMP runtimes in ``so``'s DT_NEEDED, as soname stems.

    DT_NEEDED records the SONAME of what the linker RESOLVED, which is not always the ``-l`` name asked
    for: distros ship ``libiomp5.so`` as a symlink onto LLVM's ``libomp.so`` (they are ABI-compatible),
    so requesting libiomp5 legitimately yields a ``libomp.so.5`` entry. Hence the invariant tested here
    is "exactly ONE runtime, and the same one for every compiler" rather than "the name asked for".
    """
    out = subprocess.run(["readelf", "-d", str(so)], capture_output=True, text=True).stdout
    return {name for name in OMP_SONAMES if f"[{name}.so" in out}


#: The runtime call a compiler emits to OPEN a parallel region: LLVM/Intel ``__kmpc_fork_call``, GNU
#: ``GOMP_parallel``. Its presence is the only proof the region survived compilation.
OMP_FORK_SYMBOLS = ("kmpc_fork", "GOMP_parallel")


def emits_parallel_region(so):
    """True if ``so`` actually CALLS into an OpenMP runtime to open a parallel region.

    Linking a runtime does not mean using it, and the difference is invisible to every other check here.
    ``clang -fopenmp=libgomp`` exits 0 with no warning, records libgomp in DT_NEEDED -- and emits ZERO
    fork calls, because clang generates only ``__kmpc_*`` and libgomp implements only ``GOMP_*``. The
    result links, loads, and computes the RIGHT ANSWER, sequentially. A correctness gate cannot catch it
    (a serial loop gives identical numbers) and neither can a DT_NEEDED check (exactly one runtime is
    linked). In an arena the cell would report sequential timings under a parallel label.

    So: an undefined fork symbol is what distinguishes "parallel" from "parallel-shaped".
    """
    out = subprocess.run(["nm", "-u", str(so)], capture_output=True, text=True).stdout
    return any(sym in out for sym in OMP_FORK_SYMBOLS)


def available_families():
    """The (family, compiler) pairs whose C compiler exists here. Never empty: gcc or clang is present
    on any box that can build a nest at all, and the CI runner has both."""
    return [(fam, cc) for fam, cc in FAMILIES if shutil.which(cc)]


def build_cell(tmp_path, family, compiler, mode, runtime, src=OMP_SRC, tag="k"):
    """Link one sweep cell exactly as a driver does -- ``[exe, *lane_flags, src, -o, so]``, flags BEFORE
    the source. That ordering is load-bearing: it is why a plain ``--as-needed -lomp`` is dropped as
    unused (nothing is undefined yet) and gcc's implicit trailing ``-lgomp`` wins instead. A test that
    put the source first would link the right runtime and prove nothing.

    Returns ``(runtimes, skip_reason)``; exactly one is None.
    """
    f, reason = flags.lane_flags(family, "default-fp", "default", mode, "c", 2, compiler=compiler, openmp=runtime)
    if f is None:
        return None, reason
    csrc = tmp_path / f"{tag}.c"
    csrc.write_text(src)
    so = tmp_path / f"{tag}_{compiler}_{mode}_{runtime.name}.so"
    proc = subprocess.run([compiler, *f, str(csrc), "-o", str(so)], capture_output=True, text=True)
    if proc.returncode != 0:
        pytest.fail(f"{compiler} {mode} {runtime.name} failed to link:\n{proc.stderr[-1500:]}")
    # A cell the matrix OFFERS must actually be parallel. If lane_flags emits a cell whose parallel
    # region was silently dropped, the arena times sequential code under a parallel label -- see
    # emits_parallel_region. omp-emit has the pragma in the source, so it must always survive; auto-par
    # depends on the compiler's analysis, so absence there is a cost-model decision, not a broken cell.
    if mode == "omp-emit":
        assert emits_parallel_region(so), (f"{Path(compiler).name} + {runtime.name}: the cell links "
                                           f"{sorted(linked_openmp_runtimes(so))} but emits NO OpenMP fork call -- "
                                           f"the parallel region was silently dropped, so this cell would be timed "
                                           f"as parallel while running sequentially")
    return linked_openmp_runtimes(so), None


# --- the contract ------------------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["omp-emit", "auto-par"])
def test_every_compiler_links_the_same_single_runtime(tmp_path, mode):
    """THE contract: across every available compiler, a cell links exactly ONE OpenMP runtime, and it is
    the SAME one for all of them.

    Before the fix this failed with {'libgomp', 'libomp'}: gcc's bare -fopenmp linked libgomp while
    clang's linked libomp, so a sweep spanning both put two runtimes and two thread pools in one
    process -- the dual-runtime oversubscription OpenMPRuntime exists to prevent.
    """
    seen = {}
    for family, cc in available_families():
        rts, reason = build_cell(tmp_path, family, cc, mode, flags.DEFAULT_OPENMP_RUNTIME, tag="a")
        if rts is None:
            continue  # a family that cannot link the mandated runtime is a recorded skip, not a failure
        assert len(rts) == 1, f"{cc} {mode} linked {len(rts)} OpenMP runtimes ({sorted(rts)}), must be exactly 1"
        seen[cc] = rts
    assert seen, "no compiler could link the default runtime -- the matrix would be vacuous"
    distinct = set().union(*seen.values())
    assert len(distinct) == 1, (f"the single-runtime contract is violated across compilers: {seen} -- one process "
                                f"loading these node libraries would hold {len(distinct)} runtimes + thread pools")


def test_two_different_nests_from_two_compilers_share_one_runtime(tmp_path):
    """The mixed-compiler/single-runtime pair: two UNRELATED nests (elementwise map + reduction, which
    lower to different OpenMP constructs), each built by a different compiler, as they would be when
    linked into one program. The union over the pair must still be one runtime -- that is what makes
    them safe to load together."""
    fams = available_families()
    if len(fams) < 2:
        pytest.skip(f"needs two compiler families, found {[c for _, c in fams]}")
    (fam_a, cc_a), (fam_b, cc_b) = fams[0], fams[1]
    union, built = set(), {}
    for (fam, cc), src, tag in (((fam_a, cc_a), OMP_SRC, "map"), ((fam_b, cc_b), OMP_SRC_REDUCE, "red")):
        rts, reason = build_cell(tmp_path, fam, cc, "omp-emit", flags.DEFAULT_OPENMP_RUNTIME, src=src, tag=tag)
        if rts is None:
            pytest.skip(f"{cc} cannot link the default runtime: {reason}")
        built[f"{cc}:{tag}"] = sorted(rts)
        union |= rts
    assert len(union) == 1, f"two node libraries, two compilers, {len(union)} runtimes: {built}"


@pytest.mark.parametrize("runtime_name", sorted(OPENMP_RUNTIMES))
def test_the_global_runtime_is_choosable_and_prunes_what_cannot_link_it(tmp_path, runtime_name):
    """The runtime is a KNOB, not a constant: every entry in OPENMP_RUNTIMES can be selected, and for a
    given choice each compiler either links exactly that one runtime or is pruned with a reason. A
    compiler must never silently fall back to its own default -- that is the bug, expressed as an axis.

    Pruning is not a formality: libgomp is gomp-ABI only, so clang (which emits __kmpc_*) cannot link it
    and libgomp can never be the GLOBAL runtime of a gcc+clang sweep. That asymmetry is the reason the
    default is libomp, which both families link.
    """
    runtime = OPENMP_RUNTIMES[runtime_name]
    distinct, decided = set(), {}
    for family, cc in available_families():
        rts, reason = build_cell(tmp_path, family, cc, "omp-emit", runtime, tag="c")
        if rts is None:
            assert reason, f"{cc} was pruned for {runtime_name} with no reason recorded"
            decided[cc] = f"skip: {reason}"
            continue
        assert len(rts) == 1, f"{cc} linked {sorted(rts)} for {runtime_name}; must be exactly 1"
        decided[cc] = sorted(rts)
        distinct |= rts
    assert decided, "no compiler was even considered"
    assert len(distinct) <= 1, f"{runtime_name} produced {len(distinct)} distinct runtimes: {decided}"


def test_libgomp_is_pruned_for_llvm_but_kept_for_gnu():
    """The pruned matrix is CORRECT, not merely non-empty -- asserted on the compatibility rules rather
    than on this box's toolchain, so it means the same everywhere.

    libgomp implements only the GOMP_* ABI; clang/flang/icx emit __kmpc_*. So gnu keeps it and every
    kmpc family drops it. Conversely libomp carries a GOMP-compat layer, so it serves BOTH -- which is
    exactly why it is the default global runtime.
    """
    libgomp, libomp = OPENMP_RUNTIMES["libgomp"], OPENMP_RUNTIMES["libomp"]
    assert libgomp.compatible("gcc") and not libgomp.compatible("clang")
    assert libomp.compatible("gcc") and libomp.compatible("clang")
    # ... and the flag axis honours it: the reason is recorded, not swallowed.
    f, reason = flags.lane_flags("llvm", "default-fp", "default", "omp-emit", "c", 2, compiler="clang", openmp=libgomp)
    assert f is None and reason and "libgomp" in reason
    # the native-runtime-only families accept only their own, whatever the global choice says
    assert not libomp.compatible("nvc") and OPENMP_RUNTIMES["libnvomp"].compatible("nvc")
    assert not libomp.compatible("icc") and OPENMP_RUNTIMES["libiomp5"].compatible("icc")
    assert compiler_family("icx") == "llvm" and libomp.compatible("icx")  # icx is clang-based: name-selects libomp


# --- end to end: two compilers' nests, ONE process, ONE runtime, and the numbers are right -----------
#: Loads BOTH node libraries into one process, runs both nests, and reports what got mapped. Run via
#: EXEC, not fork: the claim is "these two libraries bring in one runtime", and a forked child inherits
#: every mapping the pytest worker already had -- including runtimes other tests deliberately loaded (
#: tests/test_fork_openmp_safety.py loads libgomp on purpose), which would be counted against these two
#: libraries and fail the assertion for something they did not do. A fresh interpreter has no OpenMP
#: runtime mapped until these .so's pull one, so the measurement means exactly what it claims.
RUN_BOTH_SRC = '''
import ctypes, json, sys
import numpy as np
so_a, so_b, n = sys.argv[1], sys.argv[2], int(sys.argv[3])

def mapped():
    with open("/proc/self/maps") as fh:
        maps = fh.read()
    return sorted({x for x in ("libgomp", "libomp", "libiomp5", "libnvomp") if x + ".so" in maps})

before = mapped()
a = np.arange(n, dtype=np.float64)
lib_a, lib_b = ctypes.CDLL(so_a), ctypes.CDLL(so_b)
lib_a.kern.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int]
lib_a.kern.restype = None
lib_b.kern2.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int]
lib_b.kern2.restype = ctypes.c_double
p = a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
lib_a.kern(p, n)                 # gcc's nest:   a[i] += 1.0, in place
total = lib_b.kern2(p, n)        # clang's nest: sum(a[i] * 2.0), over gcc's output
print(json.dumps({"a": a.tolist(), "total": float(total), "runtimes": mapped(), "before": before}))
'''


def run_both_in_a_clean_process(tmp_path, so_a, so_b, n):
    """Run both nests in a FRESH interpreter and return its report (see :data:`RUN_BOTH_SRC`)."""
    script = tmp_path / "run_both.py"
    script.write_text(RUN_BOTH_SRC)
    proc = subprocess.run(
        [sys.executable, str(script), str(so_a), str(so_b), str(n)], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, f"running both nests together failed:\n{proc.stderr[-1500:]}"
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_two_compilers_nests_run_together_on_one_runtime_and_match_numpy(tmp_path):
    """THE end-to-end contract, numerically: one nest built by gcc and a different nest built by clang,
    loaded into ONE process and RUN -- sharing a single OpenMP runtime, and computing the right answer.

    The link-level tests above prove DT_NEEDED holds one runtime; they cannot prove the two libraries
    coexist at RUNTIME or that the shared runtime computes correctly. Only loading both and checking the
    numbers against numpy does, which is the actual claim: node libraries built by different compilers
    share ONE runtime and ONE thread pool, and the results are still right.
    """
    fams = {fam: cc for fam, cc in available_families()}
    if "gnu" not in fams or "llvm" not in fams:
        pytest.skip(f"needs gcc AND clang, found {sorted(fams.values())}")
    runtime = flags.DEFAULT_OPENMP_RUNTIME
    built = {}
    for fam, src, sym in (("gnu", OMP_SRC, "kern"), ("llvm", OMP_SRC_REDUCE, "kern2")):
        cc = fams[fam]
        f, reason = flags.lane_flags(fam, "default-fp", "default", "omp-emit", "c", 2, compiler=cc, openmp=runtime)
        assert f is not None, f"{cc} cannot link the global runtime {runtime.name}: {reason}"
        csrc = tmp_path / f"e2e_{sym}.c"
        csrc.write_text(src)
        so = tmp_path / f"e2e_{fam}_{sym}.so"
        proc = subprocess.run([cc, *f, str(csrc), "-o", str(so)], capture_output=True, text=True)
        assert proc.returncode == 0, f"{cc} failed to link {sym}:\n{proc.stderr[-1500:]}"
        built[fam] = so
        assert linked_openmp_runtimes(so) == linked_openmp_runtimes(built["gnu"]), "cells disagree on the runtime"

    n = 512
    res = run_both_in_a_clean_process(tmp_path, built["gnu"], built["llvm"], n)

    # ONE runtime actually mapped -- the claim, checked in the process that ran both nests.
    assert not res["before"], f"the fresh interpreter already had an OpenMP runtime mapped: {res['before']}"
    assert len(res["runtimes"]) == 1, (f"two node libraries from two compilers loaded {len(res['runtimes'])} OpenMP "
                                       f"runtimes into one process: {res['runtimes']}")
    # ... and the shared runtime computed the right answers (gcc's nest, then clang's over its output).
    expect_a = np.arange(n, dtype=np.float64) + 1.0
    np.testing.assert_allclose(np.array(res["a"]), expect_a, rtol=0, atol=0)
    np.testing.assert_allclose(res["total"], float(np.sum(expect_a * 2.0)), rtol=1e-12)


def test_the_default_runtime_gives_every_compiler_a_REAL_parallel_region(tmp_path):
    """The single-runtime choice must not silently serialize any compiler's cells.

    This is the reason libomp is the default and libgomp is pruned for the kmpc families -- and it is a
    correctness trap, not a link error: ``clang -fopenmp=libgomp`` exits 0, links libgomp, and drops
    every parallel region, because clang emits ``__kmpc_*`` and libgomp implements only ``GOMP_*``. The
    cell would be timed as parallel while running sequentially, and no numeric or DT_NEEDED check sees
    it. So assert the fork call SURVIVES for every available compiler under the global runtime.
    """
    checked = 0
    for family, cc in available_families():
        f, reason = flags.lane_flags(family,
                                     "default-fp",
                                     "default",
                                     "omp-emit",
                                     "c",
                                     2,
                                     compiler=cc,
                                     openmp=flags.DEFAULT_OPENMP_RUNTIME)
        if f is None:
            continue
        csrc = tmp_path / f"real_{cc}.c"
        csrc.write_text(OMP_SRC)
        so = tmp_path / f"real_{cc}.so"
        proc = subprocess.run([cc, *f, str(csrc), "-o", str(so)], capture_output=True, text=True)
        assert proc.returncode == 0, f"{cc}: {proc.stderr[-800:]}"
        assert emits_parallel_region(so), (f"{cc} + {flags.DEFAULT_OPENMP_RUNTIME.name}: parallel region "
                                           f"silently dropped -- would time as parallel, run sequentially")
        checked += 1
    assert checked, "no compiler exercised the default runtime"


def test_a_kmpc_compiler_on_libgomp_would_be_caught_not_silently_serialized(tmp_path):
    """The trap itself, pinned so a future 'just allow libgomp everywhere' change cannot pass unnoticed.

    Directly build the mismatch the matrix forbids (clang emitting kmpc, linked against gomp-only
    libgomp) and prove emits_parallel_region SEES the serialization. If this ever starts emitting a fork
    call, libgomp gained a kmpc layer and the prune can be revisited -- but that must be a deliberate,
    tested decision, never a default that quietly went sequential."""
    if not shutil.which("clang"):
        pytest.skip("no clang")
    csrc = tmp_path / "mismatch.c"
    csrc.write_text(OMP_SRC)
    so = tmp_path / "mismatch.so"
    proc = subprocess.run(
        ["clang", "-O2", "-fPIC", "-shared", "-fopenmp=libgomp",
         str(csrc), "-o", str(so)],
        capture_output=True,
        text=True)
    if proc.returncode != 0:
        pytest.skip("this clang refuses -fopenmp=libgomp outright (also acceptable -- no silent serial cell)")
    assert "libgomp" in linked_openmp_runtimes(so), "expected the mismatch to link libgomp"
    assert not emits_parallel_region(so), ("clang -fopenmp=libgomp emitted a fork call -- libgomp now has a kmpc "
                                           "layer, so the single-runtime prune for kmpc families can be revisited")


def test_lane_flags_names_the_runtime_rather_than_leaving_it_to_the_compiler_default():
    """The mechanism, per family, without needing the compiler installed (pure composition).

    gnu has no ``-fopenmp=<lib>``, so it must pin the runtime at LINK; llvm selects by name. The gnu
    spelling is push-state/--no-as-needed/pop-state and not a plain ``-l``: these flags precede the
    source, where nothing is undefined yet, so ``--as-needed -lomp`` would be dropped as unused and the
    driver's trailing ``-lgomp`` would win. --no-as-needed alone would link BOTH.
    """
    libomp = OPENMP_RUNTIMES["libomp"]
    gnu, _ = flags.openmp_runtime_flags("gcc", "gnu", libomp)
    assert any("--push-state,--no-as-needed,-lomp,--pop-state" in f for f in gnu), gnu
    llvm, _ = flags.openmp_runtime_flags("clang", "llvm", libomp)
    assert "-fopenmp=libomp" in llvm, llvm
    # no compiler to ask -> pure composition, no flags invented
    assert flags.openmp_runtime_flags(None, "gnu", libomp) == ([], None)
