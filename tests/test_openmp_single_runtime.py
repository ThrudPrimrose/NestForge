# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""ONE OpenMP runtime, globally, across every compiler -- the contract ``toolchain.OpenMPRuntime`` and
``toolchain.usable_openmp`` give every owned build.

Every kernel library and the driver must link the SAME OpenMP runtime, so that libraries built by
DIFFERENT compilers share one runtime and ONE thread pool. Left to a bare ``-fopenmp`` each family
links its own default (gcc -> libgomp, clang -> libomp, icx -> libiomp5), which silently puts TWO
runtimes with two thread pools in one process as soon as a sweep spans gcc and clang.

These tests assert on the ARTIFACT, not on the flag list: ``readelf -d`` over a real ``.so`` says which
runtime actually landed in DT_NEEDED. The violation these tests were written for was invisible to every
flag-level assertion, because each family's flags looked correct in isolation.

Most tests here only LINK -- linking is what selects a runtime, and it keeps this process free of the
libgomp fork hazard. The one end-to-end test that must RUN both nests does so in a fresh interpreter,
because a forked child inherits every mapping the worker already had.
"""
import json
import shutil
import subprocess
import sys

import numpy as np
import pytest

from nestforge.build.toolchain import LIBOMP, OPENMP_RUNTIMES, compiler_family, lib_linkable, usable_openmp

#: A minimal nest with an OpenMP region: enough to make the compiler link a runtime.
OMP_SRC = """#include <omp.h>
void kern(double *a, int n) {
  #pragma omp parallel for
  for (int i = 0; i < n; i++) a[i] += 1.0;
}
"""

#: A SECOND, differently-shaped nest (a reduction lowers to ``parallel for reduction``), so a single kernel
#: cannot link one runtime by luck of its shape.
OMP_SRC_REDUCE = """#include <omp.h>
double kern2(const double *a, int n) {
  double s = 0.0;
  #pragma omp parallel for reduction(+:s)
  for (int i = 0; i < n; i++) s += a[i] * 2.0;
  return s;
}
"""

#: The OpenMP runtimes a linked object can name, by DT_NEEDED soname stem.
OMP_SONAMES = ("libgomp", "libomp", "libiomp5")

#: The C compilers of the families nest-forge sweeps.
COMPILERS = ("gcc", "clang", "icx")


def linked_openmp_runtimes(so):
    """The OpenMP runtimes in ``so``'s DT_NEEDED, as soname stems.

    DT_NEEDED records the SONAME of what the linker RESOLVED: distros ship ``libiomp5.so`` as a symlink onto
    LLVM's ``libomp.so``, so the invariant is "exactly ONE runtime, the same for every compiler" rather than
    "the name asked for".
    """
    out = subprocess.run(["readelf", "-d", str(so)], capture_output=True, text=True).stdout
    return {name for name in OMP_SONAMES if f"[{name}.so" in out}


#: The runtime call a compiler emits to OPEN a parallel region: LLVM/Intel ``__kmpc_fork_call``, GNU
#: ``GOMP_parallel``. Its presence is the only proof the region survived compilation.
OMP_FORK_SYMBOLS = ("kmpc_fork", "GOMP_parallel")


def emits_parallel_region(so):
    """True if ``so`` actually CALLS into an OpenMP runtime to open a parallel region.

    ``clang -fopenmp=libgomp`` exits 0, records libgomp in DT_NEEDED -- and emits ZERO fork calls, because
    clang generates only ``__kmpc_*`` and libgomp implements only ``GOMP_*``. The result computes the right
    answer sequentially, so neither a correctness gate nor a DT_NEEDED check can catch it.
    """
    out = subprocess.run(["nm", "-u", str(so)], capture_output=True, text=True).stdout
    return any(sym in out for sym in OMP_FORK_SYMBOLS)


def available_compilers():
    """The C compilers present here. Never empty: the CI runner has gcc and clang."""
    return [cc for cc in COMPILERS if shutil.which(cc)]


def prune_reason(compiler, runtime):
    """Why ``compiler`` cannot link ``runtime`` (ABI or name selection first, then installation), or None."""
    if not runtime.compatible(compiler):
        return f"{compiler} cannot link {runtime.name} (single-runtime contract)"
    if not lib_linkable(runtime.soname, compiler):
        return f"{runtime.name} is not linkable by {compiler} (runtime not installed for it)"
    return None


def build_cell(tmp_path, compiler, runtime, src=OMP_SRC, tag="k"):
    """Link one nest the way the owned build does: the runtime's compile flags before the source, its link
    flags after the object. Returns ``(so, skip_reason)``; exactly one is None."""
    reason = prune_reason(compiler, runtime)
    if reason is not None:
        return None, reason
    csrc = tmp_path / f"{tag}.c"
    csrc.write_text(src)
    so = tmp_path / f"{tag}_{compiler}_{runtime.name}.so"
    cmd = [
        compiler, "-O2", "-fPIC", "-shared", *runtime.compile_flags(compiler),
        str(csrc), *runtime.link_flags(compiler), "-o",
        str(so)
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        pytest.fail(f"{compiler} {runtime.name} failed to link:\n{proc.stderr[-1500:]}")
    # The pragma is in the source, so a cell that links a runtime but opens no region was silently serialized.
    assert emits_parallel_region(so), (f"{compiler} + {runtime.name}: the cell links "
                                       f"{sorted(linked_openmp_runtimes(so))} but emits NO OpenMP fork call")
    return so, None


def test_every_compiler_links_the_same_single_runtime(tmp_path):
    """THE contract: across every available compiler, a cell links exactly ONE OpenMP runtime, and it is
    the SAME one for all of them. Before the fix gcc's bare -fopenmp linked libgomp while clang's linked
    libomp. The global runtime is libomp, what the owned build resolves for gcc, clang and icx."""
    assert all(usable_openmp(cc) is LIBOMP for cc in available_compilers() if compiler_family(cc) in ("gnu", "llvm"))
    seen = {}
    for cc in available_compilers():
        so, _ = build_cell(tmp_path, cc, LIBOMP, tag="a")
        if so is None:
            continue  # a pruned compiler is a recorded skip, not a failure
        rts = linked_openmp_runtimes(so)
        assert len(rts) == 1, f"{cc} linked {len(rts)} OpenMP runtimes ({sorted(rts)}), must be exactly 1"
        seen[cc] = rts
    assert seen, "no compiler could link a runtime -- the matrix would be vacuous"
    distinct = set().union(*seen.values())
    assert len(distinct) == 1, f"the single-runtime contract is violated across compilers: {seen}"


def test_two_different_nests_from_two_compilers_share_one_runtime(tmp_path):
    """Two UNRELATED nests (elementwise map + reduction), each built by a different compiler, as they would be
    when linked into one program. The union over the pair must still be one runtime."""
    compilers = available_compilers()
    assert len(compilers) >= 2, f"needs two compiler families, found {compilers} (setup_apt.sh installs gcc+clang)"
    union, built = set(), {}
    for cc, src, tag in ((compilers[0], OMP_SRC, "map"), (compilers[1], OMP_SRC_REDUCE, "red")):
        so, reason = build_cell(tmp_path, cc, LIBOMP, src=src, tag=tag)
        assert so is not None, f"{cc} cannot link the global runtime: {reason}"
        built[f"{cc}:{tag}"] = sorted(linked_openmp_runtimes(so))
        union |= linked_openmp_runtimes(so)
    assert len(union) == 1, f"two node libraries, two compilers, {len(union)} runtimes: {built}"


@pytest.mark.parametrize("runtime_name", sorted(OPENMP_RUNTIMES))
def test_the_runtime_is_choosable_and_prunes_what_cannot_link_it(tmp_path, runtime_name):
    """The runtime is a KNOB: for a given choice each compiler either links exactly that one runtime or is
    pruned with a reason, never silently falling back to its own default. libgomp is gomp-ABI only, so
    clang cannot link it -- the reason the resolved runtime is libomp, which both families link."""
    runtime = OPENMP_RUNTIMES[runtime_name]
    distinct, decided = set(), {}
    for cc in available_compilers():
        so, reason = build_cell(tmp_path, cc, runtime, tag="c")
        if so is None:
            assert reason, f"{cc} was pruned for {runtime_name} with no reason recorded"
            decided[cc] = f"skip: {reason}"
            continue
        rts = linked_openmp_runtimes(so)
        assert len(rts) == 1, f"{cc} linked {sorted(rts)} for {runtime_name}; must be exactly 1"
        decided[cc] = sorted(rts)
        distinct |= rts
    assert decided, "no compiler was even considered"
    assert len(distinct) <= 1, f"{runtime_name} produced {len(distinct)} distinct runtimes: {decided}"


def test_libgomp_is_pruned_for_llvm_but_kept_for_gnu():
    """Asserted on the compatibility rules rather than this box's toolchain: libgomp implements only GOMP_*,
    clang/flang/icx emit __kmpc_*; libomp carries a GOMP-compat layer and serves both."""
    libgomp, libomp = OPENMP_RUNTIMES["libgomp"], OPENMP_RUNTIMES["libomp"]
    assert libgomp.compatible("gcc") and not libgomp.compatible("clang")
    assert libomp.compatible("gcc") and libomp.compatible("clang")
    with pytest.raises(ValueError, match="libgomp"):  # refused with a reason, never a silently serial flag
        libgomp.compile_flags("clang")
    assert compiler_family("icx") == "llvm" and libomp.compatible("icx")  # icx is clang-based: name-selects libomp


#: Loads BOTH node libraries into one process, runs both nests, and reports what got mapped. Run via EXEC,
#: not fork: a forked child inherits runtimes other tests deliberately loaded.
RUN_BOTH_SRC = '''
import ctypes, json, sys
import numpy as np
so_a, so_b, n = sys.argv[1], sys.argv[2], int(sys.argv[3])

def mapped():
    with open("/proc/self/maps") as fh:
        maps = fh.read()
    return sorted({x for x in ("libgomp", "libomp", "libiomp5") if x + ".so" in maps})

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
    """One nest built by gcc and a different nest built by clang, loaded into ONE process and RUN -- sharing
    a single OpenMP runtime, and computing the right answer."""
    assert shutil.which("gcc") and shutil.which("clang"), \
        f"needs gcc AND clang, found {available_compilers()} (setup_apt.sh installs both)"
    built = {}
    for cc, src, tag in (("gcc", OMP_SRC, "kern"), ("clang", OMP_SRC_REDUCE, "kern2")):
        so, reason = build_cell(tmp_path, cc, LIBOMP, src=src, tag=f"e2e_{tag}")
        assert so is not None, f"{cc} cannot link the global runtime {LIBOMP.name}: {reason}"
        built[cc] = so
        assert linked_openmp_runtimes(so) == linked_openmp_runtimes(built["gcc"]), "cells disagree on the runtime"

    n = 512
    res = run_both_in_a_clean_process(tmp_path, built["gcc"], built["clang"], n)

    assert not res["before"], f"the fresh interpreter already had an OpenMP runtime mapped: {res['before']}"
    assert len(res["runtimes"]) == 1, f"two compilers' node libraries loaded {res['runtimes']} into one process"
    expect_a = np.arange(n, dtype=np.float64) + 1.0
    np.testing.assert_allclose(np.array(res["a"]), expect_a, rtol=0, atol=0)
    np.testing.assert_allclose(res["total"], float(np.sum(expect_a * 2.0)), rtol=1e-12)


def test_a_kmpc_compiler_on_libgomp_would_be_caught_not_silently_serialized(tmp_path):
    """The trap itself: clang emitting kmpc, linked against gomp-only libgomp, and emits_parallel_region
    SEES the serialization. If this ever emits a fork call, libgomp gained a kmpc layer and the prune can
    be revisited -- deliberately."""
    assert shutil.which("clang"), "no clang on PATH (setup_apt.sh installs it)"
    csrc = tmp_path / "mismatch.c"
    csrc.write_text(OMP_SRC)
    so = tmp_path / "mismatch.so"
    proc = subprocess.run(
        ["clang", "-O2", "-fPIC", "-shared", "-fopenmp=libgomp",
         str(csrc), "-o", str(so)],
        capture_output=True,
        text=True)
    assert proc.returncode == 0, f"clang -fopenmp=libgomp failed to link:\n{proc.stderr[-1500:]}"
    assert "libgomp" in linked_openmp_runtimes(so), "expected the mismatch to link libgomp"
    assert not emits_parallel_region(so), ("clang -fopenmp=libgomp emitted a fork call -- libgomp now has a kmpc "
                                           "layer, so the single-runtime prune for kmpc families can be revisited")
