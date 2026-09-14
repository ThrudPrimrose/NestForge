# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``run_isolated`` must survive a live OpenMP pool in the parent -- whichever runtime holds it.

``fork()`` duplicates only the calling thread. A child that enters a parallel region while the parent's
pool is live blocks forever on pool threads that no longer exist, and the failure has no symptom beyond
a wall-clock timeout: no crash, no compiler running, the parent simply sits in ``select()``.

libgomp deadlocks exactly this way and installs no ``pthread_atfork`` handler; LLVM's libomp installs
one and recovers. Since the OpenMP runtime is a configurable axis -- libgomp is a valid gnu-only choice
-- ``run_isolated`` must not be safe merely because the default runtime happens to be the forgiving one.
:func:`pause_openmp_pools` is what makes that true, so these tests poison the parent on purpose and
assert the child still runs.
"""

import ctypes
import inspect
import os
import select
import shutil
import subprocess

import numpy as np
import pytest

from nestforge.build.toolchain import OpenMPRuntime, lib_linkable
from nestforge.build.isolation import (
    ERROR_CHARS,
    OMP_PAUSE_MODES,
    OMP_PAUSE_SOFT,
    OMP_RUNTIME_SONAMES,
    pause_openmp_pools,
    run_isolated,
)

OMP_SRC = """#include <omp.h>
void kern(double *a, int n) {
  #pragma omp parallel for
  for (int i = 0; i < n; i++) a[i] += 1.0;
}
"""

N = 4096

#: (runtime, mode) -> does omp_pause_resource_all ACTUALLY tear the thread pool down? MEASURED here, not
#: assumed, by counting threads in /proc/self/task across the call:
#:     libgomp soft 16->1   libgomp hard 16->1   libomp soft 16->16   libomp hard 16->2
#: libomp's SOFT pause is the outlier: it leaves the pool fully up and still returns 0 ("success"), so the
#: return code cannot detect it -- only the thread count can. That cell's fork is safe anyway, but for a
#: different reason (libomp's pthread_atfork handler rebuilds the child's pool), which is precisely why
#: "the child didn't hang" is too weak an observable to pin tear-down with. The default -- libgomp + soft
#: -- is the one that must genuinely tear down, and does.
TEARS_DOWN_POOL = {("gomp", "soft"): True, ("gomp", "hard"): True, ("omp", "soft"): False, ("omp", "hard"): True}


def thread_count():
    """Live threads in THIS process, straight from procfs -- no ``ps`` shell-out. A live OpenMP pool shows
    up here as one thread per core beyond the baseline, so tear-down is directly observable rather than
    inferred from a fork that happened not to hang."""
    return len(os.listdir("/proc/self/task"))


# gcc and both OpenMP runtimes are hard requirements of this file, not optional extras: gcc is the repo's
# default compiler and libgomp is what it links by default, so "no gcc" or "no libgomp" is a broken
# environment to surface, never a skip to hide behind -- a skip here would silently retire THE regression
# this file exists to pin, and would fail CI anyway (the unit set runs under NESTFORGE_CI_NO_SKIP=1).
def build(tmp_path, runtime):
    """A kernel with an OpenMP region, linked against ``runtime`` ("gomp" or "omp").

    libomp is pinned with the SAME flags the owned build uses (:meth:`~nestforge.build.toolchain.OpenMPRuntime.
    link_flags`) rather than a bare ``gcc -print-file-name``. That is not a nicety: on the CI runner gcc does
    not know where libomp lives (libomp-18-dev puts it under /usr/lib/llvm-18/lib, off gcc's path), so the bare
    probe returns "libomp.so" and a hand-rolled check either wrongly skips or wrongly fails. If gcc cannot link
    it, the runtime genuinely is not installed -- a hard requirement missing, which ASSERTS (a red test), never
    a silent skip (the unit set runs under NESTFORGE_CI_NO_SKIP, where a skip fails the session anyway)."""
    assert shutil.which("gcc"), "gcc is required to build the OpenMP kernel this file's regression needs"
    src = tmp_path / "k.c"
    src.write_text(OMP_SRC)
    so = tmp_path / f"k_{runtime}.so"
    extra = []
    if runtime != "gomp":  # gcc's default IS libgomp; anything else must be pinned at link, after the source
        assert lib_linkable(runtime, "gcc"), (
            f"lib{runtime}.so is a hard requirement of this regression but is not installed / linkable by "
            f"gcc here -- install it (e.g. libomp-dev); an absent second runtime is a broken env to surface "
            f"loudly, not a skip to hide behind (the unit set runs under NESTFORGE_CI_NO_SKIP anyway)"
        )
        extra = OpenMPRuntime(name=f"lib{runtime}", soname=runtime).link_flags("gcc")
    proc = subprocess.run(
        ["gcc", "-O2", "-fPIC", "-shared", "-fopenmp", str(src), *extra, "-o", str(so)], capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr[-800:]
    needed = subprocess.run(["readelf", "-d", str(so)], capture_output=True, text=True).stdout
    # DT_NEEDED may show libomp.so.5 for a libiomp5 request (ABI-compat symlink); accept the resolved one.
    assert any(f"[lib{r}.so" in needed for r in (runtime, "omp")), f"expected lib{runtime} in DT_NEEDED, got:\n{needed}"
    return so


#: Worker threads the poisoning region must bring up. Pinned through omp_set_num_threads rather than left
#: to the environment: the repo's standard test prefix exports OMP_NUM_THREADS=1, under which libgomp
#: spawns no workers at all -- there is then no pool to deadlock a forked child on and none to tear down,
#: so every regression in this file passes VACUOUSLY and the 900s-hang guarantee is silently unpinned.
POOL_THREADS = min(4, os.cpu_count() or 1)


def call_kernel(so, n=N, threads=POOL_THREADS):
    """Enter an OpenMP parallel region. Used for BOTH the parent's poisoning and the child's work, so
    the two differ only in which side of the fork they run on."""
    lib = ctypes.CDLL(str(so))
    # resolves through the .so's DT_NEEDED runtime; overrides OMP_NUM_THREADS for this thread
    lib.omp_set_num_threads.argtypes = [ctypes.c_int]
    lib.omp_set_num_threads.restype = None
    lib.omp_set_num_threads(threads)
    lib.kern.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int]
    lib.kern.restype = None
    a = np.zeros(n)
    lib.kern(a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n)
    return a


@pytest.mark.parametrize("runtime", ["gomp", "omp"])
def test_forked_child_runs_openmp_after_the_parent_already_did(tmp_path, runtime):
    """THE regression: parent enters a parallel region, THEN forks a child that enters one.

    Without pause_openmp_pools the libgomp case hangs until the caller's timeout -- which is what took
    out two s000 cells in CI for 900s apiece, with no compiler running and no error to read. A short
    timeout here means a regression reports in seconds rather than wedging the suite.
    """
    so = build(tmp_path, runtime)
    call_kernel(so)  # poison: the parent's pool is now live
    res = run_isolated(lambda: {"total": float(call_kernel(so).sum())}, timeout=60.0)
    assert "error" not in res, f"lib{runtime}: forked child failed after the parent used OpenMP: {res}"
    assert res["total"] == float(N), f"lib{runtime}: child computed {res['total']}, expected {N}"


@pytest.mark.parametrize("runtime", ["gomp", "omp"])
@pytest.mark.parametrize("mode", sorted(OMP_PAUSE_MODES))
def test_the_parent_can_still_use_openmp_after_pausing(tmp_path, runtime, mode):
    """Pausing must not cost the parent anything, under EITHER tear-down mode: a paused runtime
    re-initialises on its next parallel region. Otherwise run_isolated would fix the fork by breaking
    every caller that later runs a kernel itself -- the pool is a cache, and tearing it down is not the
    same as disabling OpenMP."""
    so = build(tmp_path, runtime)
    call_kernel(so)
    pause_openmp_pools(OMP_PAUSE_MODES[mode])
    np.testing.assert_allclose(call_kernel(so), np.ones(N))  # pool rebuilt, still correct


@pytest.mark.parametrize("runtime", ["gomp", "omp"])
@pytest.mark.parametrize("mode", sorted(OMP_PAUSE_MODES))
def test_both_teardown_modes_make_the_fork_safe(tmp_path, runtime, mode):
    """BOTH omp_pause_resource_t options must buy fork safety -- but NOT always by tearing the pool down:
    libgomp drops it under either mode, whereas libomp's SOFT pause leaves the whole pool up (see
    TEARS_DOWN_POOL) and the child runs anyway on libomp's pthread_atfork handler. Where a pause DOES tear
    down, the modes differ only in what else is freed (hard also discards threadprivate data, soft keeps it).

    Tested explicitly because the default is the WEAKER one: soft is chosen so pausing cannot destroy
    state a caller expected to keep, which is only defensible if soft genuinely tears the pool down.
    An unmeasured assumption there would silently return every libgomp fork to a 900s hang.
    """
    so = build(tmp_path, runtime)
    call_kernel(so)  # poison the parent
    pause_openmp_pools(OMP_PAUSE_MODES[mode])

    # fork by hand: run_isolated pauses internally, which would mask whether THIS mode did the work.
    import os

    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(r)
        try:
            os.write(w, b"ok" if float(call_kernel(so).sum()) == float(N) else b"bad")
        finally:
            os.close(w)
            os._exit(0)
    os.close(w)
    ready, _, _ = select.select([r], [], [], 60.0)
    got = os.read(r, 8) if ready else b""
    if not ready:
        os.kill(pid, 9)
    os.waitpid(pid, 0)
    why = "produced nothing (hung on its parallel region, then was killed)" if not got else f"computed {got.decode()}"
    assert got == b"ok", f"lib{runtime} + omp_pause_{mode}: child {why}"


def test_pausing_is_safe_when_no_openmp_runtime_is_loaded():
    """The common case: most calls fork from a process holding no OpenMP runtime at all. Pausing must be
    a silent no-op there and, above all, must not LOAD a runtime to ask it (RTLD_NOLOAD) -- that would
    add a runtime the process never needed, on the very path meant to keep the fork clean."""
    before = mapped_omp()
    pause_openmp_pools()
    pause_openmp_pools()  # idempotent
    assert mapped_omp() == before, "pausing loaded an OpenMP runtime that was not already mapped"


def mapped_omp():
    with open("/proc/self/maps") as fh:
        maps = fh.read()
    return sorted({n for n in ("libgomp", "libomp", "libiomp5") if n + ".so" in maps})


def test_the_pause_drops_the_thread_count_for_the_default_runtime(tmp_path):
    """The DIRECT observable of tear-down -- not the weaker inference 'a forked child did not hang'. After
    a poisoning parallel region libgomp holds a thread per core; pause_openmp_pools() must DROP that count
    (pool torn down), where doing nothing leaves it up. This is the default cell (gomp, soft), which
    TEARS_DOWN_POOL records as a genuine tear-down and on which the whole guarantee rests; the count comes
    straight from /proc/self/task. The libomp+soft outlier (pool left fully up, yet forks safely via its
    atfork handler) is deliberately NOT asserted here: with both runtimes mapped in one interpreter its
    survival cannot be isolated from libgomp's drop, so TEARS_DOWN_POOL only documents it."""
    assert TEARS_DOWN_POOL[("gomp", "soft")], "the default cell must be a genuine tear-down"
    so = build(tmp_path, "gomp")
    call_kernel(so)  # poison: libgomp's pool is now POOL_THREADS deep
    busy = thread_count()
    # A pool that never came up makes "the count dropped" unfalsifiable, and the drop is the ONLY direct
    # observable of tear-down -- assert the precondition rather than let the test pass on an empty pool.
    assert busy > 1, f"the poisoning region brought up no worker threads (count {busy}); nothing to tear down"
    assert thread_count() == busy, "the thread count moved with no pause -- the measurement is not stable"
    pause_openmp_pools()  # default mode is soft (asserted separately)
    assert thread_count() < busy, f"(gomp, soft): pool NOT torn down, thread count stayed at {busy}"


def test_the_default_pause_mode_is_soft():
    """SOFT is the DEFAULT on purpose: the weakest reset that still buys fork safety, because it KEEPS
    threadprivate data where hard destroys it. Assert the default ITSELF -- not merely that soft works
    when asked for explicitly -- so a silent flip of the default to hard (which would discard caller state
    on every fork) is caught here, not later in some caller's lost threadprivate buffer."""
    default = inspect.signature(pause_openmp_pools).parameters["mode"].default
    assert default == OMP_PAUSE_SOFT, (
        f"pause_openmp_pools default mode is {default!r}, expected soft ({OMP_PAUSE_SOFT}) -- soft is the "
        f"weakest reset that still buys fork safety and must remain the default"
    )


def test_a_mapped_runtime_without_the_pause_symbol_is_warned_not_silent(monkeypatch):
    """The best-effort skip must be VISIBLE. Simulate a runtime that is mapped but predates OpenMP 5.0 (no
    omp_pause_resource_all): pause_openmp_pools must WARN that it left that runtime's pool up, never pass
    over it in silence -- a silent skip hides exactly the condition under which a forked child deadlocks.
    A plain object() stands in for the mapped-but-symbol-less library."""

    def fake_cdll(name, mode=0):
        if name == OMP_RUNTIME_SONAMES[0]:
            return object()  # mapped, but no omp_pause_resource_all attribute -> the AttributeError branch
        raise OSError("not loaded in this process")

    monkeypatch.setattr(ctypes, "CDLL", fake_cdll)
    with pytest.warns(UserWarning, match="omp_pause_resource_all"):
        pause_openmp_pools()


def test_a_child_exception_survives_the_pipe_intact():
    """The error sentinel must carry the whole wrapped message. ``translator`` raises RuntimeError with
    ``stderr[-2000:]`` embedded; a 200-char cap sliced that off at the first traceback frame's path and
    left an e2e failure with no evidence of what the subprocess actually said."""
    tail = "LAST-FRAME-MARKER"
    long_message = "x" * 2500 + tail

    def boom():
        raise RuntimeError(long_message)

    res = run_isolated(boom, timeout=60.0)
    assert tail in res["error"], f"message truncated to {len(res['error'])} chars -- the tail is the diagnosis"
    assert len(long_message) <= ERROR_CHARS  # the cap must clear translator's own 2000-char stderr slice
