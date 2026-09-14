# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Run a compiled kernel in a forked child, so a segfault or runaway loop in freshly-compiled code
cannot take down the parent (a pytest run or a sweep rank).

``os.fork`` shares memory copy-on-write, so nothing is pickled and the child's ``.so`` mapping is
released on ``_exit``. ``work_fn`` must return a small JSON-able dict; a crash, timeout or malformed
result comes back as an ``{"error": ...}`` sentinel.
"""
from __future__ import annotations

import faulthandler
import ctypes
import json
import os
import select
import signal
import time
import warnings
from typing import Callable, Dict

#: OpenMP runtimes whose thread pool must be torn down before a fork (see :func:`pause_openmp_pools`).
#: Probed by the sonames a linked node library actually records in DT_NEEDED.
OMP_RUNTIME_SONAMES = ("libgomp.so.1", "libomp.so.5", "libomp.so", "libiomp5.so", "libnvomp.so")

#: ``omp_pause_resource_t`` (OpenMP 5.0). Both tear the pool down (what buys fork safety); ``hard`` also
#: frees threadprivate data, so ``soft`` is the default.
OMP_PAUSE_SOFT = 1
OMP_PAUSE_HARD = 2

#: name -> ``omp_pause_resource_t`` value, for a config/CLI knob.
OMP_PAUSE_MODES = {"soft": OMP_PAUSE_SOFT, "hard": OMP_PAUSE_HARD}

#: chars of a child exception kept in the ``{"error": ...}`` sentinel. Must clear the longest wrapped
#: subprocess message (``translator`` embeds ``stderr[-2000:]``) or the traceback is lost.
ERROR_CHARS = 4000


def pause_openmp_pools(mode: int = OMP_PAUSE_SOFT) -> None:
    """Tear down the thread pool of every OpenMP runtime ALREADY loaded here, so the coming fork is safe.

    ``fork()`` duplicates only the calling thread, so a child entering a parallel region with the parent's
    pool live hangs forever; libgomp installs no ``pthread_atfork`` handler to recover (libomp does).
    ``RTLD_NOLOAD``: only pause a runtime already mapped -- plain ``CDLL`` would LOAD one this process
    never needed. Best effort but never silent: a missing/refusing symbol warns, since an unhardened fork
    really does deadlock.
    """
    for soname in OMP_RUNTIME_SONAMES:
        try:
            lib = ctypes.CDLL(soname, mode=os.RTLD_NOLOAD)
        except OSError:
            continue  # not loaded in this process: nothing to pause
        try:
            pause = lib.omp_pause_resource_all
        except AttributeError:
            warnings.warn(f"{soname}: no omp_pause_resource_all (pre-OpenMP-5.0 runtime); its thread pool "
                          f"was NOT torn down before the fork -- fork safety for this runtime now rests on "
                          f"its own pthread_atfork handler, if it installs one (libgomp installs none).")
            continue  # best effort, but no longer SILENT: the caller can see the fork was left unhardened
        pause.argtypes = [ctypes.c_int]
        pause.restype = ctypes.c_int
        if pause(mode) != 0:  # e.g. called from within a parallel region: the pool was NOT torn down
            warnings.warn(f"{soname}: omp_pause_resource_all(mode={mode}) returned non-zero; its thread "
                          f"pool was NOT torn down before the fork.")


def quiet_fatal_signals() -> None:
    """In the forked child, drop the faulthandler inherited from pytest: on a segfault it dumps the
    PARENT's stack into the captured output. The parent reports the crash from the child's exit signal.

    No try/except: ``faulthandler.disable()`` returns a bool and has no error path (probed never-enabled,
    closed-file, closed-fd, and inside a fork with a pending ``dump_traceback_later``)."""
    faulthandler.disable()


def run_isolated(work_fn: Callable[[], Dict], timeout: float = 900.0) -> Dict:
    """Run ``work_fn`` in a forked child and return its dict, or an ``{"error": ...}`` sentinel on
    crash / timeout / malformed output. The parent always survives.

    ``timeout`` guards a runaway kernel, so it must exceed the longest legitimate run; large-problem jobs
    pass a larger value explicitly."""
    pause_openmp_pools()  # a pool live across the fork deadlocks the child's first parallel region
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:  # child
        os.close(r)
        quiet_fatal_signals()
        try:
            payload = json.dumps(work_fn())
        except BaseException as e:  # any Python-level failure comes back as an error (a segfault does not)
            # 200 chars cut a wrapped compiler/numpyto traceback off at the first frame's path and left the
            # failure undiagnosable. The parent reads to EOF in a loop, so a longer message costs nothing.
            payload = json.dumps({"error": f"{type(e).__name__}: {str(e)[:ERROR_CHARS]}"})
        try:
            os.write(w, payload.encode())
        finally:
            os.close(w)
            os._exit(0)
    # parent: read the result with a wall-clock deadline, then reap (killing a hung child)
    os.close(w)
    start, buf, timed_out = time.perf_counter(), b"", True
    try:
        while True:
            remaining = timeout - (time.perf_counter() - start)
            if remaining <= 0:
                break  # deadline hit -> timed_out stays True
            ready, _, _ = select.select([r], [], [], remaining)
            if not ready:
                break  # deadline hit
            chunk = os.read(r, 65536)
            if not chunk:  # EOF: the child closed the pipe (finished writing, or died)
                timed_out = False
                break
            buf += chunk
    finally:
        os.close(r)
    if timed_out:
        reaped, status = os.waitpid(pid, os.WNOHANG)
        if reaped == 0:  # genuinely still running -> runaway; kill and reap
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
            return {"error": f"timeout after {timeout:.0f}s (runaway kernel)"}
    else:
        _, status = os.waitpid(pid, 0)  # EOF seen: the child is exiting -> a blocking reap is safe
    if os.WIFSIGNALED(status):  # a segfault etc. never reached the os.write, so buf is empty
        return {"error": f"crashed (signal {os.WTERMSIG(status)})"}
    try:
        return json.loads(buf) if buf else {"error": "child produced no result"}
    except json.JSONDecodeError:
        return {"error": "child produced malformed result (crashed mid-write)"}
