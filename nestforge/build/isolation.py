# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Run a compiled kernel in a forked child, so a segfault or runaway loop cannot take down the
parent; a crash, timeout, or malformed result comes back as an ``{"error": ...}`` sentinel."""

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

#: OpenMP runtimes whose thread pool must be torn down before a fork.
OMP_RUNTIME_SONAMES = ("libgomp.so.1", "libomp.so.5", "libomp.so", "libiomp5.so")

#: ``omp_pause_resource_t`` (OpenMP 5.0); ``hard`` also frees threadprivate data, so ``soft`` is default.
OMP_PAUSE_SOFT = 1
OMP_PAUSE_HARD = 2

OMP_PAUSE_MODES = {"soft": OMP_PAUSE_SOFT, "hard": OMP_PAUSE_HARD}

#: must clear the longest wrapped subprocess message or the traceback is lost.
ERROR_CHARS = 4000


def pause_openmp_pools(mode: int = OMP_PAUSE_SOFT) -> None:
    """Tear down every loaded OpenMP runtime's pool before a fork (a live pool deadlocks the child)."""
    for soname in OMP_RUNTIME_SONAMES:
        try:
            lib = ctypes.CDLL(soname, mode=os.RTLD_NOLOAD)  # only pause a runtime already mapped
        except OSError:
            continue
        try:
            pause = lib.omp_pause_resource_all
        except AttributeError:
            warnings.warn(
                f"{soname}: no omp_pause_resource_all (pre-OpenMP-5.0 runtime); its thread pool "
                f"was NOT torn down before the fork -- fork safety for this runtime now rests on "
                f"its own pthread_atfork handler, if it installs one (libgomp installs none)."
            )
            continue
        pause.argtypes = [ctypes.c_int]
        pause.restype = ctypes.c_int
        if pause(mode) != 0:
            warnings.warn(
                f"{soname}: omp_pause_resource_all(mode={mode}) returned non-zero; its thread "
                f"pool was NOT torn down before the fork."
            )


def quiet_fatal_signals() -> None:
    """Drop the pytest-inherited faulthandler so a segfault does not dump the parent's stack."""
    faulthandler.disable()


def run_isolated(work_fn: Callable[[], Dict], timeout: float = 900.0) -> Dict:
    """Run ``work_fn`` in a forked child; returns its dict, or an ``{"error": ...}`` sentinel on
    crash, timeout, or malformed output."""
    pause_openmp_pools()
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(r)
        quiet_fatal_signals()
        try:
            payload = json.dumps(work_fn())
        except BaseException as e:  # any Python-level failure comes back as an error (a segfault does not)
            payload = json.dumps({"error": f"{type(e).__name__}: {str(e)[:ERROR_CHARS]}"})
        try:
            os.write(w, payload.encode())
        finally:
            os.close(w)
            os._exit(0)
    os.close(w)
    start, buf, timed_out = time.perf_counter(), b"", True
    try:
        while True:
            remaining = timeout - (time.perf_counter() - start)
            if remaining <= 0:
                break  # deadline hit -> timed_out stays True
            ready, _, _ = select.select([r], [], [], remaining)
            if not ready:
                break
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
