# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``run_isolated`` is the net under every freshly-compiled kernel: the whole point is that a
segfault, a hang or a malformed write in code the compiler produced five seconds ago becomes a RESULT
the sweep can record, never a dead parent.

The happy path is exercised constantly (every emit-and-validate test forks through here). The failure
paths are what the function exists for, and none of them was asserted -- so this pins each one,
including that the parent is still alive afterwards to report it.

A child that dies takes its coverage data with it; that is why these tests assert on the returned
dict rather than on lines executed inside the fork.
"""

import os
import signal
import time

import pytest

from nestforge.build.isolation import ERROR_CHARS, run_isolated


def test_a_result_comes_back_from_the_child_unchanged():
    assert run_isolated(lambda: {"value": 41 + 1, "name": "ok"}, timeout=60) == {"value": 42, "name": "ok"}


def test_an_exception_in_the_child_returns_an_error_instead_of_killing_the_sweep():
    """A kernel that raises during marshalling must not take the parent with it: the sweep records
    the reason and moves to the next cell."""

    def boom():
        raise ValueError("bad argument shape")

    result = run_isolated(boom, timeout=60)
    assert result["error"].startswith("ValueError: bad argument shape")


def test_the_error_text_is_truncated_so_one_bad_kernel_cannot_flood_the_report():
    """A C++ template error pasted into an exception can run to megabytes; the record is a table
    cell, not a log."""

    def verbose():
        raise RuntimeError("x" * (ERROR_CHARS * 3))

    result = run_isolated(verbose, timeout=60)
    assert len(result["error"]) <= ERROR_CHARS + len("RuntimeError: ")


def test_a_segfault_is_reported_as_a_signal_and_the_parent_survives():
    """THE case the fork exists for: freshly-compiled code faults. The signal number is kept because
    it distinguishes a genuine memory fault from a kernel the runtime aborted."""

    def crash():
        os.kill(os.getpid(), signal.SIGSEGV)
        return {"unreachable": True}

    result = run_isolated(crash, timeout=60)
    assert result["error"] == f"crashed (signal {int(signal.SIGSEGV)})"
    assert os.getpid() > 0  # the parent is still here to assert it


def test_a_child_that_exits_without_writing_is_an_error_not_an_empty_success():
    """An empty pipe must not decode as "no differences found" -- an absent result is a failure."""

    def vanish():
        os._exit(0)

    assert run_isolated(vanish, timeout=60) == {"error": "child produced no result"}


def test_a_runaway_child_is_killed_and_reported_rather_than_hanging_the_sweep():
    """The timeout guards an infinite loop in generated code. It must also REAP: a sweep that leaks a
    zombie per runaway kernel exhausts the process table long before it finishes."""

    def runaway():
        time.sleep(120)
        return {}

    start = time.perf_counter()
    result = run_isolated(runaway, timeout=1.0)
    elapsed = time.perf_counter() - start
    assert "timeout after" in result["error"]
    assert elapsed < 30, "the deadline did not fire; it waited for the child instead"
    with pytest.raises(ChildProcessError):
        os.waitpid(-1, os.WNOHANG)  # nothing left unreaped


def test_a_child_returning_something_unserialisable_reports_the_reason():
    """``work_fn`` promises a JSON-able dict. Breaking that promise is the author's bug, and it comes
    back named rather than as a silent empty result."""
    result = run_isolated(lambda: {"array": object()}, timeout=60)
    assert "error" in result and "TypeError" in result["error"]
