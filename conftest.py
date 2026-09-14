"""Repo-root pytest config.

When NESTFORGE_CI_NO_SKIP is set (CI unit set), a skipped test is a failure: the unit set must run
with zero skips. Locally the env var is unset, so skips behave normally.
"""

import os

import pytest


def pytest_configure(config):
    """Materialise the corpus's gitignored ``_dace.py`` ONCE, before collection. optarena regenerates
    them on demand (they are never committed), so a fresh checkout has none and the corpus tests would
    KeyError. Doing it here -- in the xdist CONTROLLER only, before workers fan out -- keeps parallel
    workers from racing on the non-atomic generate-and-write. Best-effort: if optarena/dace is not
    importable, the corpus tests report that themselves."""
    if vars(config).get("workerinput") is not None:
        return  # xdist worker: the controller already materialised the corpus
    try:
        from nestforge.corpus.bench import materialize_dace_corpus

        materialize_dace_corpus()
    except Exception:
        pass


def pytest_sessionfinish(session, exitstatus):
    if "NESTFORGE_CI_NO_SKIP" not in os.environ:
        return
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is None:
        return
    skipped = reporter.stats.get("skipped", [])
    if not skipped:
        return
    reporter.write_line(f"NESTFORGE_CI_NO_SKIP: {len(skipped)} skipped test(s) not allowed in the unit set:")
    for report in skipped:
        reporter.write_line(f"  SKIPPED {report.nodeid}")
    session.exitstatus = 1


@pytest.fixture(autouse=True)
def reset_extern_lib_env():
    """``ExternLibEnv`` is a PROCESS-GLOBAL class (DaCe resolves library environments by module-level
    name), so its accumulated link line survives from one test into the next. Without this reset the
    static-offload e2e passes or fails purely on FILE ORDER: run after a test that swapped in a shared
    variant, it inherits that test's ``-rpath`` and its "statically in, not loaded" assertion fails.
    CI's ordering happened to be a safe one -- luck, not a guarantee, and any reordering or shuffle
    would have broken it.
    """
    from nestforge.ir.libnode import ExternLibEnv

    ExternLibEnv.reset()
    yield
    ExternLibEnv.reset()
