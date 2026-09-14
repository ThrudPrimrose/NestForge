# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Smoke tests for the repo's shell scripts and the ruff format gate.

These don't run apt/installs -- they guard that each script is syntactically valid, self-documents
(``--help`` exits 0), rejects a bad flag, and that the tree is ruff-formatted and lint-clean (so a commit
that skipped pre-commit is caught by the tests too)."""

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
SH_SCRIPTS = ["setup_apt.sh"]


@pytest.mark.parametrize("name", SH_SCRIPTS)
def test_script_is_executable_and_syntactically_valid(name):
    p = SCRIPTS / name
    assert p.exists(), f"{name} is missing"
    assert p.stat().st_mode & 0o111, f"{name} is not executable"
    r = subprocess.run(["bash", "-n", str(p)], capture_output=True, text=True)
    assert r.returncode == 0, f"bash -n {name} failed:\n{r.stderr}"


@pytest.mark.parametrize("name", SH_SCRIPTS)
def test_script_help_exits_zero(name):
    r = subprocess.run(["bash", str(SCRIPTS / name), "--help"], capture_output=True, text=True)
    assert r.returncode == 0, f"{name} --help exited {r.returncode}"
    assert r.stdout.strip(), f"{name} --help printed nothing"


@pytest.mark.parametrize("name", SH_SCRIPTS)
def test_script_rejects_unknown_flag(name):
    r = subprocess.run(["bash", str(SCRIPTS / name), "--not-a-real-flag"], capture_output=True, text=True)
    assert r.returncode != 0, f"{name} accepted an unknown flag"


@pytest.mark.parametrize("check", [["format", "--check"], ["check"]], ids=["format", "lint"])
def test_committed_tree_is_ruff_clean(check):
    """The pre-commit ruff hooks would leave the tree unchanged: formatted and lint-clean."""
    r = subprocess.run([sys.executable, "-m", "ruff", *check, str(REPO)], capture_output=True, text=True)
    assert r.returncode == 0, f"ruff {' '.join(check)} failed -- run pre-commit run --all-files\n{r.stdout}{r.stderr}"
