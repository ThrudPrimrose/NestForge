# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Smoke tests for the repo's shell scripts (setup_apt / format) and the format gate.

These don't run apt/installs -- they guard that each script is syntactically valid, self-documents
(``--help`` exits 0), rejects a bad flag, and that the tree is actually formatted (so a future edit that
skips ``scripts/format.sh`` is caught in CI, not in review)."""
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
SH_SCRIPTS = ["setup_apt.sh", "format.sh"]


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


def test_repo_python_and_cpp_are_formatted():
    """``scripts/format.sh --check`` passes on the committed tree: yapf (python, 120) + clang-format
    (C/C++, 160) both clean. This is the format regression guard the CI gate also runs."""
    assert shutil.which("clang-format"), "clang-format not on PATH (pyproject dev extra installs it)"
    r = subprocess.run(["bash", str(SCRIPTS / "format.sh"), "--check"], capture_output=True, text=True)
    assert r.returncode == 0, f"tree not formatted -- run scripts/format.sh\n{r.stdout}\n{r.stderr}"
