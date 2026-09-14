# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The CPU quick start runs every phase on fuse_diamond and leaves each phase's artifact behind."""

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Tuple

import pytest

from nestforge.build import flags

REPO = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.integration

#: The guide characters ``describe_graph`` draws in front of a line's text.
TREE_GUIDE = re.compile(r"^[|` -]*")
#: A map nest's line: label, bracketed domain, then its reads.
MAP_LINE = re.compile(r"^\S+  \[\w+=[^\]]+\]  reads=")
#: A loop nest's line: label and a bare ``var=start:end`` domain.
LOOP_LINE = re.compile(r"^\S+  \w+=\S+$")


def nest_kinds(tree: str) -> dict:
    """How many map nests and loop nests a saved structure tree shows."""
    texts = [TREE_GUIDE.sub("", line) for line in tree.splitlines()[1:]]
    return {"maps": sum(bool(MAP_LINE.match(t)) for t in texts), "loops": sum(bool(LOOP_LINE.match(t)) for t in texts)}


def lib_paths(node):
    """Every non-empty ``lib_path`` a saved SDFG records, at any depth."""
    if isinstance(node, dict):
        yield from ([node["lib_path"]] if node.get("lib_path") else [])
        for value in node.values():
            yield from lib_paths(value)
    elif isinstance(node, list):
        for value in node:
            yield from lib_paths(value)


@pytest.fixture(scope="module")
def quickstart_run(tmp_path_factory) -> Tuple[Path, str]:
    """The CPU quick start, run once for every test here: its output folder and what it printed."""
    out = tmp_path_factory.mktemp("quickstart")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": os.pathsep.join([str(REPO), *sys.path])}
    command = [sys.executable, str(REPO / "examples" / "quickstart.py"), "--device", "cpu", "--out", str(out)]
    run = subprocess.run(command, capture_output=True, text=True, env=env, cwd=out, timeout=1800)
    assert run.returncode == 0, run.stderr[-4000:]
    return out, run.stdout


def test_cpu_quickstart_prints_one_line_per_phase_and_saves_every_artifact(quickstart_run):
    out, stdout = quickstart_run

    phases = [line.split()[0] for line in stdout.splitlines() if line[:1].isdigit()]

    assert phases == ["0", "1", "2", "3", "4", "5"]
    assert sorted(p.name for p in out.glob("*.sdfg")) == [
        "0-normalize.sdfg",
        "1-shape-kernels.sdfg",
        "2-define-scopes.sdfg",
        "4-optimize-kernels.sdfg",
    ]
    kernel_dir = out / "kernels" / "extcall_0"
    assert sorted(p.name for p in kernel_dir.iterdir()) == ["extcall_0.cpp", "libextcall_0.a"]
    (bound,) = list(lib_paths(json.loads((out / "4-optimize-kernels.sdfg").read_text())))
    assert Path(bound).read_bytes() == (kernel_dir / "libextcall_0.a").read_bytes()
    config = json.loads((out / "5-sweep-configurations.json").read_text())
    assert list(config) == ["extcall_0"]
    assert list(config["extcall_0"]) == ["compiler", "fp_mode", "cost_model", "flags", "time_us"]
    assert config["extcall_0"]["fp_mode"] in flags.FP_LEVELS
    assert config["extcall_0"]["cost_model"] in flags.COST_MODELS
    assert "-O3" in config["extcall_0"]["flags"]
    linking_frames = [p for p in (out / "program").glob("*.cpp") if 'extern "C" void extcall_0(' in p.read_text()]
    assert len(linking_frames) == 1


def test_cpu_quickstart_prints_and_saves_the_kernel_dependency_lines(quickstart_run):
    out, stdout = quickstart_run

    lines = (out / "kernel_deps.txt").read_text().splitlines()

    assert lines[0].startswith("extcall_0: ")
    assert all(f"  {line}" in stdout.splitlines() for line in lines)


def test_cpu_quickstart_shows_four_loop_nests_becoming_one_map_in_its_saved_trees(quickstart_run):
    out, stdout = quickstart_run

    trees = {label: (out / "trees" / f"{label}.txt").read_text() for label in ("0-input", "1-cpf", "2-shaped")}

    assert nest_kinds(trees["0-input"]) == {"maps": 0, "loops": 4}
    assert nest_kinds(trees["1-cpf"]) == {"maps": 1, "loops": 0}
    assert nest_kinds(trees["2-shaped"]) == {"maps": 1, "loops": 0}
    assert all(tree.rstrip("\n") in stdout for tree in trees.values())
