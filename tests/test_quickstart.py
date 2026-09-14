# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The CPU quick start runs every phase on fuse_diamond and leaves each phase's artifact behind."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from nestforge.build import flags

REPO = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.integration


def test_cpu_quickstart_prints_one_line_per_phase_and_saves_every_artifact(tmp_path):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": os.pathsep.join([str(REPO), *sys.path])}
    command = [sys.executable, str(REPO / "examples" / "quickstart.py"), "--device", "cpu", "--out", str(tmp_path)]

    run = subprocess.run(command, capture_output=True, text=True, env=env, cwd=tmp_path, timeout=1800)

    assert run.returncode == 0, run.stderr[-4000:]
    phases = [line.split()[0] for line in run.stdout.splitlines() if line[:1].isdigit()]
    assert phases == ["0", "1", "2", "3", "4", "5"]
    assert sorted(p.name for p in tmp_path.glob("*.sdfg")) == [
        "0-normalize.sdfg", "1-shape-kernels.sdfg", "2-define-scopes.sdfg", "4-optimize-kernels.sdfg"
    ]
    kernel_dir = tmp_path / "kernels" / "extcall_0"
    assert sorted(p.name for p in kernel_dir.iterdir()) == ["extcall_0.cpp", "libextcall_0.a"]
    assert str(kernel_dir / "libextcall_0.a") in (tmp_path / "4-optimize-kernels.sdfg").read_text()
    config = json.loads((tmp_path / "5-sweep-configurations.json").read_text())
    assert list(config) == ["extcall_0"]
    assert list(config["extcall_0"]) == ["compiler", "fp_mode", "cost_model", "flags", "time_us"]
    assert config["extcall_0"]["fp_mode"] in flags.FP_LEVELS
    assert config["extcall_0"]["cost_model"] in flags.COST_MODELS
    assert "-O3" in config["extcall_0"]["flags"]
    linking_frames = [p for p in (tmp_path / "program").glob("*.cpp") if 'extern "C" void extcall_0(' in p.read_text()]
    assert len(linking_frames) == 1
