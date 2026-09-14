# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Translate a ``*_numpy.py`` kernel plus its ``BenchSpec`` into C / C++ / Fortran via
hpcagent_bench's ``numpyto`` driver."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from nestforge.build.toolchain import COMPILE_TIMEOUT_S

if TYPE_CHECKING:
    from hpcagent_bench.spec import BenchSpec

DRIVER = "numpyto_common.cli"

__all__ = ["DRIVER", "translate"]


def translate(
    spec: BenchSpec,
    numpy_path: str | Path,
    name: str,
    out_dir: str | Path,
    target: str = "c",
    precision: str = "float64",
) -> list[Path]:
    """Translate the ``*_numpy.py`` kernel at ``numpy_path`` into ``target`` source under ``out_dir``.

    :returns: the generated source files, C then C++ then Fortran.
    """
    from hpcagent_bench import emit_bridge  # deferred: hpcagent_bench imports nestforge at top level

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with emit_bridge.bench_info_tempfile(spec) as bench_info:
        cmd = [
            sys.executable,
            "-m",
            DRIVER,
            "--target",
            target,
            "--kernel",
            str(numpy_path),
            "--bench-info",
            str(bench_info),
            "--out",
            str(out),
            "--precision",
            precision,
        ]
        # Bound the compile so a pathological kernel cannot hang the rank forever.
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=COMPILE_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"numpyto timed out for {name} (target={target}) (ceiling is NF_COMPILE_TIMEOUT)")
    if res.returncode != 0:
        raise RuntimeError(f"numpyto failed for {name} (target={target}):\n{res.stderr[-2000:]}")
    return sorted(out.glob(f"{name}_*.c")) + sorted(out.glob(f"{name}_*.cpp")) + sorted(out.glob(f"{name}_*.f90"))
