# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Native surface for hpcagent_bench's **numpy translator**: turn a ``*_numpy.py`` kernel plus its
``BenchSpec`` manifest into C / C++ / Fortran source via hpcagent_bench's ``numpyto`` driver.

Wrapping it here (alongside :mod:`nestforge.corpus.bench`) keeps the rest of nest-forge depending on
``nestforge.*`` rather than reaching into the optarena dependency directly.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import List, Union

from nestforge.build.toolchain import COMPILE_TIMEOUT_S

from hpcagent_bench import emit_bridge
from hpcagent_bench.spec import BenchSpec

#: hpcagent_bench's numpy -> {C, C++, Fortran} translator CLI entry point.
DRIVER = "numpyto_common.cli"

__all__ = ["BenchSpec", "DRIVER", "translate"]


def translate(spec: BenchSpec,
              numpy_path: Union[str, Path],
              name: str,
              out_dir: Union[str, Path],
              target: str = "c",
              precision: str = "float64") -> List[Path]:
    """Translate the ``*_numpy.py`` kernel at ``numpy_path`` into ``target`` source under ``out_dir``.

    :param spec: the kernel's :class:`BenchSpec` (shapes, dtypes, signature).
    :param name: the kernel base name (the generated files are ``<name>_<variant>.<ext>``).
    :returns: the generated source files, C then C++ then Fortran.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with emit_bridge.bench_info_tempfile(spec) as bench_info:
        cmd = [
            sys.executable, "-m", DRIVER, "--target", target, "--kernel",
            str(numpy_path), "--bench-info",
            str(bench_info), "--out",
            str(out), "--precision", precision
        ]
        # Bound the numpyto AOT translate+compile so a pathological kernel can't hang the rank
        # forever (matches toolchain.run's NF_COMPILE_TIMEOUT ceiling); a timeout is just a translate
        # failure -> caller records the cell as errored and continues.
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=COMPILE_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"numpyto timed out for {name} (target={target}) "
                               f"(ceiling is NF_COMPILE_TIMEOUT)")
    if res.returncode != 0:
        raise RuntimeError(f"numpyto failed for {name} (target={target}):\n{res.stderr[-2000:]}")
    return (sorted(out.glob(f"{name}_*.c")) + sorted(out.glob(f"{name}_*.cpp")) + sorted(out.glob(f"{name}_*.f90")))
