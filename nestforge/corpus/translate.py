# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Extracted kernel -> numpy + manifest -> C/C++/Fortran sources through hpcagent_bench's translator."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List

import yaml

from nestforge.ir.emit_numpy import nest_to_numpy
from nestforge.ir.emit_yaml import manifest_dict
from nestforge.ir.extract import Boundary
from nestforge.corpus.translator import translate

if TYPE_CHECKING:
    from hpcagent_bench.spec import BenchSpec


@dataclass(slots=True)
class Prepared:
    """A kernel written as files the translator consumes."""
    name: str
    numpy_path: Path
    yaml_path: Path
    numpy_source: str
    manifest: Dict
    spec: BenchSpec


def prepare(boundary: Boundary,
            name: str,
            out_dir: os.PathLike,
            sizes: Dict[str, int] = None,
            preset: str = "S") -> Prepared:
    """Write ``<name>_numpy.py`` + ``<name>.yaml`` for one extracted kernel and build its ``BenchSpec``."""
    from hpcagent_bench.spec import BenchSpec  # deferred: hpcagent_bench imports nestforge at top level
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    numpy_source = nest_to_numpy(boundary, fn_name=name)
    manifest = manifest_dict(boundary, name, sizes=sizes, preset=preset)
    numpy_path = out / f"{name}_numpy.py"
    numpy_path.write_text(numpy_source)
    yaml_path = out / f"{name}.yaml"
    yaml_path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    spec = BenchSpec.from_yaml(dict(manifest), source=str(yaml_path))
    return Prepared(name, numpy_path, yaml_path, numpy_source, manifest, spec)


def emit_sources(prep: Prepared, out_dir: os.PathLike, target: str = "c", precision: str = "float64") -> List[Path]:
    """Run the numpy translator; return the generated source files."""
    return translate(prep.spec, prep.numpy_path, prep.name, out_dir, target=target, precision=precision)
