# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Loads hpcagent_bench's benchmark tracks as SDFGs -- the nest-forge kernel corpus. Each kernel's
``_dace.py`` binds hpcagent_bench's precision global, so it must be stamped to fp64 before import."""
from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterator, List, Optional

import numpy as np

import dace

from nestforge.build.arena import resolve_shape
from nestforge.ir.extract import Boundary

if TYPE_CHECKING:
    from types import ModuleType
    from hpcagent_bench.spec import BenchSpec

#: Tracks whose ``_dace.py`` this module generates on demand (gitignored, never committed).
DACE_TRACKS = ("loop_level_reasoning", "scientific_computing", "machine_learning")


def set_precision_fp64() -> None:
    """Fix hpcagent_bench's kernel dtype global to float64 before kernel modules import it."""
    import hpcagent_bench.frameworks.dace_framework as dfw
    dfw.dc_float = dace.float64
    dfw.dc_complex_float = dace.complex128


@dataclass(slots=True)
class CorpusKernel:
    """One optarena kernel that ships a ``@dace.program`` dace impl."""
    short_name: str  # registry key, e.g. "hpc/dense_linear_algebra/gemm/gemm"
    module_path: str  # canonical dotted name, used only as the sys.modules cache key
    dace_file: Path  # the kernel's ``_dace.py`` on disk (source of truth)
    spec: BenchSpec

    def module(self) -> ModuleType:
        """Imports the kernel's ``_dace.py`` by file path, sidestepping ``hpcagent_bench.benchmarks``
        namespace-package resolution (which can bind a stray duplicate ``benchmarks/`` root)."""
        if self.module_path in sys.modules:
            return sys.modules[self.module_path]
        spec = importlib.util.spec_from_file_location(self.module_path, self.dace_file)
        module = importlib.util.module_from_spec(spec)
        sys.modules[self.module_path] = module
        spec.loader.exec_module(module)
        return module

    def program(self) -> dace.frontend.python.parser.DaceProgram:
        """The kernel's entry ``@dace.program``, selected by the manifest's ``func_name`` (falls back
        to the last-defined program, since a module may define helpers before it and a GPU variant after)."""
        set_precision_fp64()
        module = self.module()
        entry = vars(module).get(self.spec.func_name)
        if isinstance(entry, dace.frontend.python.parser.DaceProgram):
            return entry
        programs = [v for v in vars(module).values() if isinstance(v, dace.frontend.python.parser.DaceProgram)]
        if not programs:
            raise LookupError(f"no @dace.program found in {self.dace_file}")
        return programs[-1]  # entry is defined after its helpers

    def to_sdfg(self, simplify: bool = True) -> dace.SDFG:
        return self.program().to_sdfg(simplify=simplify)


def module_path(short_name: str) -> str:
    """Canonical dotted name for a kernel's ``_dace.py`` (a stable sys.modules cache key)."""
    *dirs, module_name = short_name.split("/")
    return f"hpcagent_bench.benchmarks.{'.'.join(dirs)}.{module_name}_dace"


def iter_dace_kernels(track: Optional[str] = None) -> Iterator[CorpusKernel]:
    """Yields every corpus kernel that ships a ``_dace.py`` impl, optionally filtered by track."""
    # deferred: hpcagent_bench imports nestforge at top level
    from hpcagent_bench import autogen
    from hpcagent_bench.spec import KERNELS, BenchSpec
    for short_name in KERNELS:
        if track is not None and not short_name.startswith(f"{track}/"):
            continue
        module_name = short_name.rsplit("/", 1)[-1]
        dace_file = KERNELS[short_name].parent / f"{module_name}_dace.py"
        if not dace_file.exists() and short_name.split("/", 1)[0] in DACE_TRACKS:
            autogen.ensure(short_name, ("dace", ))  # regenerate hpcagent_bench's gitignored _dace.py on demand
        if not dace_file.exists():
            continue
        yield CorpusKernel(short_name=short_name,
                           module_path=module_path(short_name),
                           dace_file=dace_file,
                           spec=BenchSpec.load(short_name))


def materialize_dace_corpus(track: Optional[str] = None) -> None:
    """Generates every missing ``_dace.py`` up front; call once, serially, before a parallel test run
    -- concurrent xdist workers would otherwise race the same non-atomic write."""
    # deferred: hpcagent_bench imports nestforge at top level
    from hpcagent_bench import autogen
    from hpcagent_bench.spec import KERNELS
    for short_name in KERNELS:
        if short_name.split("/", 1)[0] not in DACE_TRACKS:
            continue
        if track is not None and not short_name.startswith(f"{track}/"):
            continue
        autogen.ensure(short_name, ("dace", ))


def dace_kernel_names(track: Optional[str] = None) -> List[str]:
    return [k.short_name for k in iter_dace_kernels(track)]


def preset_sizes(kernel: CorpusKernel, preset: str) -> Dict[str, int]:
    """Concrete shape-symbol sizes for one preset rung, read from the kernel's manifest (skips
    non-int fuzz-spec entries)."""
    from hpcagent_bench.sizing import is_plain_int  # deferred: hpcagent_bench imports nestforge at top level
    rung = kernel.spec.parameters.get(preset, {})
    return {sym: int(size) for sym, size in rung.items() if is_plain_int(size)}


def index_fills(manifest_name: Optional[str],
                boundary: Boundary,
                sizes: Dict[str, int],
                seed: Optional[int] = 0) -> Dict[str, np.ndarray]:
    """Valid-subscript fill values for the nest's manifest-declared integer INDEX arrays, at the SDFG
    descriptor's dtype -- a permutation fill, not the default all-zero uniform-float-cast fill that
    would degrade a gather/scatter to a same-index race once lowered to a ``dace.map``."""
    # deferred: hpcagent_bench imports nestforge at top level
    from hpcagent_bench.initialize import fill_index_array
    from hpcagent_bench.spec import BenchSpec
    if manifest_name is None:
        return {}
    spec = BenchSpec.load(manifest_name)
    if spec.init is None:
        return {}
    rng = np.random.default_rng(seed)
    arrays = boundary.standalone_sdfg.arrays
    fills: Dict[str, np.ndarray] = {}
    for name, declared in sorted(spec.init.dtypes.items()):
        if np.dtype(declared).kind not in "iu" or name not in boundary.inputs:
            continue
        dtype = np.dtype(arrays[name].dtype.type)
        if dtype.kind not in "iu":
            continue  # the manifest calls it an index but the nest holds it as a float: not a subscript
        fills[name] = fill_index_array(resolve_shape(arrays[name].shape, sizes), dtype, rng=rng)
    return fills
