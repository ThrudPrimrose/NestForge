# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Load hpcagent_bench's own benchmark tracks as SDFGs -- the entire nest-forge kernel corpus.

optarena ships each kernel as ``<name>_numpy.py`` (oracle) + ``<name>.yaml`` (BenchSpec) and, for every
track, a ``<name>_dace.py`` holding a ``@dace.program`` -- import it, ``to_sdfg`` it, feed it to the
lowering pass. ``loop_level_reasoning`` is a superset of TSVC-2 (every ``s###``/``vXX`` kernel lives there
under a ``tsvc_2_<key>`` stem, alongside kernels with descriptive names).

Kernels bind hpcagent_bench's ``dc_float`` precision global at import time, so it must be stamped to fp64
before any kernel module imports.
"""
from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterator, List, Optional

import numpy as np

import dace

from hpcagent_bench import autogen
from hpcagent_bench.initialize import fill_index_array
from hpcagent_bench.sizing import is_plain_int
from hpcagent_bench.spec import KERNELS, BenchSpec

from nestforge.build.arena import resolve_shape
from nestforge.ir.extract import Boundary

if TYPE_CHECKING:
    from types import ModuleType

#: Tracks whose ``_dace.py`` this module materializes on demand (gitignored, never committed --
#: ``autogen.ensure`` regenerates it on demand, at most once per kernel per process).
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
        """Import the kernel's ``_dace.py`` by file path.

        Loading by path (not ``import_module``) sidesteps ``hpcagent_bench.benchmarks`` namespace-package
        resolution, which can non-deterministically bind to a stray duplicate ``benchmarks/`` root.
        """
        if self.module_path in sys.modules:
            return sys.modules[self.module_path]
        spec = importlib.util.spec_from_file_location(self.module_path, self.dace_file)
        module = importlib.util.module_from_spec(spec)
        sys.modules[self.module_path] = module
        spec.loader.exec_module(module)
        return module

    def program(self) -> dace.frontend.python.parser.DaceProgram:
        """The kernel's *entry* ``@dace.program``, selected by the manifest's ``func_name``.

        A module often defines helper programs before it and a ``*_gpu`` variant after, so neither
        "first" nor "last" is reliable; mirrors hpcagent_bench's own ``_import_kernel``.
        """
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
    """Yield every corpus kernel that ships a ``_dace.py`` impl, optionally filtered by track.

    :param track: one of :data:`DACE_TRACKS`, or ``None`` for all.
    """
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
    """Generate every missing ``_dace.py`` for the dace-bearing tracks up front (gitignored, so a fresh
    checkout has none). Safe to call repeatedly. Call ONCE, serially, before a parallel test run:
    :func:`autogen.ensure` writes non-atomically, so concurrent xdist workers must not race the same
    kernel."""
    for short_name in KERNELS:
        if short_name.split("/", 1)[0] not in DACE_TRACKS:
            continue
        if track is not None and not short_name.startswith(f"{track}/"):
            continue
        autogen.ensure(short_name, ("dace", ))


def dace_kernel_names(track: Optional[str] = None) -> List[str]:
    return [k.short_name for k in iter_dace_kernels(track)]


def preset_sizes(kernel: CorpusKernel, preset: str) -> Dict[str, int]:
    """Concrete shape-symbol sizes for one preset rung (``"S"``, ``"M"``, ``"L"``, ...), read from the
    kernel's own manifest ``parameters`` block. A rung entry that is not a plain int (a fuzz spec) is
    skipped -- only ``preset`` rungs carry those, never a named preset."""
    rung = kernel.spec.parameters.get(preset, {})
    return {sym: int(size) for sym, size in rung.items() if is_plain_int(size)}


def index_fills(manifest_name: Optional[str],
                boundary: Boundary,
                sizes: Dict[str, int],
                seed: Optional[int] = 0) -> Dict[str, np.ndarray]:
    """Valid-subscript values for the nest's integer INDEX arrays, as the kernel's manifest declares them.
    Feed the result to :func:`nestforge.build.arena.make_inputs` as ``given``.

    The manifest declares e.g. ``ip: int32`` as a PERMUTATION of ``[0, N)``, whereas the default
    uniform-float fill cast to int collapses to ALL-ZEROS -- degrading a gather to a cached read of
    ``b[0]`` and turning a conflict-free scatter into a race on ``a[0]`` once lowered to a ``dace.map``.

    Only MANIFEST-declared integer arrays the nest actually READS are filled, at the SDFG descriptor's
    dtype -- the width the compiled code reads across the ABI. ``manifest_name`` is a :data:`KERNELS`
    key's stem (``"S"`` for ``kernel.short_name``); ``None`` -> ``{}``. ``seed=None`` draws fresh entropy
    (fuzz); an int pins the fill.
    """
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
