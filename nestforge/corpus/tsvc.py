# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""TSVC corpus adapter: the SDFG source (DaCe ``@dace.program``) and the native-baseline source
(OptArena ``foundation`` track) for the TSVC-2 microkernels.

Two upstreams, joined per kernel by its short name (``s000``, ``vpv``, ...):

  * **SDFG source** -- DaCe's ``performance_regression_jobs/tsvc_corpus.py`` (151 typed
    ``@dace.program`` kernels), loaded by file path since it is a script, not a package; overridable
    with ``NESTFORGE_TSVC_CORPUS``. hpcagent_bench's ``foundation`` track ships no ``_dace.py``.
  * **native baseline** -- ``foundation/tsvc_2_<key>_reference.cpp``, symbol ``<key>_d``: the arena's
    "how well does this compiler auto-vectorize the reference loop" column.

Not a bijection: a kernel with an SDFG but no ``_reference.cpp`` runs without the native column.
"""
from __future__ import annotations

import functools
import importlib.util
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Dict, List, Optional, Tuple

import numpy as np

import yaml

import dace
from dace.transformation.passes.canonicalize.normalize_floor_division import NormalizeFloorDivision
from dace import symbolic
from dace.transformation.auto.auto_optimize import auto_optimize
from dace.transformation.passes.canonicalize import canonicalize
from dace.transformation.passes.symbol_propagation import SymbolPropagation

from hpcagent_bench.initialize import fill_index_array
from hpcagent_bench.spec import KERNELS, BenchSpec

from nestforge.build.arena import resolve_shape
from nestforge.build.toolchain import raw_signature, split_params
from nestforge.corpus.bench import iter_dace_kernels
from nestforge.ir.extract import Boundary, trip_count_symbols
from nestforge.phases.schedule import get_fusion_strategy

#: Shape symbols the sizing logic samples/fixes; every other boundary symbol is a scalar loop
#: parameter (taken from the kernel's registered ``params``) or the corpus multiplier ``S``.
_SHAPE_SYMS = ("LEN_1D", "LEN_2D", "LEN_3D")
#: FALLBACK preset scale, for a symbol the kernel's own manifest does not list and for ``PROF``, which no
#: manifest defines (it sizes one fp64 array ~128 MiB past the GH200 Grace L3 ~114 MB, keeping timings
#: memory-bound; see perf/README_tsvc_full.md). :func:`preset_size` prefers the manifest: hpcagent_bench
#: authors the ladder PER KERNEL from a work/depth model (S ~10 KB, M ~1.3 GB, L ~4 GB, XL ~15 GB working
#: set), so one global table cannot stand in for it -- as a mirror this was 3000x under M and L.
_PRESET = {
    "LEN_1D": {
        "S": 512,
        "M": 32768,
        "L": 131072,
        "PROF": 16777216,
        "XL": 268435455
    },
    "LEN_2D": {
        "S": 64,
        "M": 256,
        "L": 512,
        "PROF": 4096,
        "XL": 1024
    },
    "LEN_3D": {
        "S": 16,
        "M": 48,
        "L": 96,
        "PROF": 256,
        "XL": 200
    },
}
#: Fallback ``(lo, hi)`` sample range per shape symbol when the OptArena yaml carries no preset for
#: it (a 1D kernel's yaml lists only ``LEN_1D``, so a 2D scratch's ``LEN_2D`` falls back here). The
#: ``LEN_2D`` ceiling keeps the square ``LEN_2D**2`` buffers to a few MB.
_SYM_RANGE = {"LEN_1D": (4096, 131072), "LEN_2D": (64, 512)}
#: Fixed size per shape symbol when ``--random-sizes`` is off (the OptArena ``M`` preset for
#: ``LEN_1D``; a moderate square for ``LEN_2D``).
_SYM_FIXED = {"LEN_1D": 32768, "LEN_2D": 256}
#: Preset names never sampled: hpcagent_bench's ``XL`` ``LEN_1D`` is 268435455 (~2 GiB fp64), impractical.
_SKIP_PRESETS = frozenset({"XL"})

#: corpus name -> the DaCe ``performance_regression_jobs`` script that defines its ``@dace.program`` s.
_CORPUS_FILE = {"tsvc2": "tsvc_corpus.py", "tsvc2_5": "tsvc_2_5_corpus.py"}


def corpus_candidates(filename: str) -> List[Path]:
    """Where a corpus script might live: an explicit override, then ``performance_regression_jobs``
    beside each entry of ``dace.__path__`` (package dir or repo root depending on the editable-install
    layout, so both it and its parent are tried)."""
    override = os.environ.get("NESTFORGE_TSVC_CORPUS") if filename == _CORPUS_FILE["tsvc2"] else None
    cands = [Path(override)] if override else []
    for p in dace.__path__:
        base = Path(p)
        cands.append(base / "performance_regression_jobs" / filename)
        cands.append(base.parent / "performance_regression_jobs" / filename)
    return cands


@functools.lru_cache(maxsize=None, typed=True)
def corpus_module(corpus: str = "tsvc2") -> ModuleType:
    """Load a DaCe TSVC corpus script by path (a script, not an importable package). ``tsvc2`` =
    ``tsvc_corpus.py`` (151 kernels), ``tsvc2_5`` = ``tsvc_2_5_corpus.py`` (65 kernels)."""
    if corpus not in _CORPUS_FILE:
        raise ValueError(f"unknown corpus {corpus!r}; known: {sorted(_CORPUS_FILE)}")
    filename = _CORPUS_FILE[corpus]
    src = next((c for c in corpus_candidates(filename) if c.exists()), None)
    if src is None:
        raise FileNotFoundError(
            f"cannot locate DaCe's performance_regression_jobs/{filename} (the {corpus} @dace.program "
            f"source); looked in {[str(c) for c in corpus_candidates(filename)]}. Set "
            "NESTFORGE_TSVC_CORPUS to its path.")
    spec = importlib.util.spec_from_file_location(f"nestforge_{corpus}_corpus", src)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@functools.lru_cache(maxsize=None, typed=True)
def corpus_symbol_values(corpus: str = "tsvc2") -> Dict[str, int]:
    """Concrete values for a corpus' own SCALAR symbols -- multiplier ``S`` plus tile/offset symbols like
    ``T``/``K``/``SSYM`` -- as the corpus script itself declares them.

    Both spellings are read: ``tsvc2`` binds a lone ``S_VALUE``, ``tsvc2_5`` a ``SIZES`` dict. The value
    must be the corpus' own since loop bounds are written to stay in range for it. ``_SHAPE_SYMS`` are
    dropped -- sizing those is the arena's job, so a corpus' ``LEN_1D`` must not shadow the sampled one.

    ``foundation`` has no corpus script: every symbol it binds comes from the kernel's own manifest, which
    :func:`sample_sizes` reads through ``yaml_presets``. Empty here, never an error."""
    if corpus == "foundation":
        return {}
    namespace = vars(corpus_module(corpus))
    values = dict(namespace.get("SIZES", {}))
    if "S_VALUE" in namespace:
        values.setdefault("S", namespace["S_VALUE"])
    return {sym: int(val) for sym, val in values.items() if sym not in _SHAPE_SYMS}


@dataclass(slots=True)
class TsvcKernel:
    """One TSVC kernel: its DaCe ``@dace.program`` SDFG source + its OptArena native baseline."""
    key: str  # short name, e.g. "s000" (the corpus "s000_d_single" with the "_d_single" suffix dropped)
    program: object  # the DaCe ``@dace.program``
    regime: str  # "1d" or "2d"
    params: Dict[str, int]  # scalar loop parameters (e.g. n1, n3) with their registered values
    corpus: str = "foundation"  # source corpus ("foundation" | "tsvc2" | "tsvc2_5")
    tags: frozenset = field(default_factory=frozenset)

    @property
    def foundation_entry(self) -> Optional[Path]:
        """This kernel's OptArena manifest path from the registry, or ``None`` with no foundation entry.

        ``tsvc2`` lives under the ``tsvc_2_<key>`` stem, ``tsvc2_5`` under the bare ``<key>``; both are
        tried (prefixed first) so a ``tsvc2_5`` kernel does not silently fall back to defaults."""
        for stem in (f"tsvc_2_{self.key}", self.key):
            registry_key = f"foundation/{stem}/{stem}"
            if registry_key in KERNELS:
                return Path(KERNELS[registry_key])
        return None

    @property
    def native_cpp(self) -> Optional[Path]:
        """The timing-free C++ baseline, or ``None`` if this kernel has no foundation entry.

        ``_reference.cpp``, not the retired ``_native.cpp``: both were the upstream microkernel with the
        ``time_ns`` instrumentation stripped -- semantically identical for all 151 tsvc_2 kernels, differing
        only in whether the symbol carried the ``_single`` suffix -- so hpcagent_bench collapsed them onto
        the one name its own collector maintains."""
        entry = self.foundation_entry
        if entry is None:
            return None
        cpp = entry.with_name(f"{entry.stem}_reference.cpp")
        return cpp if cpp.exists() else None

    @property
    def native_symbol(self) -> str:
        return f"{self.key}_d"

    @property
    def yaml_path(self) -> Optional[Path]:
        """The kernel's OptArena manifest, or ``None`` when it has no foundation entry."""
        entry = self.foundation_entry
        return entry if entry is not None and entry.exists() else None

    @property
    def bench_name(self) -> Optional[str]:
        """The kernel's OptArena short name (the manifest stem), or ``None`` when it has no manifest."""
        return self.yaml_path.stem if self.yaml_path is not None else None


#: Stem hpcagent_bench files the TSVC-2 kernels under. Stripped so a kernel keeps ONE key whatever corpus
#: it was loaded from: ``--only s221`` and every recorded result key stay valid across the switch. Verified
#: collision-free -- no bare foundation kernel is named like a de-stemmed ``tsvc_2_*`` one.
FOUNDATION_TSVC_STEM = "tsvc_2_"


def foundation_key(name: str) -> str:
    """A foundation registry name reduced to the corpus-independent kernel key."""
    return name[len(FOUNDATION_TSVC_STEM):] if name.startswith(FOUNDATION_TSVC_STEM) else name


def corpus_metadata() -> Dict[str, Tuple[str, Dict[str, int], frozenset]]:
    """``key -> (regime, params, tags)`` from the DaCe TSVC registry, for enriching foundation kernels.

    hpcagent_bench manifests carry problem-size presets but NOT a kernel's scalar loop parameters (s122's
    ``n1``/``n3``, s162's ``k``), and :func:`sample_sizes` raises on a work-deciding symbol it cannot size
    -- so without this the 11 kernels that have such parameters would drop out of a foundation sweep.
    ``regime``/``tags`` come along because they are reported per kernel and would otherwise read as "1d,
    untagged" for every foundation kernel.

    BEST-EFFORT: a checkout without DaCe's corpus script still gets every foundation kernel, and only those
    11 fail loudly at sizing. Returning an empty map is therefore correct, not a silent degradation."""
    try:
        return {k.key: (k.regime, dict(k.params), k.tags) for k in iter_tsvc_kernels(corpus="tsvc2")}
    except (FileNotFoundError, ImportError, AttributeError):
        return {}


def iter_tsvc_kernels(only: Optional[List[str]] = None, corpus: str = "foundation") -> List[TsvcKernel]:
    """Every kernel of a corpus, optionally filtered to the ``only`` short names.

    ``foundation`` (the DEFAULT) is the installed hpcagent_bench track and the only source that ships each
    kernel's manifest and native baseline. It is the default because it is the corpus actually installed
    with the project -- the two below are scripts inside a DaCe checkout, which a bare clone need not have.

    It covers all of ``tsvc2`` but is NOT a superset of both: :func:`foundation_coverage_gap` reports
    three ``tsvc2_5`` kernels missing from it (``ext_war_sym``, ``iv_additive``, ``iv_multiplicative``).
    Ask that function rather than trusting a count written here -- it is the check, and it is what the
    sweep scripts should gate on before telling an operator that one corpus subsumes another.

    ``tsvc2`` ships a ``KERNELS`` registry of descriptors (``s000_d_single`` -> key ``s000``, with
    regime/params/tags); ``tsvc2_5`` ships a ``collect()`` of bare ``@dace.program`` s (key = function
    name, no regime/params metadata)."""
    want = set(only) if only else None
    out: List[TsvcKernel] = []
    if corpus == "foundation":
        # The whole hpcagent_bench track, TSVC included. Sizing, native baselines and manifests all hang
        # off `foundation_entry`, which resolves a bare <key> stem, so nothing below this line is
        # TSVC-specific. Programs are materialized eagerly: measured ~2 ms each, ~0.5 s for the corpus.
        meta = corpus_metadata()
        for kernel in sorted(iter_dace_kernels("foundation"), key=lambda k: k.short_name):
            key = foundation_key(kernel.short_name.rsplit("/", 1)[-1])
            if want is not None and key not in want:
                continue
            regime, params, tags = meta.get(key, ("1d", {}, frozenset()))
            out.append(
                TsvcKernel(key=key, program=kernel.program(), regime=regime, params=params, corpus=corpus, tags=tags))
        return out
    module = corpus_module(corpus)
    if corpus == "tsvc2_5":
        for prog in module.collect():
            key = prog.f.__name__
            if want is not None and key not in want:
                continue
            out.append(TsvcKernel(key=key, program=prog, regime="1d", params={}, corpus=corpus))
        return out
    for k in module.KERNELS:
        key = k.name[:-len("_d_single")] if k.name.endswith("_d_single") else k.name
        if want is not None and key not in want:
            continue
        out.append(
            TsvcKernel(key=key, program=k.program, regime=k.regime, params=dict(k.params), corpus=corpus, tags=k.tags))
    return out


def foundation_coverage_gap() -> Dict[str, List[str]]:
    """corpus -> keys it carries that the ``foundation`` track does not.

    NOT empty: measured today it returns three ``tsvc2_5`` keys (``ext_war_sym``, ``iv_additive``,
    ``iv_multiplicative``). That is exactly why this is a live check rather than a claim -- foundation is
    maintained independently, and a sweep that defaults to it and says nothing reads as full corpus
    coverage while quietly omitting whatever drifted out.

    Keys are compared after :func:`foundation_key` normalisation, so the two sides speak the same names.
    """
    foundation = {k.key for k in iter_tsvc_kernels(corpus="foundation")}
    gap: Dict[str, List[str]] = {}
    for corpus in ("tsvc2", "tsvc2_5"):
        gap[corpus] = sorted(k.key for k in iter_tsvc_kernels(corpus=corpus) if k.key not in foundation)
    return gap


#: The optimization modes applied to a kernel's SDFG *before* the loopnest-splitting pass runs. This is
#: the DaCe-side opt-pipeline axis: each value is a distinct lowering the external compiler then sees.
OPT_MODES = ("simplify-parallel", "canonicalize", "auto-opt")


def build_sdfg(kernel: TsvcKernel, opt_mode: str = "simplify-parallel") -> dace.SDFG:
    """The kernel's SDFG after an optimization mode, ready for the loopnest-splitting pass.

    Pre-split axis:
    - ``"simplify-parallel"`` -- ``simplify`` + ``LoopToMap`` + ``MapFusion`` (V+H) + ``simplify``: the
      reference map-nest form the external compiler vectorizes/tiles.
    - ``"canonicalize"``      -- DaCe's extended-branch statement-level normal form (normalized
      loops/maps, lifted reductions, parallelized where sound).
    - ``"auto-opt"``          -- DaCe's full CPU ``auto_optimize`` pipeline, measured as its own axis
      rather than assumed.

    ``SymbolPropagation`` runs for EVERY mode: the frontend names a derived loop bound with a fresh symbol
    on an inter-state edge (s122's ``n1_minus_1 = n1 - 1``), which the splitter leaves outside the nest as
    an unbindable free argument. DaCe's ``Simplify`` does not include the pass, so without this
    s122/s172/s4114 were silently dropped.
    """
    sdfg = kernel.program.to_sdfg(simplify=True)
    SymbolPropagation().apply_pass(sdfg, {})
    if opt_mode == "simplify-parallel":
        get_fusion_strategy("maximal-fusion")(sdfg)  # Phase 1: fuse everything legal
    elif opt_mode == "canonicalize":
        canonicalize(sdfg, target="cpu")
    elif opt_mode == "auto-opt":
        auto_optimize(sdfg, dace.DeviceType.CPU)
    else:
        raise ValueError(f"unknown opt_mode {opt_mode!r}; expected one of {OPT_MODES}")
    # forced for EVERY mode (only `canonicalize` runs it internally): a residual sympy floor() reaches
    # codegen as an index that truncates term by term
    NormalizeFloorDivision().apply_pass(sdfg, {})
    return sdfg


@functools.lru_cache(maxsize=None, typed=True)
def manifest_presets(path: Path) -> Dict[str, Dict[str, int]]:
    """``{shape_symbol: {preset: size}}`` from an OptArena manifest yaml -- the AUTHORITATIVE ladder.
    Preset NAMES are kept (an unnamed list cannot answer "how big is M here"), and no preset is dropped;
    :func:`yaml_presets` applies ``_SKIP_PRESETS`` where sizes are SAMPLED, which is the only place a
    rung's absence is a policy rather than a fact. Cached per path: the yaml read + parse is re-run for
    every arena cell over the same kernel otherwise."""
    params = yaml.safe_load(path.read_text()).get("parameters", {})
    presets: Dict[str, Dict[str, int]] = {}
    for name, mapping in params.items():
        if not isinstance(mapping, dict):
            continue
        for sym, size in mapping.items():
            presets.setdefault(sym, {})[name] = int(size)
    return presets


def yaml_presets(kernel: TsvcKernel) -> Dict[str, List[int]]:
    """``{shape_symbol: [preset sizes]}`` from the kernel's OptArena yaml, skipping the ``XL`` preset."""
    if kernel.yaml_path is None:
        return {}
    return {
        sym: [size for name, size in rungs.items() if name not in _SKIP_PRESETS]
        for sym, rungs in manifest_presets(kernel.yaml_path).items()
    }


def preset_size(kernel: TsvcKernel, sym: str, preset: str) -> int:
    """Size for one shape symbol at one preset, the kernel's own manifest deciding."""
    if kernel.yaml_path is not None:
        size = manifest_presets(kernel.yaml_path).get(sym, {}).get(preset)
        if size is not None:
            return size
    fallback = _PRESET.get(sym, {}).get(preset)
    if fallback is None:
        raise ValueError(f"kernel {kernel.key!r}: no size for {sym!r} at preset {preset!r} -- neither its "
                         f"manifest ({kernel.yaml_path}) nor the fallback table lists that rung")
    return int(fallback)


def key_seed(key: str) -> int:
    """A stable per-kernel seed offset (so every kernel gets different random sizes, reproducibly).
    ``hash`` is salted per process, so derive from the characters instead."""
    return sum((i + 1) * ord(c) for i, c in enumerate(key)) & 0xFFFF


@functools.lru_cache(maxsize=None, typed=True)
def free_symbols_of(dim) -> frozenset:
    """The free symbol names of one shape-dimension expression, memoized. Keyed on the dimension
    object itself (a sympy expression already, in practice) -- ``.free_symbols`` walks the whole
    expression tree, and the same handful of dims (``LEN_1D``, ``N - 1``, ...) recur across every
    array in a corpus."""
    return frozenset(str(s) for s in symbolic.pystr_to_symbolic(dim).free_symbols)


def shape_symbols(sdfg: dace.SDFG) -> set:
    """Symbols that appear in an array's shape (the ones that actually size a buffer)."""
    return {sym for desc in sdfg.arrays.values() for dim in desc.shape for sym in free_symbols_of(dim)}


def sample_sizes(kernel: TsvcKernel,
                 boundary: Boundary,
                 seed: int = 0,
                 random_sizes: bool = False,
                 preset: Optional[str] = None) -> Dict[str, int]:
    """A concrete value for every boundary symbol.

    A symbol that sizes an array (``LEN_1D``/``LEN_2D``/``LEN_3D``) takes the fixed ``preset`` scale
    (``S``/``M``/``L``/``XL``) when given -- so every compiler/language sees the same size -- else the
    OptArena preset range (random when ``random_sizes``, seeded; else the ``M`` default). A registered
    scalar parameter takes its own value; a corpus-bound scalar (``S``, ``T``, ``K``, ...) takes
    :func:`corpus_symbol_values`.

    A leftover symbol takes ``0`` in exactly two proven-safe cases: it is not an argument of the
    standalone SDFG (unreachable), or it cannot change the TRIP COUNT (absent from
    :func:`trip_count_symbols` and every shape) -- a leaked outer index (s1115) or induction start
    (s123) that picks WHICH element is touched, never HOW MUCH work runs. This does not claim 0 is the
    corpus's real starting value; the lowering is validated, not the kernel's initial state.

    Anything else raises: sizing it 0 would validate vacuously against a degenerate bound.
    """
    rng = np.random.default_rng(seed + key_seed(kernel.key))
    presets = yaml_presets(kernel)
    shape_syms = shape_symbols(boundary.standalone_sdfg)
    corpus_values = corpus_symbol_values(kernel.corpus)
    # what the nest takes, and what can change how much work it does -- together these decide the
    # 0-vs-raise below. Shapes join the trip-count set: both size the work.
    nest_arglist = set(boundary.standalone_sdfg.arglist())
    work_syms = trip_count_symbols(boundary.standalone_sdfg) | shape_syms
    sizes: Dict[str, int] = {}
    for sym in boundary.symbols:
        if sym in kernel.params:
            sizes[sym] = int(kernel.params[sym])
        elif sym in corpus_values and sym not in shape_syms:
            # Guarded on shape_syms: a corpus scalar that also sizes a buffer here is the sampler's to size.
            sizes[sym] = corpus_values[sym]
        elif preset and (sym in _PRESET or sym in presets):
            sizes[sym] = preset_size(kernel, sym, preset)
        elif sym in _SHAPE_SYMS and sym in shape_syms and sym in _SYM_RANGE:
            lo, hi = _SYM_RANGE[sym]
            if presets.get(sym):
                lo, hi = min(presets[sym]), max(presets[sym])
            if sym == "LEN_2D":
                hi = min(hi, _SYM_RANGE["LEN_2D"][1])  # keep LEN_2D**2 buffers bounded regardless of preset
                lo = min(lo, hi)
            sizes[sym] = int(rng.integers(lo, hi + 1)) if random_sizes else min(_SYM_FIXED[sym], hi)
        elif sym in shape_syms:
            # shape symbol outside the known 1D/2D pair with no preset (LEN_3D): small default so a 3D
            # buffer stays modest
            sizes[sym] = _PRESET.get(sym, {}).get("M", 64)
        elif sym not in nest_arglist:
            sizes[sym] = 0  # never passed to the nest: the value provably cannot reach the computation
        elif sym not in work_syms:
            sizes[sym] = 0  # leaked index: selects WHICH element, never HOW MUCH work (see docstring)
        else:
            raise ValueError(f"{kernel.corpus} kernel {kernel.key!r}: boundary symbol {sym!r} is read by the nest and "
                             f"sizes its work (it appears in a loop bound, a map range, an inter-state condition or an "
                             f"array shape), but is not a registered parameter, a corpus symbol "
                             f"({sorted(corpus_values)}) or a known shape symbol, so nothing here knows its value. "
                             f"Sizing it 0 would validate vacuously against a degenerate result -- bind it in the "
                             f"corpus (SIZES/S_VALUE) or in the kernel's params.")
    return sizes


def index_fills(kernel: TsvcKernel,
                boundary: Boundary,
                sizes: Dict[str, int],
                seed: Optional[int] = 0) -> Dict[str, np.ndarray]:
    """Valid-subscript values for the nest's integer INDEX arrays, as the kernel's OptArena manifest
    declares them. Feed the result to :func:`nestforge.arena.make_inputs` as ``given``.

    The manifest declares e.g. ``ip: int32`` as a PERMUTATION of ``[0, N)``, whereas the default
    uniform-float fill cast to int collapses to ALL-ZEROS -- degrading a gather to a cached read of
    ``b[0]`` and turning a conflict-free scatter into a race on ``a[0]`` once lowered to a ``dace.map``.

    Only MANIFEST-declared integer arrays the nest actually READS are filled (an unused
    ``(LEN_2D,LEN_2D)`` fp64 buffer is 2 GiB at ``XL``), at the SDFG descriptor's dtype -- the width the
    compiled code reads across the ABI. ``seed=None`` draws fresh entropy (fuzz); an int pins the fill.
    """
    return index_fills_for_manifest(kernel.bench_name, boundary, sizes, seed=seed)


def index_fills_for_manifest(manifest_name: Optional[str],
                             boundary: Boundary,
                             sizes: Dict[str, int],
                             seed: Optional[int] = 0) -> Dict[str, np.ndarray]:
    """:func:`index_fills` keyed by the OptArena manifest NAME rather than a :class:`TsvcKernel`, for a
    caller iterating OptArena kernels rather than the TSVC corpus. ``None`` -> ``{}``."""
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


#: C base types the arena can bind for a native baseline. Kept equal to the keys of
#: :data:`nestforge.perf.harness.C_BASE` -- that mapping is what turns these strings into ctypes types, so a
#: type accepted here but absent there would KeyError mid-bind. `tests/test_perf_units.py` pins the two.
NATIVE_C_BASE = frozenset({"double", "float", "int64_t", "int"})

#: Type qualifiers stripped before the base type is read. Matched as whole WORDS -- a substring strip turns
#: a parameter named ``const_term`` into ``_term``. Both spellings of restrict: the TSVC baselines are C99
#: (``restrict``) while DaCe-generated code is C++ (``__restrict__``).
QUALIFIERS = re.compile(r"\b(?:const|volatile|restrict|__restrict__|__restrict)\b")

#: Namespace qualification on a base type. The C++ baselines spell the ``<cstdint>`` types both bare
#: (``int64_t``) and qualified (``std::int64_t``) -- the same type and the same ABI, so the qualifier is
#: dropped rather than enumerated into :data:`NATIVE_C_BASE`, which mirrors ``harness.C_BASE``'s ctypes keys.
#: Left unhandled, every gather/scatter kernel (whose index array is the one declared ``std::int64_t``)
#: raised and was dropped from the sweep.
NAMESPACE = re.compile(r"(?:\w+\s*::\s*)+")


def native_signature(cpp_text: str, symbol: str) -> List[Tuple[str, str, bool]]:
    """Parse the ``extern "C"`` baseline signature ``void <symbol>(...)`` into
    ``[(param_name, base_ctype, is_pointer), ...]`` in declaration order.

    Base ctype is one of :data:`NATIVE_C_BASE` -- the types :data:`nestforge.perf.harness.C_BASE` can bind.
    A type outside that set is REFUSED rather than defaulted: the previous fallback made every unrecognised
    declaration an ``int``, so a ``bool`` or ``long`` parameter bound as a 4-byte int with no error anywhere
    -- an ABI mismatch ctypes cannot catch and the numbers cannot reveal.

    ``const`` / ``__restrict__`` are stripped as whole WORDS and the list is split on TOP-LEVEL commas
    (:func:`nestforge.toolchain.split_params`); a substring strip corrupts a parameter named ``const_term``.

    The entry is located by :func:`nestforge.toolchain.raw_signature`, shared with the timing harness, not
    by a local regex. A looser ``\\b<symbol>\\s*\\(`` also matches the DOC COMMENT that names the function:
    ``// argmax_value_d (s314): ...`` sits directly above the declaration in the foundation baselines, so
    the parameter list came back as ``s314`` and the kernel was dropped from every sweep.
    """
    params: List[Tuple[str, str, bool]] = []
    for raw in split_params(raw_signature(cpp_text, symbol, "c")):
        tok = re.sub(QUALIFIERS, "", raw).strip()
        if not tok or tok == "void":
            continue
        is_ptr = "*" in tok
        name = re.split(r"[\s*]+", tok)[-1]
        base = NAMESPACE.sub("", tok[:tok.rfind(name)].replace("*", "").strip())
        if base not in NATIVE_C_BASE:
            raise ValueError(f"native baseline parameter {name!r} of {symbol!r} has C type {base!r}, which the "
                             f"arena cannot bind (known: {sorted(NATIVE_C_BASE)}); binding it as a default width "
                             "would be a silent ABI mismatch")
        params.append((name, base, is_ptr))
    return params
