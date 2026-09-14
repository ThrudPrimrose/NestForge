# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Emit a standalone numpy/python kernel from an extracted nest.

Walks a state's dataflow topologically: library nodes -> the numpy op (via
:mod:`nestforge.ir.emit_libnode`), map scopes -> ``for`` loops with the tasklet body inlined, free
tasklets -> their Python code inlined.

Memory model is **C-style**: the kernel allocates nothing. Every array -- inputs, outputs, the DaCe
``__return`` value, scratch transients -- is a pre-allocated buffer *parameter* written in place
(``name[:] = ...``); only scalar transients become plain python locals. There is no ``return``. Each
array's data name doubles as its python variable. Unsupported constructs raise
:class:`UnsupportedNest` rather than mis-emit silently.
"""
from __future__ import annotations

import ast
import atexit
import copy
import functools
import inspect
import hashlib
import importlib.util
import re
import shutil
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Callable, Dict, List, Mapping, Optional

import numpy
import sympy

import dace
from dace import symbolic
from dace.frontend.operations import detect_reduction_type as _detect_reduction_type
from dace.sdfg import nodes
from dace.sdfg.state import BreakBlock, ConditionalBlock, ContinueBlock, LoopRegion, ReturnBlock
from dace.sdfg.utils import dfs_topological_sort

from nestforge.ir.emit_libnode import (UnsupportedLibraryNode, emit_library_node, index_str, is_scalar, read_expr,
                                    scalar_local, write_lhs)
from nestforge.ir.extract import Boundary

try:
    from dace.transformation.interstate.expand_nested_sdfg_inputs import ExpandNestedSDFGInputs
except ImportError:  # the pass ships only on the DaCe `extended` branch nest-forge targets
    ExpandNestedSDFGInputs = None


class UnsupportedNest(Exception):
    """The nest uses a construct the numpy emitter does not handle."""


def access(sdfg: dace.SDFG, name: str, subset: dace.subsets.Range) -> str:
    """Element access: a scalar-transient local ``name``, else the indexed buffer element ``name[idx]``.

    A scalar transient is a plain local; every other container (arrays and passed-in scalars alike)
    is a buffer parameter and is indexed.
    """
    if scalar_local(sdfg, name):
        return name
    return f"{name}[{index_str(subset)}]"


def connector_pattern(conn_expr: Dict[str, str]) -> Optional[re.Pattern]:
    """The whole-word alternation pattern matching every connector name in ``conn_expr``, or ``None``
    when there is nothing to substitute. Compiled ONCE per tasklet (:func:`tasklet_lines` builds it
    before its per-line loop) rather than once per code line -- the pattern depends only on the
    connector set, not on the line being rewritten."""
    if not conn_expr:
        return None
    return re.compile(r"\b(" + "|".join(re.escape(c) for c in sorted(conn_expr, key=len, reverse=True)) + r")\b")


def sub_connectors(code: str, conn_expr: Dict[str, str], pattern: Optional[re.Pattern] = None) -> str:
    """Replace whole-word connector tokens in a tasklet's Python code with their expressions.

    Single-pass substitution, so a replacement expression containing another connector's name is not
    itself re-substituted. ``pattern`` may be a precomputed :func:`connector_pattern`; it is derived
    from ``conn_expr`` when omitted.
    """
    if not conn_expr:
        return code
    pattern = pattern if pattern is not None else connector_pattern(conn_expr)
    return pattern.sub(lambda m: conn_expr[m.group(0)], code)


#: DaCe dtype cast (``dace.<name>(x)``) -> the numpy scalar constructor that spells it. Restricted to
#: the fixed-width dtypes so a non-dtype ``dace.<attr>`` (``dace.math.sqrt``, ``dace.define_local``) is
#: never blindly rewritten to a nonexistent ``np.<attr>``; ``bool`` maps to ``np.bool_`` (``np.bool``
#: was removed in NumPy 2).
_DACE_DTYPES = {
    "bool": "np.bool_",
    "int8": "np.int8",
    "int16": "np.int16",
    "int32": "np.int32",
    "int64": "np.int64",
    "uint8": "np.uint8",
    "uint16": "np.uint16",
    "uint32": "np.uint32",
    "uint64": "np.uint64",
    "float16": "np.float16",
    "float32": "np.float32",
    "float64": "np.float64",
    "complex64": "np.complex64",
    "complex128": "np.complex128",
}
_DACE_CAST = re.compile(r"\bdace\.(" + "|".join(_DACE_DTYPES) + r")\b")
#: a BARE dtype cast (``int64(mats_index)``) as ``symbolic.symstr`` renders a DaCe typecast inside an
#: array subscript -- no ``dace.``/``np.`` prefix, so :data:`_DACE_CAST` never sees it and the emitted
#: index would raise ``NameError: name 'int64' is not defined``. The lookbehind skips a qualified
#: ``np.int64(`` / ``dace.int64(`` and any ``x.int64(`` attribute, matching only the standalone call.
_BARE_CAST = re.compile(r"(?<![\w.])(" + "|".join(_DACE_DTYPES) + r")\s*\(")

#: bare math intrinsic (as DaCe exposes it in tasklet code) -> the numpy function that computes it.
_MATH_INTRINSICS = {
    "sqrt": "np.sqrt",
    "cbrt": "np.cbrt",
    "exp": "np.exp",
    "exp2": "np.exp2",
    "expm1": "np.expm1",
    "log": "np.log",
    "log2": "np.log2",
    "log10": "np.log10",
    "log1p": "np.log1p",
    "sin": "np.sin",
    "cos": "np.cos",
    "tan": "np.tan",
    "asin": "np.arcsin",
    "acos": "np.arccos",
    "atan": "np.arctan",
    "atan2": "np.arctan2",
    "sinh": "np.sinh",
    "cosh": "np.cosh",
    "tanh": "np.tanh",
    "floor": "np.floor",
    "ceil": "np.ceil",
    "fabs": "np.abs",
    "sign": "np.sign",
}
_INTRINSIC_CALL = re.compile(r"(?<![\w.])(" + "|".join(_MATH_INTRINSICS) + r")(?=\s*\()")

#: DaCe sympy user-functions -- ``symstr`` renders a subset index / map bound in function form because
#: sympy has no operator for them -> the numpy/python expression computing the same integer value.
#: ``int_floor``/``int_ceil`` are deliberately ABSENT: they are already the exact spelling both ends
#: want, and expanding them to ``//`` loses that. The C translator lowers a ``//`` back to ``int_floor``
#: anyway (and has no ``//`` for ceil at all, so the expansion had to be open-coded), while dace itself
#: reads a ``//`` back as ``sympy.floor``, which distributes over a sum and truncates each term on its
#: own. Left as calls, both are resolved by name: :func:`load_emitted` binds them in the emitted module's
#: namespace, and the C prelude defines them as type-dispatching macros.
#: ``Max``/``Min`` are VARIADIC in sympy and render as scalar index/bound expressions, so the Python
#: builtins are the right target: they keep exact integer semantics (a numpy scalar would leak into
#: ``range()`` and array subscripts). ``apply_call`` passes every parsed argument, so ``*a`` handles any
#: arity. ``__builtins__`` is always present in a module namespace, so ``max``/``min``/``abs`` resolve
#: without importing anything.
_USERFUNC_REWRITES = {
    # ``int_floor`` exists because sympy mis-simplifies a floor division, NOT because python needs a
    # helper for it: ``//`` is already floored for both signs. Emitting the operator keeps the numpy
    # portable -- a translator reads ``ast.FloorDiv`` and lowers it with its own correct helper
    # (numpyto intercepts it into an ``int_floor`` macro), where a bare CALL would be an unknown name.
    # ``int_ceil`` has no operator and stays a call.
    "int_floor": lambda a, b: f"(({a}) // ({b}))",
    "ipow": lambda a, b: f"(({a}) ** ({b}))",
    "Mod": lambda a, b: f"(({a}) % ({b}))",
    "Max": lambda *a: f"max({', '.join(a)})",
    "Min": lambda *a: f"min({', '.join(a)})",
    "Abs": lambda a: f"abs({a})",
}


def int_floor(a: int, b: int) -> int:
    """``floor(a / b)`` -- python ``//`` is already floored for both signs."""
    return a // b


def int_ceil(a: int, b: int) -> int:
    """``ceil(a / b)``, sign-robust (``== (a + b - 1) // b`` for ``b > 0``)."""
    return -((-a) // b)


#: The names an emitted kernel calls but does not define -- every load of an emitted module must bind
#: them: ``np`` for the casts/intrinsics and ``int_ceil`` for a ceiling division, which has no python
#: operator. ``int_floor`` is rewritten to ``//`` (see :data:`_USERFUNC_REWRITES`) and is kept bound
#: only for a path that renders a bound without going through :func:`normalize_casts`.
#: Never build this namespace by hand at a call site -- use :func:`load_emitted`, which is the one place
#: that knows the full set (a hand-rolled ``{"np": np}`` is how ``int_floor`` went missing in CI).
EMITTED_BUILTINS = {"np": numpy, "int_floor": int_floor, "int_ceil": int_ceil}

#: The same three names as SOURCE, so an emitted kernel can be a SELF-CONTAINED module instead of one
#: that only runs under :func:`load_emitted`'s injected namespace. A representation handed to an agent
#: has to be numpy it can paste into a file and run, and that a translator can read without being told
#: what ``int_floor`` means -- an injected builtin is neither.
#:
#: GENERATED from the functions above rather than hand-copied. A hand-written copy has no mechanism keeping
#: it in step: an edit to ``int_ceil``'s sign handling would update the callable that every validated run
#: uses and leave the emitted text computing the old index arithmetic, with nothing to catch the drift and
#: only a standalone-emitted kernel affected.
STANDALONE_PREAMBLE = "import numpy as np\n\n\n" + "\n\n\n".join(
    inspect.getsource(fn).strip() for fn in (int_floor, int_ceil)) + "\n"


def standalone_source(fn_name: str, args: List[str], body: List[str]) -> str:
    """A rendered kernel plus the preamble that makes it importable on its own -- pure numpy, no
    injected namespace."""
    return f"{STANDALONE_PREAMBLE}\n\n{render(fn_name, args, body)}"


@functools.lru_cache(maxsize=None, typed=True)
def emitted_dir() -> Path:
    """Process-lifetime directory holding the emitted kernel sources handed to the import machinery.

    The files must OUTLIVE the modules loaded from them: ``linecache`` reads the source lazily, so a
    deleted file turns every frame of an emitted kernel into a blank line in the traceback -- exactly the
    frames worth reading when an oracle disagrees with a compiled variant. ``/tmp`` is tmpfs here, so the
    sources cost RAM rather than disk I/O, and the whole tree goes at interpreter exit.
    """
    path = Path(tempfile.mkdtemp(prefix="nestforge-emitted-"))
    atexit.register(shutil.rmtree, path, ignore_errors=True)
    return path


def load_emitted(source: str, name: str) -> ModuleType:
    """Import emitted numpy kernel ``source`` as a real module, with :data:`EMITTED_BUILTINS` pre-bound.

    The source becomes a file and goes through the normal import machinery, so the kernel gets a genuine
    module namespace (``__name__``, ``__file__``, a source-backed traceback) instead of a bare dict. Pull
    the kernel out with ``vars(module)[name]``; ``name`` only labels the module and its file.

    The file name carries a HASH OF THE SOURCE, not a counter. CPython invalidates ``__pycache__`` on
    (mtime, size), so two different kernels written to one path within the same second at the same byte
    length silently reuse the first one's bytecode -- the second import returns the FIRST kernel. That is
    a wrong-answer bug, not a slow one: the caller validates and times a kernel it did not emit. A
    counter did not prevent it either, since :func:`nestforge.isolation.run_isolated` forks, so every
    child inherits the same next value and writes the same path. Keying on content makes distinct
    sources distinct files, and lets identical sources legitimately share one cache entry.
    """
    path = emitted_dir() / f"{name}_{hashlib.sha256(source.encode()).hexdigest()[:16]}.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location(f"nestforge_emitted.{name}", path)
    module = importlib.util.module_from_spec(spec)
    module.__dict__.update(EMITTED_BUILTINS)  # bound BEFORE exec: the source references them at call time
    spec.loader.exec_module(module)
    return module


#: ``(dace.)?math.<fn>`` (a qualified intrinsic the bare-name rewrite deliberately skips) -> its numpy
#: form. Extra numpy-verbatim names beyond :data:`_MATH_INTRINSICS` that a TSVC/HPC tasklet may spell
#: qualified; anything else is left as ``math.<fn>`` (the emitter refuses rather than guess a bad name).
_NP_VERBATIM_MATH = frozenset({"power", "arcsin", "arccos", "arctan", "arctan2", "maximum", "minimum", "abs"})
_MATH_PREFIX_CALL = re.compile(r"\b(?:dace\.)?math\.(\w+)(?=\s*\()")


def apply_call(code: str, name: str, fn: Callable[..., str]) -> str:
    """Rewrite every ``name(arg0, arg1)`` call in ``code`` via ``fn(arg0, arg1)``, matching balanced
    parentheses so nested arguments stay intact. Leftmost-first with a rescan from the start; the outer
    :func:`rewrite_userfuncs` fixpoint loop resolves calls nested inside the replacements."""
    pat = re.compile(rf"(?<![\w.]){re.escape(name)}\s*\(")
    while True:
        m = pat.search(code)
        if not m:
            return code
        i, depth, bracket, cur, args = m.end(), 1, 0, "", []
        while i < len(code) and depth > 0:
            ch = code[i]
            if ch == "(":
                depth += 1
                cur += ch
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    break
                cur += ch
            elif ch in "[{":
                bracket += 1
                cur += ch
            elif ch in "]}":
                bracket -= 1
                cur += ch
            elif ch == "," and depth == 1 and bracket == 0:
                # Only a TOP-LEVEL comma separates arguments. A subscript's comma (``a[i, j]``) is
                # part of one argument: splitting there hands the rewrite the wrong arity, and the
                # pieces splice back with unmatched brackets -- the emitted C then failed to parse
                # (TSVC s1111/s1113) or left the call unrewritten to leak into C (s111's int_ceil).
                args.append(cur)
                cur = ""
            else:
                cur += ch
            i += 1
        if depth != 0:
            return code  # unbalanced (should not happen on emitted code): leave it for the caller to hit
        args.append(cur)
        code = code[:m.start()] + fn(*(a.strip() for a in args)) + code[i + 1:]


def rewrite_userfuncs(code: str) -> str:
    """Rewrite DaCe sympy user-functions (:data:`_USERFUNC_REWRITES`) to numpy/python, to a fixpoint so
    cross-function nesting (``ipow`` inside a ``Max`` argument) fully resolves."""
    for _ in range(16):  # bounded: every rewrite strictly removes one user-function call
        new = code
        for name, fn in _USERFUNC_REWRITES.items():
            new = apply_call(new, name, fn)
        if new == code:
            return code
        code = new
    return code


def rewrite_math_prefix(code: str) -> str:
    """Rewrite a qualified ``dace.math.sin`` / ``math.sin`` to ``np.sin`` (the bare-name rewrite skips
    qualified forms via its lookbehind). Uses the same intrinsic map, so ``asin`` -> ``np.arcsin``."""

    def repl(m: re.Match[str]) -> str:
        fn = m.group(1)
        if fn in _MATH_INTRINSICS:
            return _MATH_INTRINSICS[fn]
        if fn in _NP_VERBATIM_MATH:
            return f"np.{fn}"
        return f"math.{fn}"

    return _MATH_PREFIX_CALL.sub(repl, code)


#: a C++ ``decltype(<connector>)`` cast prefix that DaCe emits in generated tasklet code (e.g. the
#: ``numpy.linspace`` tasklet writes ``decltype(__out)(stop - start)`` to force the connector's type). The
#: connector name holds no parentheses, so this strips just the ``decltype(...)`` prefix and leaves the
#: parenthesized value it casts -- numpy promotes types itself, so the cast is a no-op on the reference.
_DECLTYPE_CAST = re.compile(r"\bdecltype\s*\([^()]*\)")


@functools.lru_cache(maxsize=None, typed=True)
def normalize_casts(code: str) -> str:
    """Rewrite DaCe dtype casts, math intrinsics, and sympy user-functions to numpy so the kernel needs
    no ``dace``/``math`` runtime import.

    All rewrites are value preserving. Order matters: the C++ ``decltype`` cast is stripped first (it wraps
    a parenthesized value the later rewrites still see); the qualified ``math.`` rewrite runs before the
    bare-name one, whose lookbehind then correctly skips the produced ``np.sin``; dtype casts and
    user-functions are independent.
    """
    code = _DECLTYPE_CAST.sub("", code)
    code = _DACE_CAST.sub(lambda m: _DACE_DTYPES[m.group(1)], code)
    code = _BARE_CAST.sub(lambda m: f"{_DACE_DTYPES[m.group(1)]}(", code)
    code = rewrite_math_prefix(code)
    code = _INTRINSIC_CALL.sub(lambda m: _MATH_INTRINSICS[m.group(1)], code)
    return rewrite_userfuncs(code)


#: Every DaCe precondition trap is a connectorless CPP tasklet holding this. Two passes emit it
#: (canonicalize's symbol assumptions, the scatter-conflict guard), so match the shape, not a label.
_TRAP_GUARD = re.compile(r"^\s*if\s*\((?P<cond>.+)\)\s*\{\s*__builtin_trap\s*\(\s*\)\s*;?\s*\}\s*;?\s*$", re.DOTALL)

#: C spellings with a Python equivalent. ``!`` needs the lookahead so ``!=`` survives intact.
_C_TO_PYTHON = ((re.compile(r"&&"), " and "), (re.compile(r"\|\|"), " or "), (re.compile(r"!(?!=)"), " not "),
                (re.compile(r"\btrue\b"), "True"), (re.compile(r"\bfalse\b"), "False"))


def trap_guard_lines(tasklet: nodes.Tasklet) -> List[str] | None:
    """Python equivalent of a C trap guard, or ``None`` if this tasklet is not one.

    ``__builtin_trap()`` aborts on a violated precondition, so an oracle that drops the guard is a
    different program from the kernel it validates. The condition is ``sym2cpp`` output; the rewrites
    below plus :func:`normalize_casts` cover it.
    """
    matched = _TRAP_GUARD.match(tasklet.code.as_string)
    if matched is None:
        return None
    cond = matched.group("cond")
    for pattern, replacement in _C_TO_PYTHON:
        cond = pattern.sub(replacement, cond)
    cond = re.sub(r"\s+", " ", normalize_casts(cond)).strip()  # eval-mode parse rejects a leading space
    try:
        ast.parse(cond, mode="eval")
    except SyntaxError as exc:
        raise UnsupportedNest(f"trap guard {tasklet.label} has a condition that is not translatable to "
                              f"python: {cond!r}") from exc
    return [f"if {cond}:", f"    raise AssertionError({f'violated assumption in {tasklet.label}'!r})"]


@functools.lru_cache(maxsize=None, typed=True)
def reduction_type(wcr_str: str) -> dace.dtypes.ReductionType:
    """Cached :func:`detect_reduction_type` -- the WCR lambda string repeats across every edge of a
    reduction (one map may have thousands of iterations sharing the same handful of WCR strings), and
    the underlying detector re-parses the lambda as Python each call."""
    return _detect_reduction_type(wcr_str)


#: reduction type -> ``(accumulator, term) -> combined expression`` for a WCR (augmented) write.
_WCR_BINOP = {
    dace.dtypes.ReductionType.Sum: lambda acc, t: f"{acc} + {t}",
    dace.dtypes.ReductionType.Product: lambda acc, t: f"{acc} * {t}",
    dace.dtypes.ReductionType.Max: lambda acc, t: f"np.maximum({acc}, {t})",
    dace.dtypes.ReductionType.Min: lambda acc, t: f"np.minimum({acc}, {t})",
}


def tasklet_lines(state: dace.SDFGState, sdfg: dace.SDFG, tasklet: nodes.Tasklet) -> List[str]:
    """The tasklet's Python code with connectors substituted by the array element they name.

    A plain output connector is substituted by the target element it writes. A **WCR** (reduction)
    output -- a scatter-accumulate like ``hist[bin] += w`` -- becomes an augmented assignment: the
    body writes a fresh temporary, then the target is combined with it (``target = target + tmp`` for
    Sum, ``np.maximum`` for Max, ...). Sequential emission keeps this correct when several iterations
    hit the same target element (the whole point of the WCR).
    """
    if not tasklet.in_connectors and not tasklet.out_connectors:
        # No connectors -> no data effect, whatever the language. The one meaningful case is the
        # precondition trap; anything else contributes nothing but provenance.
        return trap_guard_lines(tasklet) or [f"# no-op tasklet ({tasklet.label}): no connectors, no data effect"]
    if tasklet.code.language != dace.dtypes.Language.Python:
        raise UnsupportedNest(f"tasklet {tasklet.label} is not Python ({tasklet.code.language})")
    conn_expr: Dict[str, str] = {}
    for e in state.in_edges(tasklet):
        if e.dst_conn is not None:
            conn_expr[e.dst_conn] = access(sdfg, e.data.data, e.data.subset)
    wcr_updates: List[str] = []
    for e in state.out_edges(tasklet):
        if e.src_conn is None:
            continue
        target = access(sdfg, e.data.data, e.data.subset)
        if e.data.wcr is None:
            conn_expr[e.src_conn] = target
            continue
        combine = _WCR_BINOP.get(reduction_type(e.data.wcr))
        if combine is None:
            raise UnsupportedNest(f"tasklet {tasklet.label} has an unsupported WCR {e.data.wcr!r}")
        temp = f"__wcr_{e.src_conn}"
        conn_expr[e.src_conn] = temp  # the body writes the temporary, then we accumulate into target
        wcr_updates.append(f"{target} = {combine(target, temp)}")
    pattern = connector_pattern(conn_expr)
    lines = [normalize_casts(sub_connectors(line, conn_expr, pattern)) for line in tasklet.code.as_string.splitlines()]
    # The accumulate lines are built here, not from tasklet code, but they embed the target's rendered
    # subset -- a strided/derived index renders as sympy ``int_floor(...)``, which is not python. They need
    # the same normalization the body lines get.
    return lines + [normalize_casts(u) for u in wcr_updates]


def copy_side(sdfg: dace.SDFG, name: str, subset: Optional[dace.subsets.Range]) -> str:
    """One side of a memlet copy, rendered as a squeezed view. A scalar local stays bare and a size-1
    buffer reads its element; every other array drops its length-1 axes, so both sides reduce to the
    same packed shape (``(N, 1)`` and ``(1, N)`` both become the ``(N,)`` view ``a[:, 0]`` / ``a[0, :]``).
    A DaCe memlet copy moves elements in volume order -- exactly this squeezed vector-to-vector copy --
    which keeps a reshape (``(N,1)`` buffer) and a transpose (``q[:, 0] = v[0, :]``) both correct."""
    if scalar_local(sdfg, name):
        return name
    desc = sdfg.arrays[name]
    if is_scalar(desc):
        return f"{name}[0]"
    if subset is None:
        subset = dace.subsets.Range.from_array(desc)
    return f"{name}[{index_str(subset)}]"  # keep_singleton default: length-1 axes collapse away


def copy_direction(edge: dace.sdfg.graph.MultiConnectorEdge) -> tuple:
    """``(src_name, src_subset, dst_subset)`` for one access-node -> access-node copy edge.

    ``memlet.subset`` indexes ``memlet.data`` (a DaCe invariant), so whichever endpoint ``data`` names
    takes ``subset`` and the other takes ``other_subset``.

    The SOURCE is tested first, and that order is the whole point: on an in-place copy both endpoints
    carry the SAME name, so both tests match and the order decides. DaCe resolves the tie the same way
    (``Memlet.try_initialize``: "in case both point to the same array, prefer ... ``is_data_src=True``"),
    i.e. ``subset`` is the source range. Testing the destination first inverted exactly that case, so
    ``A[i] = A[j]`` emitted ``A[j] = A[i]`` -- silently, since every other copy has two distinct names
    and only one test can match.
    """
    m = edge.data
    if m.data == edge.src.data:
        return edge.src.data, m.subset, m.other_subset
    if m.data == edge.dst.data:
        return edge.src.data, m.other_subset, m.subset
    raise UnsupportedNest(f"copy memlet {m.data!r} names neither {edge.src.data!r} nor {edge.dst.data!r}")


def copy_lines(state: dace.SDFGState, sdfg: dace.SDFG, dst: nodes.AccessNode) -> List[str]:
    """Emit ``dst[..] = src[..]`` for each memlet copy feeding ``dst`` from an access node or a map entry.

    Simplified SDFGs stage tasklet operands through scratch access nodes -- a plain tasklet-less data
    copy -- in two forms this handles:

    * **access-node -> access-node** (``s = A[i]`` / ``B[:] = A[k, :, :]``): the memlet names one side
      in ``memlet.data``/``subset`` and the other in ``other_subset``; we resolve which is the source.
    * **map-entry -> access-node**: DaCe requires an indexed array read to pass through a data node, so
      an in-map element read is staged as ``b_index = b[i]`` (a scalar access node fed by the map
      entry). The memlet names the OUTER source array in ``memlet.data``/``subset``; the scratch is the
      destination (``other_subset``). Emitting this load lets a later ``a[b_index]`` gather resolve --
      ``b_index`` is bound to ``b[i]`` before its first use.
    """
    lines: List[str] = []
    for e in state.in_edges(dst):
        m = e.data
        if m.is_empty():
            continue  # an empty memlet is a happens-before/ordering edge (StateFusion sequencing), no data
        if isinstance(e.src, nodes.AccessNode):
            src_name, src_sub, dst_sub = copy_direction(e)
        elif isinstance(e.src, nodes.MapEntry):
            # Staged array read through the map entry: the memlet names the outer source array + the
            # element/range read; the scratch access node is the destination.
            src_name, src_sub, dst_sub = m.data, m.subset, m.other_subset
        else:
            # A Tasklet / MapExit source's WCR is emitted where that edge is owned: tasklet_lines (the
            # tasklet out-edge names the outer array) and map_exit_writes (the exit's in-edges) handle the
            # accumulate, so this copy edge is a genuine no-op. A LibraryNode / NestedSDFG source ignores
            # its out-edge WCR, but that is refused at the source's own emitter (out_lhs / emit_nested_sdfg),
            # so it never reaches a silent overwrite here either.
            continue
        lhs, rhs, dst_read = copy_sides(sdfg, dst.data, dst_sub, src_name, src_sub)
        if m.wcr is not None:
            # A reduction *copy* (AccessNode -> AccessNode carrying a WCR, e.g. a privatized accumulator
            # copied back): accumulate rather than overwrite -- ``dst = combine(dst, src)``. Tasklet WCRs
            # go through their own path in tasklet_lines; this is the copy-edge analogue.
            combine = _WCR_BINOP.get(reduction_type(m.wcr))
            if combine is None:
                raise UnsupportedNest(f"reduction (WCR) copy into {dst.data} has an unsupported WCR {m.wcr!r}")
            rhs = combine(dst_read, rhs)
        lines.append(normalize_casts(f"{lhs} = {rhs}"))  # a strided subset may render an int_floor/int_ceil index
    return lines


def copy_sides(sdfg: dace.SDFG, dst_name: str, dst_sub: Optional[dace.subsets.Range], src_name: str,
               src_sub: Optional[dace.subsets.Range]) -> tuple:
    """``(lhs, rhs, dst_read)`` for one data copy -- the shared body of :func:`copy_lines` and
    :func:`map_exit_writes`, which render the same assignment from different edges.

    Same-rank copy: both sides render as squeezed views (:func:`copy_side`), and a ``None`` subset on
    one side mirrors the other's so a partial copy is not silently widened. Mirroring happens only
    between arrays of the SAME SHAPE -- a same-rank reshape (``pos[:, 1:2]`` into an ``[N, 1]`` buffer)
    must keep each side's own subset, or the source's column is written out of bounds on the
    destination.

    Rank-changing copy (``(N,) <-> (N, 1)``, or an array element staged into a scalar): a scalar-local
    side stays bare, the reshaping side keeps its subset explicit so a point index collapses the rank,
    and a ``None`` side is the whole array (:func:`reshape_side`).

    ``dst_read`` is the destination rendered for READING -- a WCR copy accumulates into it.
    """
    if len(sdfg.arrays[src_name].shape) == len(sdfg.arrays[dst_name].shape):
        if sdfg.arrays[src_name].shape == sdfg.arrays[dst_name].shape:
            src_sub = src_sub if src_sub is not None else dst_sub
            dst_sub = dst_sub if dst_sub is not None else src_sub
        return (copy_side(sdfg, dst_name, dst_sub), copy_side(sdfg, src_name,
                                                              src_sub), copy_side(sdfg, dst_name, dst_sub))
    return (reshape_side(sdfg, dst_name, dst_sub,
                         write=True), reshape_side(sdfg, src_name, src_sub,
                                                   write=False), reshape_side(sdfg, dst_name, dst_sub, write=False))


def reshape_side(sdfg: dace.SDFG, name: str, subset: Optional[dace.subsets.Range], write: bool) -> str:
    """One side of a rank-changing copy: bare for a scalar local, explicit ``name[idx]`` for the
    reshaping (rank-collapsing) side, else the whole array via the access oracle."""
    if scalar_local(sdfg, name):
        return name
    if subset is None:
        return write_lhs(sdfg, name, None) if write else read_expr(sdfg, name, None)
    return f"{name}[{index_str(subset)}]"


def reject_underranked_codeblock_index(inner: dace.SDFG) -> None:
    """Refuse a nested SDFG whose inter-state code indexes a multi-dim array with too few indices.

    ``ExpandNestedSDFGInputs`` widens a collapsed size-1 inner array to the full outer array and
    offsets references by the map index -- but for a reference inside an inter-state *condition* or
    *assignment* (``I_0 = I[0]``) it adds only the first map dimension, leaving ``getAcc_I_0[__i0]``
    on an ``(N, N)`` array (a numpy row, not the element). That DaCe-pass gap would emit
    ``if <array>:``, so we reject with a precise reason instead of a broken kernel.
    """
    for region in inner.all_control_flow_regions():
        for e in region.edges():
            codes = list(e.data.assignments.values())
            if not e.data.is_unconditional():
                codes.append(e.data.condition.as_string)
            for code in codes:
                try:
                    tree = ast.parse(code)
                except SyntaxError as exc:
                    raise UnsupportedNest(f"inter-state code {code!r} is not parseable Python") from exc
                for sub in ast.walk(tree):
                    if not (isinstance(sub, ast.Subscript) and isinstance(sub.value, ast.Name)):
                        continue
                    desc = inner.arrays.get(sub.value.id)
                    if desc is None or len(desc.shape) <= 1:
                        continue
                    ndims = len(sub.slice.elts) if isinstance(sub.slice, ast.Tuple) else 1
                    if ndims < len(desc.shape):
                        raise UnsupportedNest(
                            f"nested SDFG under-indexes {sub.value.id!r} ({ndims} of {len(desc.shape)} dims) "
                            "in inter-state code -- ExpandNestedSDFGInputs offsets multi-dim conditions incompletely")


def reconcile_connector_descriptor(inner: dace.SDFG, sdfg: dace.SDFG, outer: str) -> None:
    """Make the inner descriptor for connector array ``outer`` agree with the buffer it aliases.

    The connector IS the outer buffer, so it adopts the outer descriptor outright. Matching shapes are
    not enough: ``transient`` decides the SPELLING (a scalar transient is a bare local, everything else
    is indexed), and a connector is never transient inside while the buffer it aliases may well be
    outside. Returning early on equal shapes left exactly that gap -- a fused map's ``__map_fusion_T``
    scalar was written ``__map_fusion_T[0] = ...`` by the inner body, read bare by the outer one, and
    allocated by neither (``scratch_arrays`` skips size-1 transients), so the emitted C referenced an
    undeclared name.

    Two shapes that genuinely differ are refused unless both are single-element (a ``Scalar`` inside, a
    size-1 ARRAY outside -- a nested return; both spell one element). Adopting a differing outer shape
    would silently re-rank the body: an under-offset multi-dim connector emits ``Z[j]`` -- a whole row --
    where ``Z[j, k]`` was meant. :func:`reject_underranked_codeblock_index` catches that for inter-state
    code but not for dataflow memlets, so refuse it here instead of papering over it.
    """
    inner_desc, outer_desc = inner.arrays[outer], sdfg.arrays[outer]
    same_shape = [str(d) for d in inner_desc.shape] == [str(d) for d in outer_desc.shape]
    if not same_shape and not (is_scalar(inner_desc) and is_scalar(outer_desc)):
        raise UnsupportedNest(f"nested SDFG connector {outer!r} is {inner_desc.shape} inside but "
                              f"{outer_desc.shape} outside; the extents differ, so the inner body indexes a "
                              "different shape than the buffer it aliases -- not emittable as numpy")
    inner.arrays[outer] = copy.deepcopy(outer_desc)


def emit_nested_sdfg(state: dace.SDFGState, sdfg: dace.SDFG, node: nodes.NestedSDFG) -> List[str]:
    """Inline a nested SDFG (e.g. one map iteration's sub-kernel) as flat statements, in place.

    :func:`expand_nested_sdfg_inputs` has already widened every in/out connector to the *full* outer
    array, and DaCe offsets the inner memlets by the enclosing map index, so the inner body reads and
    writes the outer buffers directly (``Z[j, k]``) using the map symbols already in scope. Emitting
    it is then: bind the symbol mapping, alias each connector array to the outer array it binds,
    rename any private transient that would shadow an outer buffer, and emit the inner body. Inner
    control flow (a masked write to ``Z[j, k]``) stays correct because the write lands on the outer
    array in place, leaving the other elements untouched.
    """
    for e in state.out_edges(node):
        if e.data.wcr is not None:
            # This function replays the inner body only (emit_region below); it never applies a WCR carried
            # on the nested SDFG's OUTER output edge, so an accumulate would silently become an overwrite.
            # At a map exit map_exit_writes already refuses this; guarding here also covers a nested SDFG at
            # state-body level, which that guard never sees. Refuse -> the ExternalCall uses the DaCe variant.
            raise UnsupportedNest(
                f"nested SDFG output into {e.data.data} carries a reduction (WCR) that emit_nested_sdfg does "
                "not apply; not emittable as numpy -- fall back to the DaCe variant")
    if ExpandNestedSDFGInputs is None:
        raise UnsupportedNest("nested SDFG emission needs ExpandNestedSDFGInputs (DaCe extended branch)")
    inner = copy.deepcopy(node.sdfg)
    conns = {e.dst_conn: e.data.data for e in state.in_edges(node) if e.data.data is not None}
    conns.update({e.src_conn: e.data.data for e in state.out_edges(node) if e.data.data is not None})
    for conn, outer in conns.items():
        if conn != outer:
            inner.replace(conn, outer)
        if outer in sdfg.arrays:
            reconcile_connector_descriptor(inner, sdfg, outer)
    reject_underranked_codeblock_index(inner)
    # A private (non-connector) inner transient becomes a plain python local. That only works for a
    # scalar; a private *array* transient would be emitted as ``name[:] = ...`` yet appears in no
    # signature (``scratch_arrays`` scans the outer SDFG only), so it is refused rather than left
    # undefined. Connector arrays alias an outer buffer parameter and are exempt.
    outer_names = set(sdfg.arrays)
    node_id = state.node_id(node)
    for name, desc in list(inner.arrays.items()):
        if name in conns.values():
            continue
        if not is_scalar(desc):
            raise UnsupportedNest(f"nested SDFG private transient {name!r} is a non-scalar array; not allocated")
        # A private inner name that collides with an outer buffer would shadow it -- rename it.
        if name in outer_names:
            inner.replace(name, f"_ns{node_id}_{name}")

    lines = symbol_mapping_lines(node.symbol_mapping, state.node_id(node))
    lines += emit_region(inner, inner)
    return lines


def symbol_mapping_lines(mapping: Dict[str, object], node_id: int) -> List[str]:
    """Bind a nested SDFG's ``symbol_mapping`` -- SIMULTANEOUSLY when the bindings interfere.

    The mapping is a substitution, applied all at once. Emitting it as ordered assignments is only
    equivalent while no target appears on a later right-hand side; a swap ``{i: j, j: i}`` emits
    ``i = j`` then ``j = i``, and both end up holding the old ``j``.

    So: read every right-hand side into a temp first, then assign. Only when there IS interference --
    the plain form is what the reader (and the C translator) sees the rest of the time.
    """
    binds = [(str(sym), normalize_casts(str(expr))) for sym, expr in mapping.items() if str(sym) != str(expr)]
    if not binds:
        return []
    targets = {sym for sym, _ in binds}
    reads = set()
    for _, expr in binds:
        reads |= {str(s) for s in symbolic.pystr_to_symbolic(expr).free_symbols}
    if not (targets & reads):
        return [f"{sym} = {expr}" for sym, expr in binds]
    temps = [(f"_nsym{node_id}_{sym}", sym, expr) for sym, expr in binds]
    return ([f"{tmp} = {expr}" for tmp, _, expr in temps] + [f"{sym} = {tmp}" for tmp, sym, _ in temps])


def map_exit_writes(state: dace.SDFGState, sdfg: dace.SDFG, entry: nodes.MapEntry) -> List[str]:
    """Writes that leave the map through its exit from an in-scope AccessNode.

    A tasklet that writes an outer array does so directly (its out-edge memlet names the target, handled
    in :func:`tasklet_lines`). But a canonicalized reduction stages its result in a privatized
    accumulator AccessNode *inside* the map, whose out-edge to the MapExit carries the WCR into the real
    output (``_priv_out += acc`` for a Sum). :func:`copy_lines` cannot emit that write -- it hangs off
    the outer output node, which is fed by the MapExit (a passthrough it skips). Emit it here, inside the
    loop body, accumulating for a WCR edge and overwriting otherwise, so the reduction reaches the buffer.

    Mirrors the MapEntry-read convention in :func:`copy_lines`: on an edge into the exit the memlet names
    the OUTER destination array (``m.data``/``m.subset``); the in-scope access node is the source, read at
    ``m.other_subset``.
    """
    lines: List[str] = []
    for e in state.in_edges(state.exit_node(entry)):
        if not isinstance(e.src, nodes.AccessNode) or state.entry_node(e.src) is not entry:
            # Only a tasklet WCR out-edge accumulates on emit: tasklet_lines rewrites it as
            # ``target = target + tmp``. Every other source at this exit -- a nested map, a library node,
            # or a NESTED SDFG -- emits its body only (emit_nested_sdfg calls emit_region and never applies
            # the out-edge WCR), so a reduction reaching the exit from one of them would silently become an
            # overwrite (last-write-wins), mis-emitting the reduction as a no-op -- the very bug this
            # function exists to fix, one nest-level in. Refuse those so the ExternalCall falls back to the
            # DaCe variant rather than emit a wrong kernel.
            if e.data.wcr is not None and not isinstance(e.src, nodes.Tasklet):
                raise UnsupportedNest(
                    f"reduction (WCR) leaves the map exit from a {type(e.src).__name__}, not an in-scope "
                    "accumulator access node; not emittable as numpy -- fall back to the DaCe variant")
            continue
        m = e.data
        dst_name, dst_sub, src_sub = m.data, m.subset, m.other_subset
        src_name = e.src.data
        if src_name == dst_name and m.wcr is None:
            # a plain self-edge moves nothing out of the map. A self-edge carrying a WCR is NOT a no-op:
            # it is an in-place reduction (accumulator and outer array share a name), and skipping it
            # would silently emit the reduction as nothing -- fall through to the accumulate below.
            continue
        lhs, rhs, dst_read = copy_sides(sdfg, dst_name, dst_sub, src_name, src_sub)
        if m.wcr is not None:
            combine = _WCR_BINOP.get(reduction_type(m.wcr))
            if combine is None:
                raise UnsupportedNest(f"reduction (WCR) write-out into {dst_name} has an unsupported WCR {m.wcr!r}")
            rhs = combine(dst_read, rhs)
        lines.append(normalize_casts(f"{lhs} = {rhs}"))
    return lines


def range_stop(end: sympy.Expr, step: sympy.Expr, what: str) -> sympy.Expr:
    """Python's exclusive ``range`` stop for a DaCe range whose ``end`` is INCLUSIVE.

    One past the last element in the direction of travel: ``end + 1`` ascending, ``end - 1`` descending.
    A blanket ``+ 1`` is right only for a positive step -- for ``range(N-1, -1, -1)`` it emits
    ``range(N-1, 0, -1)`` and drops element 0, silently. A step whose sign is not decidable has no sound
    stop, so refuse rather than guess a direction.
    """
    sign = sympy.sign(sympy.sympify(step))
    if sign not in (1, -1):
        raise UnsupportedNest(f"{what} has step {step} of undecidable sign; no sound python range stop")
    return end + sign


def map_headers(entry: nodes.MapEntry) -> List[str]:
    """One ``for`` header per map dimension, outermost first, unindented."""
    headers: List[str] = []
    for param, (beg, end, step) in zip(entry.map.params, entry.map.range.ranges):
        stop = range_stop(end, step, f"map parameter {param!r}")
        headers.append(
            normalize_casts(f"for {param} in range({symbolic.symstr(beg)}, {symbolic.symstr(stop)}, "
                            f"{symbolic.symstr(step)}):"))  # a bound may render an int_floor/int_ceil
    return headers


def map_body_lines(state: dace.SDFGState, sdfg: dace.SDFG, entry: nodes.MapEntry) -> List[str]:
    """What one map scope COMPUTES, unindented and without its ``for`` headers.

    Split out of :func:`map_lines` so a caller that already knows the iteration domain -- the agent's
    tree prints it on the kernel line -- can ask for the body alone instead of re-deriving it by
    dropping ``len(params)`` lines off the front of the emitted block and dedenting the rest by
    ``4 * len(params)``. That arithmetic was right only while every header occupied exactly one line
    and every body line carried the full indent.
    """
    body: List[str] = []
    scope = state.scope_subgraph(entry, include_entry=False, include_exit=False)
    for node in dfs_topological_sort(scope):
        # ``scope_subgraph`` returns the whole subtree, including a nested map's own descendants; emit only
        # the DIRECT children of THIS map here and let the recursion below handle a nested map's body, so a
        # grandchild is not also emitted flat at the wrong depth (mirrors the guard in ``state_body``).
        if state.entry_node(node) is not entry:
            continue
        if isinstance(node, nodes.Tasklet):
            body.extend(tasklet_lines(state, sdfg, node))
        elif isinstance(node, nodes.AccessNode):
            body.extend(copy_lines(state, sdfg, node))
        elif isinstance(node, nodes.NestedSDFG):
            body.extend(emit_nested_sdfg(state, sdfg, node))
        elif isinstance(node, nodes.MapEntry):
            # A map nested in a map -> emit it as deeper ``for`` loops. ``map_lines`` returns a
            # self-contained header+body block; splicing it into this body and applying the uniform
            # per-level indent below preserves the relative nesting.
            body.extend(map_lines(state, sdfg, node))
        elif isinstance(node, nodes.LibraryNode):
            raise UnsupportedNest(f"{type(node).__name__} nested inside a map is not yet emitted")

    body.extend(map_exit_writes(state, sdfg, entry))  # reductions/writes leaving the map via its exit
    return body


def map_lines(state: dace.SDFGState, sdfg: dace.SDFG, entry: nodes.MapEntry) -> List[str]:
    """Emit a map scope as ``for`` loops over pre-allocated buffers (no allocation of its own)."""
    headers = map_headers(entry)
    body = map_body_lines(state, sdfg, entry)
    lines = ["    " * depth + h for depth, h in enumerate(headers)]
    lines += ["    " * len(headers) + bl for bl in (body or ["pass"])]
    return lines


def state_body(sdfg: dace.SDFG, state: dace.SDFGState) -> List[str]:
    """Numpy statements for a whole state, in dataflow order (library nodes + maps + tasklets)."""
    lines: List[str] = []
    for node in dfs_topological_sort(state):
        if state.entry_node(node) is not None:
            continue  # emitted as part of its enclosing map scope
        if isinstance(node, nodes.MapEntry):
            lines.extend(map_lines(state, sdfg, node))
        elif isinstance(node, nodes.LibraryNode):
            try:
                lines.extend(emit_library_node(node, state, sdfg))
            except UnsupportedLibraryNode as exc:
                raise UnsupportedNest(str(exc)) from exc
        elif isinstance(node, nodes.Tasklet):
            lines.extend(tasklet_lines(state, sdfg, node))
        elif isinstance(node, nodes.AccessNode):
            lines.extend(copy_lines(state, sdfg, node))
        elif isinstance(node, nodes.NestedSDFG):
            lines.extend(emit_nested_sdfg(state, sdfg, node))
    return lines


def ordered_blocks(region: dace.sdfg.state.ControlFlowRegion) -> List:
    """Blocks of a control-flow region (SDFG or LoopRegion) in execution order."""
    return list(dfs_topological_sort(region, [region.start_block]))


def body_or_pass(lines: List[str]) -> List[str]:
    """A block body that must be non-empty Python. A region emitting only ``# ...`` provenance
    comments (an empty state/loop) would leave a ``while``/``if``/``def`` body with no statement --
    an ``IndentationError`` -- so append ``pass`` when no real statement is present."""
    return lines if any(not ln.lstrip().startswith("#") for ln in lines) else lines + ["pass"]


def targets_continue(loop: LoopRegion) -> bool:
    """True if some ``ContinueBlock`` inside ``loop`` targets ``loop`` itself (not an inner loop)."""
    return any(isinstance(b, ContinueBlock) and innermost_loop(b) is loop for b in loop.all_control_flow_blocks())


def emit_loop(loop: LoopRegion, sdfg: dace.SDFG) -> List[str]:
    """Emit a ``LoopRegion`` as init + ``while`` (do-while when ``inverted``) around its body.

    Bounds come from DaCe's canonical ``init``/``condition``/``update`` statements, so a
    ``for t in range(...)`` round-trips as ``t = 1 / while (t < TSTEPS): ... / t = (t + 1)``.
    ``init``/``update`` are optional (a bare ``while``); the condition is required.

    A ``continue`` needs the update carried to it. DaCe emits ``for (init; cond; update)``, where
    ``continue`` still runs the update; python's ``while`` puts the update in the body, where ``continue``
    skips it -- so the loop variable never advances and the emitted oracle HANGS instead of disagreeing.
    The non-inverted form emits the update in front of each ``continue``; the inverted form cannot be
    patched that way (its exit test lives in the body too, so a ``continue`` skips the test as well) and is
    refused instead.
    """
    if loop.loop_condition is None:
        raise UnsupportedNest(f"loop {loop.label} has no condition")
    cond = normalize_casts(loop.loop_condition.as_string.strip())
    init = normalize_casts(loop.init_statement.as_string.strip()) if loop.init_statement is not None else None
    update = normalize_casts(loop.update_statement.as_string.strip()) if loop.update_statement is not None else None
    if loop.inverted and targets_continue(loop):
        raise UnsupportedNest(f"loop {loop.label} is inverted (do-while) and contains a continue; the emitted "
                              "`while True:` carries its exit test in the body, so a `continue` would skip the "
                              "test and loop forever")
    body = body_or_pass(emit_region(loop, sdfg, continue_update=None if loop.inverted else update))
    ind = "    "

    lines: List[str] = []
    if init is not None:
        lines.append(init)
    if loop.inverted:  # do-while: body executes before the condition is tested
        lines.append("while True:")
        lines += [ind + b for b in body]
        test = [f"{ind}if not ({cond}):", f"{ind}{ind}break"]
        upd = [ind + update] if update is not None else []
        # update_before_condition selects whether the increment precedes or follows the test.
        lines += (upd + test) if loop.update_before_condition else (test + upd)
    else:
        lines.append(f"while {cond}:")
        lines += [ind + b for b in body]
        if update is not None:
            lines.append(ind + update)
    return lines


def emit_conditional(cond_block: ConditionalBlock, sdfg: dace.SDFG, continue_update: Optional[str] = None) -> List[str]:
    """Emit a ``ConditionalBlock`` as ``if``/``elif``/``else`` over its branches.

    Branches are ``(condition, region)`` and DaCe takes the FIRST whose condition holds, so they are
    emitted in the order they are stored: ``if``, then ``elif``, and a final unconditional branch
    (``condition is None``) as ``else``.

    An unconditional branch that is NOT last is refused, because DaCe refuses it too -- its own codegen
    raises ``Missing branch condition for non-final conditional branch``. Reordering it to the end
    instead (the previous shape) invented semantics the SDFG does not have: it made a keyed branch
    stored after the unconditional one live, and two unconditional branches emitted two ``else:``
    clauses, a SyntaxError in the generated kernel.
    """
    ind = "    "
    lines: List[str] = []
    keyword = "if"
    last = len(cond_block.branches) - 1
    for index, (condition, region) in enumerate(cond_block.branches):
        if condition is None and index != last:
            raise UnsupportedNest(f"conditional block {cond_block.label!r} has an unconditional branch at "
                                  f"position {index} of {last + 1}; DaCe codegen refuses a non-final "
                                  "unconditional branch, so there is no order to preserve")
        if condition is None:
            lines.append("else:")
        else:
            lines.append(f"{keyword} {normalize_casts(condition.as_string.strip())}:")
            keyword = "elif"
        lines += [ind + b for b in body_or_pass(emit_region(region, sdfg, continue_update))]
    return lines


def strip_scalar_local_subscript(code: str, sdfg: dace.SDFG) -> str:
    """Drop the ``[0]`` index off a scalar-transient array in a raw DaCe code string.

    DaCe refers to a size-1 array as ``A[0]``, but the emitter treats a scalar transient as a bare
    local (``A``); a raw inter-state assignment string (``bin = min(ret[0], ...)``) must match that
    convention or the bare write and the indexed read disagree (IndexError on a scalar). Only
    genuinely scalar-transient names are stripped; real arrays keep their indices.
    """
    for name, desc in sdfg.arrays.items():
        if scalar_local(sdfg, name):
            code = re.sub(rf"\b{re.escape(name)}\s*\[[^][]*\]", name, code)
    return code


def interstate_lines(region: dace.sdfg.state.ControlFlowRegion, sdfg: dace.SDFG,
                     block: dace.sdfg.state.ControlFlowBlock) -> List[str]:
    """Assignments carried on the edge(s) entering ``block`` (e.g. an indirect index ``s = A[i]``).

    DaCe hoists a data-dependent index or loop-carried scalar onto the inter-state edge that reaches
    a block; those assignments must run before the block body or the symbols they define are unbound.
    A conditional (branching) edge is an unstructured goto -- old-style state-machine control flow the
    straight-line topological emission does not model (structured branches are ``ConditionalBlock`` s,
    structured loops ``LoopRegion`` s) -- so ANY conditional edge is refused, whether or not it carries
    assignments, rather than silently emitting the successor blocks as if the branch were always taken.
    """
    lines: List[str] = []
    carrying = []
    for e in region.in_edges(block):
        if not e.data.is_unconditional():
            raise UnsupportedNest(
                f"conditional inter-state edge into {block.label} (unstructured goto/branch) is not emitted")
        if e.data.assignments:
            carrying.append(e)
    if len(carrying) > 1:
        # Straight-line emission runs every predecessor's assignments in sequence, but at runtime only ONE
        # predecessor executes. Emitting both double-applies them (two edges carrying `k = k + 1` increment
        # twice) and mixes values from a path not taken. Refuse, same as the conditional-edge case above,
        # rather than emit a wrong kernel.
        raise UnsupportedNest(
            f"{len(carrying)} inter-state edges into {block.label} carry assignments (an unstructured join); "
            "straight-line emission would apply every predecessor's assignments -- not emittable as numpy")
    for e in carrying:
        for lhs, rhs in e.data.assignments.items():
            lines.append(f"{lhs} = {strip_scalar_local_subscript(normalize_casts(rhs), sdfg)}")
    return lines


def emit_region(region: dace.sdfg.state.ControlFlowRegion,
                sdfg: dace.SDFG,
                continue_update: Optional[str] = None) -> List[str]:
    """Numpy statements for every block of a control-flow region, in execution order.

    Each block is preceded by a ``# <kind> (<label>)`` provenance comment (``# state (S)`` /
    ``# loop region (L)`` / ``# conditional (C)``) so the emitted source stays anchored to the SDFG
    region it came from -- readable output, and a handle an agent can grep a subregion by.

    ``continue_update`` is the enclosing loop's update statement, emitted in front of each ``continue``
    (see :func:`emit_loop`). It stops at the next :func:`emit_loop`, which passes its OWN update down, so
    a ``continue`` always advances the loop it actually targets.
    """
    lines: List[str] = []
    for block in ordered_blocks(region):
        lines.extend(interstate_lines(region, sdfg, block))
        if isinstance(block, dace.SDFGState):
            lines.append(f"# state ({block.label})")
            lines.extend(state_body(sdfg, block))
        elif isinstance(block, LoopRegion):
            lines.append(f"# loop region ({block.label})")
            lines.extend(emit_loop(block, sdfg))
        elif isinstance(block, ConditionalBlock):
            lines.append(f"# conditional ({block.label})")
            lines.extend(emit_conditional(block, sdfg, continue_update))
        elif isinstance(block, BreakBlock):
            lines.append("break")  # exits the enclosing while (emit_loop); its region-DFS successor is the loop exit
        elif isinstance(block, ContinueBlock):
            if continue_update is not None:
                lines.append(continue_update)  # python's while keeps the update in the body; continue would skip it
            lines.append("continue")
        elif isinstance(block, ReturnBlock):
            # Early return out of the SDFG. A whole-kernel python ``return`` matches this exactly (it exits
            # the function == exits the SDFG). Externalizing a *sub-nest* that carries one is refused up
            # front by ``reject_early_return`` (the return would only exit the nest, not the enclosing SDFG).
            lines.append("return")
        else:
            raise UnsupportedNest(f"control-flow block not yet emitted: {type(block).__name__}")
    return lines


def scratch_arrays(sdfg: dace.SDFG) -> List[str]:
    """Transient array buffers the caller must pre-allocate (scalar transients stay locals)."""
    return sorted(name for name, desc in sdfg.arrays.items() if desc.transient and not is_scalar(desc))


#: sympy heads meaning "this expression READS ARRAY DATA". DaCe renders ``A[i]`` as ``Subscript(A, i)``;
#: sympy's own indexed form is ``Indexed``. Math heads a real size may carry (``int_floor``, ``int_ceil``)
#: are deliberately absent, and ``Min``/``Max`` are not Function atoms at all.
_DATA_READ_HEADS = frozenset({"Subscript", "Indexed"})


def reads_array_data(expr: sympy.Expr, arrays: Mapping[str, dace.data.Data]) -> bool:
    """Whether ``expr`` reads the CONTENTS of an array, so its value is unknown until the kernel runs.

    Walks the expression TREE. Never use ``free_symbols`` for this: DaCe renders ``A_indptr[i]`` as
    ``Subscript(A_indptr, i)``, whose free symbols are ``{i}`` -- the array name is the Function HEAD,
    so a ``free_symbols`` test is structurally blind to the read.
    """
    for fn in expr.atoms(sympy.Function):
        if fn.func.__name__ in _DATA_READ_HEADS or fn.func.__name__ in arrays:
            return True
    return any(str(s) in arrays for s in expr.free_symbols)


def sizable(expr: sympy.Expr, known: set, arrays: Mapping[str, dace.data.Data]) -> bool:
    """Whether the CALLER can evaluate ``expr`` to a buffer extent before the kernel runs.

    True iff it reads no array data (:func:`reads_array_data`) and names no symbol outside ``known``.

    Both halves are load-bearing. spmv sized a scratch buffer by a CSR span
    (``A_indptr[M+1] - A_indptr[0]``): its residual free symbols were exactly the kernel symbols, so a
    ``free_symbols``-only check accepted it and asked the caller to allocate an extent only the data
    knows.
    """
    if reads_array_data(expr, arrays):
        return False
    return not {str(s) for s in expr.free_symbols} - known


def symbol_ranges(sdfg: dace.SDFG) -> tuple:
    """``(lo_of, hi_of)``: each non-argument symbol -> its min / max value in kernel symbols.

    Two sources feed the range of a symbol used in a buffer shape:

    * an *increasing loop variable* takes ``[init, condition-bound]`` (``i = 0`` / ``i < N`` -> ``[0,
      N]``; ``i <= N`` -> ``N + 1``);
    * an *inter-state-assigned* config symbol takes each value it is assigned (a layer size reused
      across a network -- ``N2 = S0``/``S1``/``S2``), so its range spans all of them.

    A symbol driven by several takes ``Min`` of its los / ``Max`` of its his; endpoints naming another
    such symbol are resolved recursively, so the result is in kernel symbols only.
    """
    los: Dict[str, list] = {}
    his: Dict[str, list] = {}
    for cfg in sdfg.all_control_flow_regions():
        if isinstance(cfg, LoopRegion) and cfg.loop_condition is not None:
            rel = symbolic.pystr_to_symbolic(cfg.loop_condition.as_string)
            var = cfg.loop_variable
            if isinstance(rel, (sympy.StrictLessThan, sympy.LessThan)) and str(rel.lhs) == var:
                his.setdefault(var, []).append(rel.rhs + (1 if isinstance(rel, sympy.LessThan) else 0))
                los.setdefault(var, []).append(
                    symbolic.pystr_to_symbolic(cfg.init_statement.as_string.split("=", 1)[1]) if cfg.
                    init_statement is not None else sympy.Integer(0))
        for e in cfg.edges():
            for var, rhs in e.data.assignments.items():
                try:
                    value = symbolic.pystr_to_symbolic(rhs)  # the config symbol takes exactly this value
                except Exception:
                    continue  # a non-symbolic assignment is not a usable size bound
                if reads_array_data(value, sdfg.arrays):
                    # ``pystr_to_symbolic("A_indptr[i]")`` SUCCEEDS -- it yields a Subscript -- so the
                    # try/except above never rejected a data read. Folding one in makes every shape it
                    # reaches unallocatable by the caller. Refusing here keeps the symbol un-ranged, and
                    # reject_unsizable_scratch then refuses the nest with a reason.
                    continue
                los.setdefault(var, []).append(value)
                his.setdefault(var, []).append(value)

    def resolve(bounds: Dict[str, list], combine: Callable[..., sympy.Expr]) -> Dict[str, sympy.Expr]:

        def r(expr: sympy.Expr, seen: set) -> sympy.Expr:
            for sym in list(expr.free_symbols):
                name = str(sym)
                if name in bounds and name not in seen:
                    parts = [r(b, seen | {name}) for b in bounds[name]]
                    expr = expr.subs(sym, combine(*parts) if len(parts) > 1 else parts[0])
            return expr

        return {v: r(combine(*bs) if len(bs) > 1 else bs[0], {v}) for v, bs in bounds.items()}

    return resolve(los, sympy.Min), resolve(his, sympy.Max)


def max_over_loops(dim: sympy.Expr, lo_of: Dict[str, sympy.Expr], hi_of: Dict[str, sympy.Expr], known: set,
                   arrays: Mapping[str, dace.data.Data]) -> Optional[sympy.Expr]:
    """Largest value a shape dimension takes over the loop variables' ranges, or ``None``.

    Each loop variable is substituted by the endpoint that maximises the dimension: its resolved upper
    bound where the dimension increases in it (``i + 1``), its resolved lower bound where it decreases
    (``M - i - 1``). Monotonicity is the sign of the (constant) derivative; a variable whose slope
    still holds free symbols (``R**i``, slope of unknown sign) leaves the dimension unresolved
    (``None``), as does a widened extent the caller could not evaluate (:func:`sizable`).
    """
    result = dim
    for s in list(dim.free_symbols):
        if str(s) not in hi_of:
            continue
        slope = sympy.diff(dim, s)
        if slope.free_symbols:
            return None  # non-constant slope -> monotonicity undetermined
        result = result.subs(s, hi_of[str(s)] if slope.is_nonnegative else lo_of[str(s)])
    return result if sizable(result, known, arrays) else None


def maxsize_loop_scratch(sdfg: dace.SDFG, symbols: List[str]) -> dace.SDFG:
    """Return an SDFG where a scratch transient sized by loop variables is widened to a caller-sizable
    bound, so it stays a pre-allocated parameter and is addressed with the original ``0:extent`` slices.

    Each shape dimension is replaced by its maximum over the loop ranges (:func:`max_over_loops`); a
    dimension whose extent is not a sound function of the kernel symbols (an FFT ``R**(K-i-1)``, a
    data-dependent CSR span) is left untouched and caught later by :func:`reject_unsizable_scratch`.
    Runs on a copy so the caller is not mutated.
    """
    known = set(symbols)
    # Screen with the two cheap predicates BEFORE walking the CFG. symbol_ranges pystr_to_symbolic's
    # every loop condition, init statement and interstate assignment, and on 22 of 34 measured kernels
    # nothing survives the filters below, so all of that work was discarded -- 36% of this function's
    # total time across that set, and it is called four times per nest.
    candidates = [(name, desc) for name, desc in sdfg.arrays.items()
                  if desc.transient and not is_scalar(desc) and {str(s)
                                                                 for s in desc.free_symbols} - known]
    if not candidates:
        return sdfg

    lo_of, hi_of = symbol_ranges(sdfg)
    resize: Dict[str, tuple] = {}
    for name, desc in candidates:
        new_shape = []
        for dim in desc.shape:
            sdim = sympy.sympify(dim)  # a literal-int dimension has no free symbols to widen
            if {str(s) for s in sdim.free_symbols} - known:
                widened = max_over_loops(sdim, lo_of, hi_of, known, sdfg.arrays)
                new_shape.append(widened if widened is not None else dim)
            else:
                new_shape.append(dim)
        if new_shape != list(desc.shape):
            resize[name] = tuple(new_shape)
    if not resize:
        return sdfg
    out = copy.deepcopy(sdfg)
    for name, shape in resize.items():
        old = out.arrays[name]
        out.arrays[name] = dace.data.Array(old.dtype, shape, transient=True, storage=old.storage)
    return out


def reject_unsizable_scratch(sdfg: dace.SDFG, scratch: List[str], symbols: List[str]) -> None:
    """Refuse a scratch buffer whose shape depends on a symbol the caller cannot know at allocation.

    C-style emission makes every transient array a caller-allocated parameter, so its shape must be
    fixed from the kernel's own symbols (the size arguments). Two ways it is not:

    * a *loop* variable in the extent (a per-stage FFT buffer shaped ``R**i``) -- no fixed size at
      call time;
    * a read of ARRAY DATA in the extent (spmv's CSR span ``A_indptr[M+1] - A_indptr[0]``) -- the
      caller would have to run the kernel to learn how much to allocate for it.

    Judged per dimension by :func:`sizable`, which walks the expression tree. The old
    ``free_symbols``-only test saw the CSR span's residual symbols as exactly the kernel symbols and
    accepted it, because sympy hides the indexed array's name as the Function head.

    Refuse rather than emit a kernel whose signature cannot be satisfied.
    """
    known = set(symbols)
    for name in scratch:
        for dim in sdfg.arrays[name].shape:
            sdim = sympy.sympify(dim)
            if sizable(sdim, known, sdfg.arrays):
                continue
            why = ("reads array data" if reads_array_data(sdim, sdfg.arrays) else
                   f"depends on {sorted({str(s) for s in sdim.free_symbols} - known)} (not kernel symbols)")
            raise UnsupportedNest(f"scratch buffer {name!r} has extent {dim} that {why}; "
                                  "cannot be pre-allocated C-style")


def innermost_loop(block: dace.sdfg.state.ControlFlowBlock) -> Optional[LoopRegion]:
    """The ``LoopRegion`` a ``break`` / ``continue`` inside ``block`` targets, or ``None`` when the block
    has no loop ancestor in its SDFG (walks ``parent_graph`` up to the root)."""
    region = block.parent_graph
    while region is not None:
        if isinstance(region, LoopRegion):
            return region
        region = region.parent_graph
    return None


def has_enclosing_loop(block: dace.sdfg.state.ControlFlowBlock) -> bool:
    """True if ``block`` has a ``LoopRegion`` ancestor within its SDFG."""
    return innermost_loop(block) is not None


def reject_orphan_break_continue(sdfg: dace.SDFG) -> None:
    """Refuse a ``break`` / ``continue`` with no enclosing ``LoopRegion``: the emitted statement would land
    outside any loop -- a wrong target, or a python ``SyntaxError``. Applies to a whole-SDFG kernel too,
    which is why it is separate from :func:`reject_nonexternalizable`'s early-return half."""
    for block in sdfg.all_control_flow_blocks():
        if isinstance(block, (BreakBlock, ContinueBlock)) and not has_enclosing_loop(block):
            raise UnsupportedNest(
                f"nest contains a {type(block).__name__} ({block.label}) whose target loop is outside the "
                "extracted scope; externalize the loop it breaks out of, not an inner nest")


def reject_nonexternalizable(sdfg: dace.SDFG) -> None:
    """Refuse a nest whose control flow cannot be carried into a standalone kernel unchanged.

    Both cases arise when extraction cuts a control-flow *target* out of the nest:

    * a ``ReturnBlock`` returns out of the *enclosing* SDFG. A python ``return`` in the extracted kernel
      would exit only that kernel and let the caller resume the rest of the original program -- the
      original return instead skipped everything after the nest. (A return is still emittable for a
      *whole*-SDFG kernel via :func:`sdfg_to_numpy`, where ``return`` == exit the kernel == exit the
      SDFG.)
    * a ``BreakBlock`` / ``ContinueBlock`` targets its innermost enclosing loop. If that loop was not
      pulled into the nest, the emitted ``break`` / ``continue`` lands outside any loop -- a wrong
      target (or a python ``SyntaxError``). Externalizing must take the whole loop the break exits, not
      an inner sub-nest; a break/continue with no enclosing ``LoopRegion`` in the nest is an illegal cut.

    Refuse rather than silently mis-emit.
    """
    for block in sdfg.all_control_flow_blocks():
        if isinstance(block, ReturnBlock):
            raise UnsupportedNest(f"nest contains an early return ({block.label}); a return out of the enclosing SDFG "
                                  "cannot be externalized into a standalone kernel")
    reject_orphan_break_continue(sdfg)


def render(fn_name: str, args: List[str], body: List[str]) -> str:
    lines = [f"def {fn_name}({', '.join(args)}):"]
    lines += ["    " + bl for bl in body_or_pass(body)]
    return "\n".join(lines) + "\n"


def expand_nested_sdfg_inputs(sdfg: dace.SDFG) -> dace.SDFG:
    """Return an SDFG whose nested-SDFG in/out connectors are widened to the full outer arrays.

    A nested SDFG inside a map is handed per-iteration *slices*; DaCe's ``ExpandNestedSDFGInputs``
    rewrites its descriptors and memlets to reference the whole outer array offset by the map index --
    exactly the form :func:`emit_nested_sdfg` inlines. It is semantics-preserving, so numerics are
    unchanged. Runs on a **copy** (emission is read-only), taken only when there is a nested SDFG to
    widen. When the pass is unavailable this is a no-op and :func:`emit_nested_sdfg` refuses any
    nested SDFG.
    """
    if ExpandNestedSDFGInputs is None:
        return sdfg
    if not any(isinstance(n, nodes.NestedSDFG) for state in sdfg.all_states() for n in state.nodes()):
        return sdfg
    widened = copy.deepcopy(sdfg)
    widened.apply_transformations_repeated(ExpandNestedSDFGInputs)
    return widened


def nest_to_numpy(boundary: Boundary, fn_name: str = "kernel") -> str:
    """Standalone python source ``def <fn_name>(<args>): ...`` for an extracted nest's boundary.

    Signature (all pre-allocated buffers): inputs, then extra outputs, then scratch transients, then
    size symbols. Everything is written in place; there is no return.
    """
    reject_nonexternalizable(boundary.standalone_sdfg)  # early return / orphan break cannot be externalized
    standalone = expand_nested_sdfg_inputs(boundary.standalone_sdfg)
    standalone = maxsize_loop_scratch(standalone, boundary.symbols)
    scratch = scratch_arrays(standalone)
    reject_unsizable_scratch(standalone, scratch, boundary.symbols)
    args = list(boundary.inputs)
    args += [o for o in boundary.outputs if o not in boundary.inputs]
    args += [s for s in scratch if s not in args]
    args += [s for s in boundary.symbols if s not in args]
    return render(fn_name, args, emit_region(standalone, standalone))


def sdfg_to_numpy(sdfg: dace.SDFG, fn_name: str = "kernel") -> str:
    """Standalone python source for a whole SDFG -- the corpus entry point.

    Signature is the SDFG's own arguments (non-transient arrays + ``__return`` + scalars) followed by
    scratch transient buffers and size symbols -- all caller-allocated, all written in place.
    """
    reject_orphan_break_continue(sdfg)  # a return IS emittable here (it exits the kernel == exits the SDFG)
    sdfg = expand_nested_sdfg_inputs(sdfg)
    symbols = [a for a in sdfg.arglist() if a not in sdfg.arrays]
    sdfg = maxsize_loop_scratch(sdfg, symbols)
    data_args = [a for a in sdfg.arglist() if a in sdfg.arrays]
    scratch = scratch_arrays(sdfg)
    reject_unsizable_scratch(sdfg, scratch, symbols)
    args = data_args + scratch + symbols
    return render(fn_name, args, emit_region(sdfg, sdfg))
