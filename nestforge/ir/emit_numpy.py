# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Emit a standalone numpy/python kernel from an extracted nest.

C-style: every array is a pre-allocated buffer parameter written in place; no allocation, no return.
"""

from __future__ import annotations

import ast
import atexit
import copy
import functools
import inspect
import hashlib
import importlib.util
import math
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
from dace.cpf_lowering import C_CTYPE_DTYPES
from dace.frontend.operations import detect_reduction_type
from dace.sdfg import nodes
from dace.sdfg.state import BreakBlock, ConditionalBlock, ContinueBlock, LoopRegion, ReturnBlock
from dace.sdfg.utils import dfs_topological_sort

from nestforge.ir.emit_libnode import (
    UnsupportedLibraryNode,
    emit_library_node,
    index_str,
    is_scalar,
    read_expr,
    scalar_local,
    write_lhs,
)
from nestforge.ir.extract import Boundary

try:
    from dace.transformation.interstate.expand_nested_sdfg_inputs import ExpandNestedSDFGInputs
except ImportError:  # the pass ships only on the DaCe `extended` branch nest-forge targets
    ExpandNestedSDFGInputs = None


class UnsupportedNest(Exception):
    """The nest uses a construct the numpy emitter does not handle."""


def access(sdfg: dace.SDFG, name: str, subset: dace.subsets.Range) -> str:
    """Bare local for a scalar transient, else the indexed buffer element."""
    if scalar_local(sdfg, name):
        return name
    return f"{name}[{index_str(subset)}]"


def connector_pattern(conn_expr: Dict[str, str]) -> Optional[re.Pattern]:
    """Whole-word alternation pattern matching every connector name in ``conn_expr``, or ``None``."""
    if not conn_expr:
        return None
    return re.compile(r"\b(" + "|".join(re.escape(c) for c in sorted(conn_expr, key=len, reverse=True)) + r")\b")


def sub_connectors(code: str, conn_expr: Dict[str, str], pattern: Optional[re.Pattern] = None) -> str:
    """Replace whole-word connector tokens with their expressions, single-pass (no re-substitution)."""
    if not conn_expr:
        return code
    pattern = pattern if pattern is not None else connector_pattern(conn_expr)
    return pattern.sub(lambda m: conn_expr[m.group(0)], code)


#: DaCe dtype cast -> numpy scalar constructor. Fixed-width dtypes only, so a non-dtype ``dace.<attr>``
#: is never rewritten to a nonexistent ``np.<attr>``; ``bool`` maps to ``np.bool_`` (``np.bool`` is gone in NumPy 2).
DACE_DTYPES = {
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
DACE_CAST = re.compile(r"\bdace\.(" + "|".join(DACE_DTYPES) + r")\b")

#: classic single-token C scalar spelling (``double``, ``long``, ``int8_t``, ...) -> numpy scalar
#: constructor, from DaCe's own C-scalar table (``dace.cpf_lowering.C_CTYPE_DTYPES``) -- the same
#: names DaCe's own codegen accepts as a bare cast in tasklet code. Multi-word spellings
#: (``long long``, ``unsigned int``) are dropped: they cannot appear as a python call target.
C_SCALAR_CAST_DTYPES = {name: DACE_DTYPES[dtype] for name, dtype in C_CTYPE_DTYPES.items() if name.isidentifier()}
#: every bare cast name this emitter resolves -> the numpy scalar constructor it becomes.
BARE_CAST_DTYPES = {**DACE_DTYPES, **C_SCALAR_CAST_DTYPES}
#: a bare dtype cast (``int64(x)`` or the classic C spelling ``long(x)``, unprefixed) that
#: ``symbolic.symstr``/DaCe codegen can produce; unmatched, it would raise ``NameError``. The
#: lookbehind skips a qualified ``x.int64(`` attribute.
BARE_CAST = re.compile(r"(?<![\w.])(" + "|".join(sorted(BARE_CAST_DTYPES, key=len, reverse=True)) + r")\s*\(")

#: bare math intrinsic (as DaCe exposes it in tasklet code) -> the numpy function that computes it.
MATH_INTRINSICS = {
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
INTRINSIC_CALL = re.compile(r"(?<![\w.])(" + "|".join(MATH_INTRINSICS) + r")(?=\s*\()")

#: DaCe sympy user-function -> the numpy/python expression computing the same integer value.
#: ``int_floor`` becomes ``//`` because sympy's own floor-division simplify is unsound, not because
#: python lacks an operator; ``int_ceil`` has none and stays a call, bound via :data:`EMITTED_BUILTINS`.
#: ``Max``/``Min`` map to python builtins, not numpy, to keep exact integer range/subscript semantics.
USERFUNC_REWRITES = {
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


#: Names an emitted kernel calls but does not define; :func:`load_emitted` binds them (never hand-roll this dict).
#: ``math`` is bound because a tasklet may spell an intrinsic ``math.<fn>`` (DaCe's frontend accepts it
#: alongside ``dace.*``) that :func:`rewrite_math_prefix` leaves unrewritten -- see :data:`NP_VERBATIM_MATH`.
EMITTED_BUILTINS = {"np": numpy, "math": math, "int_floor": int_floor, "int_ceil": int_ceil}

#: Same names as SOURCE, so a standalone kernel needs no injected namespace; generated from the
#: functions above so an edit to ``int_ceil`` cannot drift from the emitted text.
STANDALONE_PREAMBLE = (
    "import numpy as np\nimport math\n\n\n"
    + "\n\n\n".join(inspect.getsource(fn).strip() for fn in (int_floor, int_ceil))
    + "\n"
)


def standalone_source(fn_name: str, args: List[str], body: List[str]) -> str:
    """Rendered kernel plus the preamble that makes it importable standalone, with no injected namespace."""
    return f"{STANDALONE_PREAMBLE}\n\n{render(fn_name, args, body)}"


@functools.lru_cache(maxsize=None, typed=True)
def emitted_dir() -> Path:
    """Process-lifetime directory for emitted kernel sources; files must outlive the modules loaded from
    them, since ``linecache`` reads the source lazily and a deleted file blanks the traceback."""
    path = Path(tempfile.mkdtemp(prefix="nestforge-emitted-"))
    atexit.register(shutil.rmtree, path, ignore_errors=True)
    return path


def load_emitted(source: str, name: str) -> ModuleType:
    """Import emitted numpy kernel ``source`` as a real module, with :data:`EMITTED_BUILTINS` pre-bound."""
    # Hash, not a counter: CPython keys __pycache__ on (mtime, size), so two kernels written within the
    # same second at the same byte length would silently reuse the first one's bytecode.
    path = emitted_dir() / f"{name}_{hashlib.sha256(source.encode()).hexdigest()[:16]}.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location(f"nestforge_emitted.{name}", path)
    module = importlib.util.module_from_spec(spec)
    module.__dict__.update(EMITTED_BUILTINS)  # bound BEFORE exec: the source references them at call time
    spec.loader.exec_module(module)
    return module


#: qualified ``(dace.)?math.<fn>`` -> its numpy form; anything not listed here or in
#: :data:`MATH_INTRINSICS` is left as ``math.<fn>`` rather than guess a bad name.
NP_VERBATIM_MATH = frozenset({"power", "arcsin", "arccos", "arctan", "arctan2", "maximum", "minimum", "abs"})
MATH_PREFIX_CALL = re.compile(r"\b(?:dace\.)?math\.(\w+)(?=\s*\()")


def apply_call(code: str, name: str, fn: Callable[..., str]) -> str:
    """Rewrite every ``name(arg0, arg1)`` call in ``code`` via ``fn(arg0, arg1)``, matching balanced parens."""
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
                # Only a top-level comma separates arguments; a subscript comma (``a[i, j]``) stays inside one.
                args.append(cur)
                cur = ""
            else:
                cur += ch
            i += 1
        if depth != 0:
            return code  # unbalanced (should not happen on emitted code): leave it for the caller to hit
        args.append(cur)
        code = code[: m.start()] + fn(*(a.strip() for a in args)) + code[i + 1 :]


def rewrite_userfuncs(code: str) -> str:
    """Rewrite DaCe sympy user-functions (:data:`USERFUNC_REWRITES`) to numpy/python, to a fixpoint."""
    for _ in range(16):  # bounded: every rewrite strictly removes one user-function call
        new = code
        for name, fn in USERFUNC_REWRITES.items():
            new = apply_call(new, name, fn)
        if new == code:
            return code
        code = new
    return code


def rewrite_math_prefix(code: str) -> str:
    """Rewrite a qualified ``dace.math.sin`` / ``math.sin`` to ``np.sin``."""

    def repl(m: re.Match[str]) -> str:
        fn = m.group(1)
        if fn in MATH_INTRINSICS:
            return MATH_INTRINSICS[fn]
        if fn in NP_VERBATIM_MATH:
            return f"np.{fn}"
        return f"math.{fn}"

    return MATH_PREFIX_CALL.sub(repl, code)


#: strips a C++ ``decltype(<connector>)`` cast prefix DaCe emits to force a connector's type;
#: numpy promotes types itself, so the cast is a no-op on the parenthesized value it leaves behind.
DECLTYPE_CAST = re.compile(r"\bdecltype\s*\([^()]*\)")


@functools.lru_cache(maxsize=4096, typed=True)
def normalize_casts(code: str) -> str:
    """Rewrite DaCe dtype casts, math intrinsics, and sympy user-functions to numpy, value-preserving."""
    code = DECLTYPE_CAST.sub("", code)
    code = DACE_CAST.sub(lambda m: DACE_DTYPES[m.group(1)], code)
    code = BARE_CAST.sub(lambda m: f"{BARE_CAST_DTYPES[m.group(1)]}(", code)
    # qualified math.* must rewrite before the bare-name pass, whose lookbehind must skip the produced np.sin
    code = rewrite_math_prefix(code)
    code = INTRINSIC_CALL.sub(lambda m: MATH_INTRINSICS[m.group(1)], code)
    return rewrite_userfuncs(code)


#: Every DaCe precondition trap is a connectorless CPP tasklet holding this. Several canonicalization
#: passes emit it (symbol-assumption guards, the scatter-conflict guard, break_anti_dependence,
#: wavefront skew), so match the shape DaCe emits today (``std::abort()``), not a label.
TRAP_GUARD = re.compile(r"^\s*if\s*\((?P<cond>.+)\)\s*\{\s*std::abort\s*\(\s*\)\s*;?\s*\}\s*;?\s*$", re.DOTALL)

#: C spellings with a Python equivalent. ``!`` needs the lookahead so ``!=`` survives intact.
C_TO_PYTHON = (
    (re.compile(r"&&"), " and "),
    (re.compile(r"\|\|"), " or "),
    (re.compile(r"!(?!=)"), " not "),
    (re.compile(r"\btrue\b"), "True"),
    (re.compile(r"\bfalse\b"), "False"),
)


def trap_guard_lines(tasklet: nodes.Tasklet) -> List[str] | None:
    """Python equivalent of a C trap guard (an aborted precondition), or ``None`` if not one."""
    matched = TRAP_GUARD.match(tasklet.code.as_string)
    if matched is None:
        return None
    cond = matched.group("cond")
    for pattern, replacement in C_TO_PYTHON:
        cond = pattern.sub(replacement, cond)
    cond = re.sub(r"\s+", " ", normalize_casts(cond)).strip()  # eval-mode parse rejects a leading space
    try:
        ast.parse(cond, mode="eval")
    except SyntaxError as exc:
        raise UnsupportedNest(
            f"trap guard {tasklet.label} has a condition that is not translatable to python: {cond!r}"
        ) from exc
    return [f"if {cond}:", f"    raise AssertionError({f'violated assumption in {tasklet.label}'!r})"]


@functools.lru_cache(maxsize=4096, typed=True)
def reduction_type(wcr_str: str) -> dace.dtypes.ReductionType:
    """Cached :func:`detect_reduction_type`; the same WCR string repeats across a reduction's edges."""
    return detect_reduction_type(wcr_str)


#: reduction type -> ``(accumulator, term) -> combined expression`` for a WCR (augmented) write.
WCR_BINOP = {
    dace.dtypes.ReductionType.Sum: lambda acc, t: f"{acc} + {t}",
    dace.dtypes.ReductionType.Product: lambda acc, t: f"{acc} * {t}",
    dace.dtypes.ReductionType.Max: lambda acc, t: f"np.maximum({acc}, {t})",
    dace.dtypes.ReductionType.Min: lambda acc, t: f"np.minimum({acc}, {t})",
}


def tasklet_lines(state: dace.SDFGState, sdfg: dace.SDFG, tasklet: nodes.Tasklet) -> List[str]:
    """Tasklet's Python code with connectors substituted by the array element they name.

    :returns: for a WCR (reduction) output, an augmented assignment (body writes a temporary, then
        the target is combined with it) rather than a plain overwrite.
    """
    if not tasklet.in_connectors and not tasklet.out_connectors:
        # No connectors -> no data effect; the only meaningful case is the precondition trap.
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
        combine = WCR_BINOP.get(reduction_type(e.data.wcr))
        if combine is None:
            raise UnsupportedNest(f"tasklet {tasklet.label} has an unsupported WCR {e.data.wcr!r}")
        temp = f"__wcr_{e.src_conn}"
        conn_expr[e.src_conn] = temp  # the body writes the temporary, then we accumulate into target
        wcr_updates.append(f"{target} = {combine(target, temp)}")
    pattern = connector_pattern(conn_expr)
    lines = [normalize_casts(sub_connectors(line, conn_expr, pattern)) for line in tasklet.code.as_string.splitlines()]
    return lines + [normalize_casts(u) for u in wcr_updates]  # a strided subset may render int_floor/int_ceil


def copy_side(sdfg: dace.SDFG, name: str, subset: Optional[dace.subsets.Range]) -> str:
    """One side of a memlet copy as a squeezed view (length-1 axes dropped), matching DaCe's copy order."""
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

    Source is tested first: on an in-place copy both endpoints share one name, so this tie-break order
    (matching DaCe's own ``is_data_src=True`` preference) decides which side ``subset`` describes.
    """
    m = edge.data
    if m.data == edge.src.data:
        return edge.src.data, m.subset, m.other_subset
    if m.data == edge.dst.data:
        return edge.src.data, m.other_subset, m.subset
    raise UnsupportedNest(f"copy memlet {m.data!r} names neither {edge.src.data!r} nor {edge.dst.data!r}")


def copy_lines(state: dace.SDFGState, sdfg: dace.SDFG, dst: nodes.AccessNode) -> List[str]:
    """Emit ``dst[..] = src[..]`` for each memlet copy feeding ``dst`` from an access node or a map entry."""
    lines: List[str] = []
    for e in state.in_edges(dst):
        m = e.data
        if m.is_empty():
            continue  # an empty memlet is a happens-before/ordering edge (StateFusion sequencing), no data
        if isinstance(e.src, nodes.AccessNode):
            src_name, src_sub, dst_sub = copy_direction(e)
        elif isinstance(e.src, nodes.MapEntry):
            # a staged in-map read (b_index = b[i]): the memlet names the outer source, not the scratch dest
            src_name, src_sub, dst_sub = m.data, m.subset, m.other_subset
        else:
            # Tasklet/MapExit source WCRs are emitted at that edge's own owner (tasklet_lines /
            # map_exit_writes); a LibraryNode/NestedSDFG source is refused at its own emitter instead.
            continue
        lhs, rhs, dst_read = copy_sides(sdfg, dst.data, dst_sub, src_name, src_sub)
        if m.wcr is not None:  # a reduction copy (e.g. a privatized accumulator copied back): accumulate
            combine = WCR_BINOP.get(reduction_type(m.wcr))
            if combine is None:
                raise UnsupportedNest(f"reduction (WCR) copy into {dst.data} has an unsupported WCR {m.wcr!r}")
            rhs = combine(dst_read, rhs)
        lines.append(normalize_casts(f"{lhs} = {rhs}"))  # a strided subset may render an int_floor/int_ceil index
    return lines


def copy_sides(
    sdfg: dace.SDFG,
    dst_name: str,
    dst_sub: Optional[dace.subsets.Range],
    src_name: str,
    src_sub: Optional[dace.subsets.Range],
) -> tuple:
    """``(lhs, rhs, dst_read)`` for one data copy -- the shared body of :func:`copy_lines` and
    :func:`map_exit_writes`. ``dst_read`` is the destination rendered for reading (a WCR accumulates into it).
    """
    if len(sdfg.arrays[src_name].shape) == len(sdfg.arrays[dst_name].shape):
        if sdfg.arrays[src_name].shape == sdfg.arrays[dst_name].shape:
            # mirror a missing subset only between equal shapes: a same-rank reshape must keep its own subset
            src_sub = src_sub if src_sub is not None else dst_sub
            dst_sub = dst_sub if dst_sub is not None else src_sub
        return (
            copy_side(sdfg, dst_name, dst_sub),
            copy_side(sdfg, src_name, src_sub),
            copy_side(sdfg, dst_name, dst_sub),
        )
    return (
        reshape_side(sdfg, dst_name, dst_sub, write=True),
        reshape_side(sdfg, src_name, src_sub, write=False),
        reshape_side(sdfg, dst_name, dst_sub, write=False),
    )


def reshape_side(sdfg: dace.SDFG, name: str, subset: Optional[dace.subsets.Range], write: bool) -> str:
    """One side of a rank-changing copy: bare local, explicit index, or the whole array."""
    if scalar_local(sdfg, name):
        return name
    if subset is None:
        return write_lhs(sdfg, name, None) if write else read_expr(sdfg, name, None)
    return f"{name}[{index_str(subset)}]"


def reject_underranked_codeblock_index(inner: dace.SDFG) -> None:
    """Refuse a nested SDFG whose inter-state code indexes a multi-dim array with too few indices.

    ``ExpandNestedSDFGInputs`` offsets an inter-state condition/assignment by only the first map
    dimension, so an under-indexed reference would emit a whole row where an element was meant.
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
                            "in inter-state code -- ExpandNestedSDFGInputs offsets multi-dim conditions incompletely"
                        )


def reconcile_connector_descriptor(inner: dace.SDFG, sdfg: dace.SDFG, outer: str) -> None:
    """Make the inner descriptor for connector array ``outer`` agree with the buffer it aliases.

    The connector adopts the outer descriptor outright: matching shapes alone are not enough, since
    ``transient`` decides bare-local vs. indexed spelling and a connector is never transient inside.
    Shapes that genuinely differ are refused unless both are single-element (a nested-return scalar).
    """
    inner_desc, outer_desc = inner.arrays[outer], sdfg.arrays[outer]
    same_shape = [str(d) for d in inner_desc.shape] == [str(d) for d in outer_desc.shape]
    if not same_shape and not (is_scalar(inner_desc) and is_scalar(outer_desc)):
        raise UnsupportedNest(
            f"nested SDFG connector {outer!r} is {inner_desc.shape} inside but "
            f"{outer_desc.shape} outside; the extents differ, so the inner body indexes a "
            "different shape than the buffer it aliases -- not emittable as numpy"
        )
    inner.arrays[outer] = copy.deepcopy(outer_desc)


def emit_nested_sdfg(state: dace.SDFGState, sdfg: dace.SDFG, node: nodes.NestedSDFG) -> List[str]:
    """Inline a nested SDFG (e.g. one map iteration's sub-kernel) as flat statements, in place."""
    for e in state.out_edges(node):
        if e.data.wcr is not None:
            # This replays the inner body only; it never applies a WCR on the OUTER output edge, which
            # would silently become an overwrite. map_exit_writes guards a map exit; this covers state body too.
            raise UnsupportedNest(
                f"nested SDFG output into {e.data.data} carries a reduction (WCR) that emit_nested_sdfg does "
                "not apply; not emittable as numpy -- fall back to the DaCe variant"
            )
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
    # A private inner transient becomes a plain python local, which only works for a scalar: a private
    # array transient appears in no outer signature and would be emitted undefined.
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
    """Bind a nested SDFG's ``symbol_mapping`` simultaneously (via a temp) when a swap would interfere."""
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
    return [f"{tmp} = {expr}" for tmp, _, expr in temps] + [f"{sym} = {tmp}" for tmp, sym, _ in temps]


def map_exit_writes(state: dace.SDFGState, sdfg: dace.SDFG, entry: nodes.MapEntry) -> List[str]:
    """Writes that leave the map through its exit from an in-scope AccessNode (a privatized reduction)."""
    lines: List[str] = []
    for e in state.in_edges(state.exit_node(entry)):
        if not isinstance(e.src, nodes.AccessNode) or state.entry_node(e.src) is not entry:
            # A nested map/library/NestedSDFG source emits its body only and never applies an exit WCR,
            # which would otherwise silently become an overwrite; refuse so the caller falls back to DaCe.
            if e.data.wcr is not None and not isinstance(e.src, nodes.Tasklet):
                raise UnsupportedNest(
                    f"reduction (WCR) leaves the map exit from a {type(e.src).__name__}, not an in-scope "
                    "accumulator access node; not emittable as numpy -- fall back to the DaCe variant"
                )
            continue
        m = e.data
        dst_name, dst_sub, src_sub = m.data, m.subset, m.other_subset
        src_name = e.src.data
        if src_name == dst_name and m.wcr is None:
            continue  # a plain self-edge moves nothing; a WCR self-edge is an in-place reduction, not a no-op
        lhs, rhs, dst_read = copy_sides(sdfg, dst_name, dst_sub, src_name, src_sub)
        if m.wcr is not None:
            combine = WCR_BINOP.get(reduction_type(m.wcr))
            if combine is None:
                raise UnsupportedNest(f"reduction (WCR) write-out into {dst_name} has an unsupported WCR {m.wcr!r}")
            rhs = combine(dst_read, rhs)
        lines.append(normalize_casts(f"{lhs} = {rhs}"))
    return lines


def range_stop(end: sympy.Expr, step: sympy.Expr, what: str) -> sympy.Expr:
    """Python's exclusive ``range`` stop for a DaCe range whose ``end`` is inclusive: ``end +/- 1`` by
    step sign; a blanket ``+ 1`` would silently drop element 0 on a descending range."""
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
            normalize_casts(
                f"for {param} in range({symbolic.symstr(beg)}, {symbolic.symstr(stop)}, {symbolic.symstr(step)}):"
            )
        )  # a bound may render an int_floor/int_ceil
    return headers


def map_body_lines(state: dace.SDFGState, sdfg: dace.SDFG, entry: nodes.MapEntry) -> List[str]:
    """What one map scope computes, unindented and without its ``for`` headers."""
    body: List[str] = []
    scope = state.scope_subgraph(entry, include_entry=False, include_exit=False)
    for node in dfs_topological_sort(scope):
        # scope_subgraph returns the whole subtree; skip a grandchild here, the recursion below emits it
        if state.entry_node(node) is not entry:
            continue
        if isinstance(node, nodes.Tasklet):
            body.extend(tasklet_lines(state, sdfg, node))
        elif isinstance(node, nodes.AccessNode):
            body.extend(copy_lines(state, sdfg, node))
        elif isinstance(node, nodes.NestedSDFG):
            body.extend(emit_nested_sdfg(state, sdfg, node))
        elif isinstance(node, nodes.MapEntry):
            body.extend(map_lines(state, sdfg, node))  # a map nested in a map -> deeper for loops
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
    """Append ``pass`` if ``lines`` holds only provenance comments, so the block body is non-empty Python."""
    return lines if any(not ln.lstrip().startswith("#") for ln in lines) else lines + ["pass"]


def targets_continue(loop: LoopRegion) -> bool:
    """True if some ``ContinueBlock`` inside ``loop`` targets ``loop`` itself (not an inner loop)."""
    return any(isinstance(b, ContinueBlock) and innermost_loop(b) is loop for b in loop.all_control_flow_blocks())


def emit_loop(loop: LoopRegion, sdfg: dace.SDFG) -> List[str]:
    """Emit a ``LoopRegion`` as init + ``while`` (do-while when ``inverted``) around its body."""
    if loop.loop_condition is None:
        raise UnsupportedNest(f"loop {loop.label} has no condition")
    cond = normalize_casts(loop.loop_condition.as_string.strip())
    init = normalize_casts(loop.init_statement.as_string.strip()) if loop.init_statement is not None else None
    update = normalize_casts(loop.update_statement.as_string.strip()) if loop.update_statement is not None else None
    if loop.inverted and targets_continue(loop):
        # python's while puts the update/exit test in the body, where continue would skip both and hang forever
        raise UnsupportedNest(
            f"loop {loop.label} is inverted (do-while) and contains a continue; the emitted "
            "`while True:` carries its exit test in the body, so a `continue` would skip the "
            "test and loop forever"
        )
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
    """Emit a ``ConditionalBlock`` as ``if``/``elif``/``else`` over its branches, in their stored order."""
    ind = "    "
    lines: List[str] = []
    keyword = "if"
    last = len(cond_block.branches) - 1
    for index, (condition, region) in enumerate(cond_block.branches):
        if condition is None and index != last:  # DaCe codegen itself refuses a non-final unconditional branch
            raise UnsupportedNest(
                f"conditional block {cond_block.label!r} has an unconditional branch at "
                f"position {index} of {last + 1}; DaCe codegen refuses a non-final "
                "unconditional branch, so there is no order to preserve"
            )
        if condition is None:
            lines.append("else:")
        else:
            lines.append(f"{keyword} {normalize_casts(condition.as_string.strip())}:")
            keyword = "elif"
        lines += [ind + b for b in body_or_pass(emit_region(region, sdfg, continue_update))]
    return lines


def strip_scalar_local_subscript(code: str, sdfg: dace.SDFG) -> str:
    """Drop the ``[0]`` index off a scalar-transient array in a raw DaCe code string."""
    for name, desc in sdfg.arrays.items():
        if scalar_local(sdfg, name):
            code = re.sub(rf"\b{re.escape(name)}\s*\[[^][]*\]", name, code)
    return code


def interstate_lines(
    region: dace.sdfg.state.ControlFlowRegion, sdfg: dace.SDFG, block: dace.sdfg.state.ControlFlowBlock
) -> List[str]:
    """Assignments carried on the edge(s) entering ``block`` (e.g. an indirect index ``s = A[i]``)."""
    lines: List[str] = []
    carrying = []
    for e in region.in_edges(block):
        if not e.data.is_unconditional():
            # a conditional inter-state edge is an unstructured goto; straight-line emission cannot model it
            raise UnsupportedNest(
                f"conditional inter-state edge into {block.label} (unstructured goto/branch) is not emitted"
            )
        if e.data.assignments:
            carrying.append(e)
    if len(carrying) > 1:
        # Emitting both would double-apply them at runtime, when only one predecessor actually executes
        raise UnsupportedNest(
            f"{len(carrying)} inter-state edges into {block.label} carry assignments (an unstructured join); "
            "straight-line emission would apply every predecessor's assignments -- not emittable as numpy"
        )
    for e in carrying:
        for lhs, rhs in e.data.assignments.items():
            lines.append(f"{lhs} = {strip_scalar_local_subscript(normalize_casts(rhs), sdfg)}")
    return lines


def emit_region(
    region: dace.sdfg.state.ControlFlowRegion, sdfg: dace.SDFG, continue_update: Optional[str] = None
) -> List[str]:
    """Numpy statements for every block of a control-flow region, in execution order.

    :param continue_update: the enclosing loop's update statement, emitted in front of each
        ``continue`` since python's ``while`` keeps the update in the body.
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


#: sympy function heads meaning "this expression reads array data" (DaCe renders ``A[i]`` as ``Subscript(A, i)``).
DATA_READ_HEADS = frozenset({"Subscript", "Indexed"})


def reads_array_data(expr: sympy.Expr, arrays: Mapping[str, dace.data.Data]) -> bool:
    """Whether ``expr`` reads the CONTENTS of an array, so its value is unknown until the kernel runs.

    Walks the expression tree; ``free_symbols`` alone is blind to this (an indexed array's name is
    the Function head, e.g. ``A_indptr[i]`` has free symbols ``{i}``, not ``A_indptr``).
    """
    for fn in expr.atoms(sympy.Function):
        if fn.func.__name__ in DATA_READ_HEADS or fn.func.__name__ in arrays:
            return True
    return any(str(s) in arrays for s in expr.free_symbols)


def sizable(expr: sympy.Expr, known: set, arrays: Mapping[str, dace.data.Data]) -> bool:
    """Whether the CALLER can evaluate ``expr`` to a buffer extent before the kernel runs.

    True iff it reads no array data (:func:`reads_array_data`) and names no symbol outside ``known``;
    a ``free_symbols``-only check misses the array-data case (see :func:`reads_array_data`).
    """
    if reads_array_data(expr, arrays):
        return False
    return not {str(s) for s in expr.free_symbols} - known


def loop_init_value(loop: LoopRegion) -> sympy.Basic:
    """The loop variable's initial value from ``init_statement``, or ``0`` when the loop has none."""
    if loop.init_statement is None:
        return sympy.Integer(0)
    text = loop.init_statement.as_string
    if "=" not in text:
        raise UnsupportedNest(f"loop {loop.label!r} has an init statement {text!r} with no assignment")
    return symbolic.pystr_to_symbolic(text.split("=", 1)[1])


def symbol_ranges(sdfg: dace.SDFG) -> tuple:
    """``(lo_of, hi_of)``: each non-argument symbol -> its min / max value, in kernel symbols.

    A symbol's range comes from a loop's ``[init, condition-bound]`` and/or every value an inter-state
    edge assigns it; several sources take ``Min``/``Max``, resolved recursively into kernel symbols.
    """
    los: Dict[str, list] = {}
    his: Dict[str, list] = {}
    for cfg in sdfg.all_control_flow_regions():
        if isinstance(cfg, LoopRegion) and cfg.loop_condition is not None:
            rel = symbolic.pystr_to_symbolic(cfg.loop_condition.as_string)
            var = cfg.loop_variable
            if isinstance(rel, (sympy.StrictLessThan, sympy.LessThan)) and str(rel.lhs) == var:
                his.setdefault(var, []).append(rel.rhs + (1 if isinstance(rel, sympy.LessThan) else 0))
                los.setdefault(var, []).append(loop_init_value(cfg))
        for e in cfg.edges():
            for var, rhs in e.data.assignments.items():
                try:
                    value = symbolic.pystr_to_symbolic(rhs)  # the config symbol takes exactly this value
                except Exception:
                    continue  # a non-symbolic assignment is not a usable size bound
                if reads_array_data(value, sdfg.arrays):
                    continue  # pystr_to_symbolic parses a data read fine; leave the symbol un-ranged instead
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


def max_over_loops(
    dim: sympy.Expr,
    lo_of: Dict[str, sympy.Expr],
    hi_of: Dict[str, sympy.Expr],
    known: set,
    arrays: Mapping[str, dace.data.Data],
) -> Optional[sympy.Expr]:
    """Largest value a shape dimension takes over the loop variables' ranges, or ``None`` if unresolved."""
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
    """Widen a scratch transient sized by loop variables to a caller-sizable bound; runs on a copy."""
    known = set(symbols)
    # cheap filter before symbol_ranges walks the whole CFG: most kernels have no such candidate
    candidates = [
        (name, desc)
        for name, desc in sdfg.arrays.items()
        if desc.transient and not is_scalar(desc) and {str(s) for s in desc.free_symbols} - known
    ]
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
    """Refuse a scratch buffer whose extent is a loop variable or reads array data, per :func:`sizable`."""
    known = set(symbols)
    for name in scratch:
        for dim in sdfg.arrays[name].shape:
            sdim = sympy.sympify(dim)
            if sizable(sdim, known, sdfg.arrays):
                continue
            why = (
                "reads array data"
                if reads_array_data(sdim, sdfg.arrays)
                else f"depends on {sorted({str(s) for s in sdim.free_symbols} - known)} (not kernel symbols)"
            )
            raise UnsupportedNest(
                f"scratch buffer {name!r} has extent {dim} that {why}; cannot be pre-allocated C-style"
            )


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
    """Refuse a ``break`` / ``continue`` with no enclosing ``LoopRegion``: it would land outside any loop."""
    for block in sdfg.all_control_flow_blocks():
        if isinstance(block, (BreakBlock, ContinueBlock)) and not has_enclosing_loop(block):
            raise UnsupportedNest(
                f"nest contains a {type(block).__name__} ({block.label}) whose target loop is outside the "
                "extracted scope; externalize the loop it breaks out of, not an inner nest"
            )


def reject_nonexternalizable(sdfg: dace.SDFG) -> None:
    """Refuse a nest with an early return (exits the enclosing SDFG, not just the nest) or an orphan break/continue."""
    for block in sdfg.all_control_flow_blocks():
        if isinstance(block, ReturnBlock):
            raise UnsupportedNest(
                f"nest contains an early return ({block.label}); a return out of the enclosing SDFG "
                "cannot be externalized into a standalone kernel"
            )
    reject_orphan_break_continue(sdfg)


def render(fn_name: str, args: List[str], body: List[str]) -> str:
    lines = [f"def {fn_name}({', '.join(args)}):"]
    lines += ["    " + bl for bl in body_or_pass(body)]
    return "\n".join(lines) + "\n"


def expand_nested_sdfg_inputs(sdfg: dace.SDFG) -> dace.SDFG:
    """Return an SDFG (a copy) whose nested-SDFG in/out connectors are widened to the full outer arrays."""
    if ExpandNestedSDFGInputs is None:
        return sdfg
    if not any(isinstance(n, nodes.NestedSDFG) for state in sdfg.all_states() for n in state.nodes()):
        return sdfg
    widened = copy.deepcopy(sdfg)
    widened.apply_transformations_repeated(ExpandNestedSDFGInputs)
    return widened


def nest_to_numpy(boundary: Boundary, fn_name: str = "kernel") -> str:
    """Standalone python source ``def <fn_name>(<args>): ...`` for an extracted nest's boundary."""
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
    """Standalone python source for a whole SDFG -- the corpus entry point."""
    reject_orphan_break_continue(sdfg)  # a return IS emittable here (it exits the kernel == exits the SDFG)
    sdfg = expand_nested_sdfg_inputs(sdfg)
    symbols = [a for a in sdfg.arglist() if a not in sdfg.arrays]
    sdfg = maxsize_loop_scratch(sdfg, symbols)
    data_args = [a for a in sdfg.arglist() if a in sdfg.arrays]
    scratch = scratch_arrays(sdfg)
    reject_unsizable_scratch(sdfg, scratch, symbols)
    args = data_args + scratch + symbols
    return render(fn_name, args, emit_region(sdfg, sdfg))
