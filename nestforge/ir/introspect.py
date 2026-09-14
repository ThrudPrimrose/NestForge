# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Read-only structure inspection for the agent: ``describe_graph`` renders the SDFG as an ASCII
tree of regions/loops/kernels (each named by its canonical normal-form label); ``nest_reads_writes``
reports one nest's arrays without extracting it."""
from __future__ import annotations

import ast
import functools
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import dace
from dace import dtypes
from dace.frontend.operations import detect_reduction_type
from dace.sdfg import nodes
from dace.sdfg.state import ConditionalBlock, ControlFlowBlock, ControlFlowRegion, LoopRegion, SDFGState
from dace.frontend.python import astutils
from dace.transformation.passes.analysis import loop_analysis

from nestforge.ir.emit_libnode import UnsupportedLibraryNode
from nestforge.ir.emit_numpy import UnsupportedNest, map_body_lines, map_lines, standalone_source
from nestforge.ir.names import in_order

#: Tree drawing: the guide under a node that has siblings below it, and the one under the last child.
TEE, ELBOW, PIPE, BLANK = "|- ", "`- ", "|  ", "   "

#: Marks a numpy body line, so a statement is never mistaken for a tree row.
BODY = ": "

#: What a ``Handle`` is asked to name. ``region`` covers every control-flow block, ``nest`` every map.
Handle = Callable[[str, object], str]

#: The suffix appended to a top-level map's kernel line, when metrics are asked for.
Metrics = Callable[[nodes.MapEntry], str]


class Substitute(ast.NodeTransformer):
    """Replace each ``Name`` that has a definition with that definition's expression."""

    __slots__ = ("definitions", )

    def __init__(self, definitions: Dict[str, str]) -> None:
        self.definitions = definitions

    def visit_Name(self, node: ast.Name) -> ast.AST:
        expression = self.definitions.get(node.id)
        return ast.parse(expression, mode="eval").body if expression is not None else node


def interstate_definitions(sdfg: dace.SDFG) -> Dict[str, str]:
    """``name -> expression`` for every interstate assignment in the SDFG; a name assigned more than
    one distinct expression is dropped (which one reaches a block depends on the path taken)."""
    assigned: Dict[str, set] = {}
    for cfg in sdfg.all_control_flow_regions(recursive=True):
        for edge in cfg.edges():
            for name, expression in edge.data.assignments.items():
                assigned.setdefault(name, set()).add(expression)
    return {name: exprs.pop() for name, exprs in assigned.items() if len(exprs) == 1}


def resolve_scalars(expression: str, definitions: Dict[str, str]) -> str:
    """Fold scalar definitions into ``expression`` until only arrays, non-transients and free symbols
    are left -- ``A_index > 0.0`` becomes ``A[i + 1] > 0.0``. Each name is substituted at most once,
    so a cyclic definition (``i = i + 1`` on a back edge) terminates rather than expanding forever."""
    if not definitions:
        return expression
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:  # a condition the frontend wrote in something other than python
        return expression
    remaining = dict(definitions)
    while remaining:
        used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} & set(remaining)
        if not used:
            break
        tree = Substitute({name: remaining.pop(name) for name in used}).visit(tree)
    # ast.unparse, not astutils.unparse: this string is for a human/agent to READ, not re-parsed.
    return ast.unparse(simplify_indices(tree)).strip()


@functools.lru_cache(maxsize=None, typed=True)
def simplified_index(text: str) -> str:
    """Cached sympy round-trip for one subscript's unparsed slice text."""
    return str(dace.symbolic.simplify(dace.symbolic.pystr_to_symbolic(text)))


def simplify_indices(tree: ast.AST) -> ast.AST:
    """Rewrite every subscript index through sympy, so a hoisted read prints ``A[i + 1]`` rather than
    the ``A[(1 + (1 * i))]`` the frontend builds it as."""
    subscripts = [node for node in ast.walk(tree) if isinstance(node, ast.Subscript)]
    if not subscripts:
        return tree
    for node in subscripts:
        try:
            node.slice = ast.parse(simplified_index(astutils.unparse(node.slice)), mode="eval").body
        except (SyntaxError, TypeError, AttributeError):
            continue  # an index sympy will not take is still perfectly printable as it stands
    return ast.fix_missing_locations(tree)


def kernel_body(state: SDFGState, sdfg: dace.SDFG, entry: nodes.MapEntry, children: Dict) -> List[str]:
    """The numpy statements one kernel computes, without its ``for`` headers. Only a LEAF kernel gets
    a body -- a kernel containing another is rendered with that one as its own child row -- and an
    emitter refusal is reported on the line rather than raised, since the tree is read-only.

    :param children: the caller's ``scope_children()``, passed in so a hundred-kernel state does not
        rebuild the same scope tree once per kernel."""
    if any(isinstance(node, nodes.MapEntry) for node in children[entry]):
        return []
    try:
        return map_body_lines(state, sdfg, entry)
    except (UnsupportedNest, UnsupportedLibraryNode) as exc:
        return [f"<not emitted: {exc}>"]


def kernel_args(state: SDFGState, entry: nodes.MapEntry) -> List[str]:
    """One kernel's parameters, sorted: the arrays it touches, then the symbols its domain needs."""
    reads, writes = nest_reads_writes(state, entry)
    arrays = sorted(set(reads) | set(writes))
    symbols = sorted({str(sym) for sym in entry.map.range.free_symbols} - set(arrays))
    return arrays + symbols


def kernel_source(state: SDFGState, sdfg: dace.SDFG, entry: nodes.MapEntry) -> str:
    """ONE kernel as a complete, runnable numpy module: the loop nest inside a ``def`` with a real
    signature, on top of a preamble defining everything the body calls -- unlike ``kernel_body``'s
    fragment, this can be pasted into a file, executed, and checked against the SDFG."""
    return standalone_source(entry.map.label, kernel_args(state, entry), map_lines(state, sdfg, entry))


#: ``ReductionType`` -> how the tree spells it; anything absent renders its lowercased enum name.
REDUCTION_SPELLING = {
    dtypes.ReductionType.Sum: "+",
    dtypes.ReductionType.Product: "*",
    dtypes.ReductionType.Min: "min",
    dtypes.ReductionType.Max: "max",
    dtypes.ReductionType.Sub: "-",
    dtypes.ReductionType.Div: "/",
    dtypes.ReductionType.Logical_And: "and",
    dtypes.ReductionType.Logical_Or: "or",
    dtypes.ReductionType.Logical_Xor: "xor",
    dtypes.ReductionType.Bitwise_And: "&",
    dtypes.ReductionType.Bitwise_Or: "|",
    dtypes.ReductionType.Bitwise_Xor: "^",
}


def kernel_reductions(state: SDFGState, entry: nodes.MapEntry) -> List[str]:
    """Every reduction leaving this map, as ``<op> over <axes> -> <target>``. The reduced axes are the
    map parameters the OUTPUT subset does not mention -- a map over ``(i0, i1)`` writing ``C[i0]`` has
    collapsed ``i1``."""
    exit_node = state.exit_node(entry)
    params = set(entry.map.params)
    out: List[str] = []
    # IN-edges of the exit: NormalizeWCRSource guarantees a WCR rides AccessNode -[wcr]-> MapExit.
    for edge in state.in_edges(exit_node):
        if edge.data is None or edge.data.wcr is None:
            continue  # cheapest test first: most exit edges carry no WCR at all
        kind = detect_reduction_type(edge.data.wcr)
        op = REDUCTION_SPELLING.get(kind, kind.name.lower() if kind is not None else "?")
        written = {
            str(s)
            for r in (edge.data.subset.ranges if edge.data.subset else [])
            for b in r
            for s in dace.symbolic.pystr_to_symbolic(b).free_symbols
        }
        collapsed = [p for p in entry.map.params if p in params - written]
        over = ", ".join(collapsed) if collapsed else "-"
        out.append(f"{op} over {over} -> {edge.data.data}")
    return out


def nest_reads_writes(container: SDFGState, node: nodes.Node) -> Tuple[List[str], List[str]]:
    """Arrays a nest reads and writes (the interface arrays), without outlining it. ``container`` is the
    ``SDFGState`` holding a ``MapEntry``; ignored for a ``LoopRegion`` (which carries its own states)."""
    if isinstance(node, nodes.MapEntry):
        exit_node = container.exit_node(node)
        reads = sorted({e.data.data for e in container.in_edges(node) if e.data is not None and e.data.data})
        writes = sorted({e.data.data for e in container.out_edges(exit_node) if e.data is not None and e.data.data})
        return reads, writes
    if isinstance(node, LoopRegion):
        reads, writes = node.read_and_write_sets()
        return sorted(reads), sorted(writes)
    raise TypeError(f"not a nest node: {type(node).__name__}")


def map_domain(entry: nodes.MapEntry) -> str:
    """A map's iteration domain, ``i=0:N, j=0:M``."""
    return ", ".join(f"{p}={render_range(r)}" for p, r in zip(entry.map.params, entry.map.range))


def loop_domain(loop: LoopRegion, defs: Dict[str, str]) -> str:
    """A loop's iteration domain, map-shaped, or its resolved condition for an uncounted ``while``."""
    start = loop_analysis.get_init_assignment(loop)
    end = loop_analysis.get_loop_end(loop)
    stride = loop_analysis.get_loop_stride(loop)
    if loop.loop_variable and start is not None and end is not None:
        return f"{loop.loop_variable}={render_range((start, end, stride if stride is not None else 1))}"
    return resolve_scalars(loop.loop_condition.as_string, defs) if loop.loop_condition is not None else ""


#: ``str(end) -> simplify(end + 1)``, keyed on the string form (stable across equal sympy objects).
_END_PLUS_ONE: Dict[str, Any] = {}


def render_range(rng: Tuple[Any, Any, Any]) -> str:
    """``begin:end:step`` with the two redundant parts dropped -- an inclusive end is rendered as the
    exclusive bound a reader expects, and a unit step is left off."""
    begin, end, step = rng
    key = str(end)
    stop = _END_PLUS_ONE.get(key)
    if stop is None:
        stop = dace.symbolic.simplify(end + 1)
        _END_PLUS_ONE[key] = stop
    text = f"{begin}:{stop}"
    return text if step == 1 else f"{text}:{step}"


def describe_graph(sdfg: dace.SDFG,
                   handle: Optional[Handle] = None,
                   bodies: bool = False,
                   metrics: Optional[Metrics] = None) -> str:
    """The SDFG as an ASCII tree for the agent. Each line is one block or kernel; the guides show
    nesting. ``handle(kind, obj)``, when given, returns the session id to stamp on that line,
    ``bodies=True`` also prints what each leaf kernel computes, as numpy, under its line, and
    ``metrics(entry)`` is appended to every top-level map's line."""
    lines: List[str] = [f"SDFG '{sdfg.label}'"]
    walk_regions(sdfg, "", lines, handle, interstate_definitions(sdfg), bodies, metrics)
    return "\n".join(lines)


def stamp(text: str, handle: Optional[Handle], kind: str, obj: object) -> str:
    """Prefix a line's body with its session id, when there is one to prefix."""
    return f"[{handle(kind, obj)}] {text}" if handle is not None else text


def walk_regions(cfg: Union[dace.SDFG, ControlFlowRegion], prefix: str, lines: List[str], handle: Optional[Handle],
                 defs: Dict[str, str], bodies: bool, metrics: Optional[Metrics]) -> None:
    """Render one CFG's blocks under ``prefix``, recursing."""
    blocks = in_order(cfg)
    for index, block in enumerate(blocks):
        last = index == len(blocks) - 1
        lines.append(prefix + (ELBOW if last else TEE) + stamp(block_line(block, defs), handle, "region", block))
        below = prefix + (BLANK if last else PIPE)
        if isinstance(block, SDFGState):
            walk_state(block, below, lines, handle, bodies, metrics)
        elif isinstance(block, ConditionalBlock):
            walk_branches(block, below, lines, handle, defs, bodies, metrics)
        elif isinstance(block, ControlFlowRegion):
            walk_regions(block, below, lines, handle, defs, bodies, metrics)


def walk_branches(block: ConditionalBlock, prefix: str, lines: List[str], handle: Optional[Handle],
                  defs: Dict[str, str], bodies: bool, metrics: Optional[Metrics]) -> None:
    """A conditional's branches, in stored order (the first matching one wins, so that is execution order)."""
    for index, (condition, branch) in enumerate(block.branches):
        last = index == len(block.branches) - 1
        tag = "else" if condition is None else f"when {resolve_scalars(condition.as_string, defs)}"
        body = stamp(f"{branch.label}  {tag}", handle, "region", branch)
        lines.append(prefix + (ELBOW if last else TEE) + body)
        walk_regions(branch, prefix + (BLANK if last else PIPE), lines, handle, defs, bodies, metrics)


def walk_state(state: SDFGState, prefix: str, lines: List[str], handle: Optional[Handle], bodies: bool,
               metrics: Optional[Metrics]) -> None:
    """A state's kernels: every map nest plus any library node, nested scopes recursed into."""
    children = state.scope_children()
    if not any(isinstance(n, (nodes.MapEntry, nodes.LibraryNode)) for n in children[None]):
        return  # a state with no kernels: do not pay for the topological order nobody will read
    rank = {id(n): i for i, n in enumerate(in_order(state))}

    def descend(scope: Optional[nodes.MapEntry], pad: str) -> None:
        kernels = [
            n for n in sorted(children[scope], key=lambda n: rank.get(id(n), 0))
            if isinstance(n, (nodes.MapEntry, nodes.LibraryNode))
        ]
        for index, node in enumerate(kernels):
            last = index == len(kernels) - 1
            below = pad + (BLANK if last else PIPE)
            text = kernel_line(state, node)
            if metrics is not None and scope is None and isinstance(node, nodes.MapEntry):
                text = f"{text}  {metrics(node)}"
            lines.append(pad + (ELBOW if last else TEE) + stamp(text, handle, "nest", node))
            if isinstance(node, nodes.MapEntry):
                if bodies:
                    lines.extend(below + BODY + line for line in kernel_body(state, state.sdfg, node, children))
                descend(node, below)

    descend(None, prefix)


def block_line(block: ControlFlowBlock, defs: Dict[str, str]) -> str:
    """One control-flow block's line: its canonical label, plus its domain or condition."""
    if isinstance(block, LoopRegion):
        domain = loop_domain(block, defs)
        return f"{block.label}  {domain}" if domain else block.label
    return block.label


def kernel_line(state: SDFGState, node: nodes.Node) -> str:
    """One kernel's line: label, iteration domain, and the arrays it reads and writes."""
    if isinstance(node, nodes.LibraryNode):
        reads = sorted({e.data.data for e in state.in_edges(node) if e.data is not None and e.data.data})
        writes = sorted({e.data.data for e in state.out_edges(node) if e.data is not None and e.data.data})
        return f"{node.label}  LIBNODE  reads={reads} writes={writes}"
    reads, writes = nest_reads_writes(state, node)
    reductions = kernel_reductions(state, node)
    folds = f"  reduce=({'; '.join(reductions)})" if reductions else ""
    # a Map is data-parallel by definition, so no parallel/sequential column is needed here
    return f"{node.map.label}  [{map_domain(node)}]{folds}  reads={reads} writes={writes}"
