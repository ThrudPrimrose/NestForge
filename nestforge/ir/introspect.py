# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Read-only structure inspection for the agent (and the deterministic path).

Two views, both non-mutating -- safe to call at any point in a session:
  * :func:`describe_graph` -- the SDFG as an ASCII TREE: nested regions/loops/conditionals, each State,
    each map-nest with its normalized iteration domain and read/write arrays.
  * :func:`nest_reads_writes` -- the arrays a single nest reads and writes, without extracting it.

The tree is the agent's whole view of the program, so it is projected from the normal form
(:mod:`nestforge.ir.names`): every line names a block or kernel by its canonical
``<kind><level>_<index>`` label, and every loop and map shows a ``0:trip:1`` domain. Pass ``handle``
to also stamp each actionable line with a session id, so READING the tree and ACTING on it use one
vocabulary rather than two views the agent has to join by eyeballing labels.

Every line carries what the agent needs to act on that line and nothing else. Facts that hold for a
whole KIND of line -- that map fusion never crosses a State, that a conditional's selector stays in the
core SDFG -- belong in the phase skills, not repeated on every row of a hundred-kernel tree.

A condition is shown in terms of the ARRAYS it really reads. The frontend hoists a scalar read out to
an interstate assignment (``A_index = A[1 + i]``) and the branch then tests a name that means nothing
on its own, so :func:`resolve_scalars` folds those definitions back in until only arrays,
non-transients and free symbols are left.
"""
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


class Substitute(ast.NodeTransformer):
    """Replace each ``Name`` that has a definition with that definition's expression."""

    __slots__ = ("definitions", )

    def __init__(self, definitions: Dict[str, str]) -> None:
        self.definitions = definitions

    def visit_Name(self, node: ast.Name) -> ast.AST:
        expression = self.definitions.get(node.id)
        return ast.parse(expression, mode="eval").body if expression is not None else node


def interstate_definitions(sdfg: dace.SDFG) -> Dict[str, str]:
    """``name -> expression`` for every interstate assignment in the SDFG.

    A name assigned more than one DISTINCT expression is dropped: which one reaches a given block
    depends on the path taken, so folding either into a condition would show something the program
    does not always evaluate.
    """
    assigned: Dict[str, set] = {}
    for cfg in sdfg.all_control_flow_regions(recursive=True):
        for edge in cfg.edges():
            for name, expression in edge.data.assignments.items():
                assigned.setdefault(name, set()).add(expression)
    return {name: exprs.pop() for name, exprs in assigned.items() if len(exprs) == 1}


def resolve_scalars(expression: str, definitions: Dict[str, str]) -> str:
    """Fold scalar definitions into ``expression`` until only arrays, non-transients and free symbols
    are left -- ``A_index > 0.0`` becomes ``A[i + 1] > 0.0``.

    Each name is substituted at most ONCE. That terminates on a self-referential or cyclic definition
    (``i = i + 1`` on a back edge is ordinary), and it bounds a chain to the number of definitions
    rather than letting one expand exponentially.
    """
    if not definitions:
        return expression  # before the parse: nothing can be folded in, so nothing needs an AST
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
    # ast.unparse, not astutils.unparse: this string is for a human/agent to READ, and astutils
    # parenthesizes defensively (``A[(i + 1)]``) because its output is meant to be re-parsed as
    # tasklet code. Nothing re-parses a tree line.
    return ast.unparse(simplify_indices(tree)).strip()


@functools.lru_cache(maxsize=None, typed=True)
def simplified_index(text: str) -> str:
    """Cached sympy round-trip for one subscript's unparsed slice text (a hundred-kernel tree re-walks
    the same handful of index shapes -- ``i + 1``, ``2 * j`` -- over and over)."""
    return str(dace.symbolic.simplify(dace.symbolic.pystr_to_symbolic(text)))


def simplify_indices(tree: ast.AST) -> ast.AST:
    """Rewrite every subscript index through sympy, so a hoisted read prints ``A[i + 1]`` rather than
    the ``A[(1 + (1 * i))]`` the frontend builds it as."""
    subscripts = [node for node in ast.walk(tree) if isinstance(node, ast.Subscript)]
    if not subscripts:
        return tree  # the common case: no index to rewrite, so no sympy round-trip and no fixup walk
    for node in subscripts:
        try:
            node.slice = ast.parse(simplified_index(astutils.unparse(node.slice)), mode="eval").body
        except (SyntaxError, TypeError, AttributeError):
            continue  # an index sympy will not take is still perfectly printable as it stands
    return ast.fix_missing_locations(tree)


def kernel_body(state: SDFGState, sdfg: dace.SDFG, entry: nodes.MapEntry, children: Dict) -> List[str]:
    """The numpy statements one kernel computes, without its ``for`` headers -- the kernel line already
    shows the domain those headers iterate.

    Only a LEAF kernel gets a body. A kernel containing another is rendered with that one as its own
    child row, and ``map_lines`` recurses, so emitting here too would print the inner kernel twice.

    An emitter refusal is reported on the line rather than raised: the tree is a read-only view, and a
    nest the numpy projection cannot express is exactly what the agent needs to be told about.

    ``children`` is the caller's ``scope_children()``, passed in rather than rebuilt: this runs once
    per kernel, and a hundred-kernel state would otherwise construct the same scope tree a hundred
    times.
    """
    if any(isinstance(node, nodes.MapEntry) for node in children[entry]):
        return []
    try:
        return map_body_lines(state, sdfg, entry)
    except (UnsupportedNest, UnsupportedLibraryNode) as exc:
        return [f"<not emitted: {exc}>"]


def kernel_args(state: SDFGState, entry: nodes.MapEntry) -> List[str]:
    """One kernel's parameters: the arrays it touches, then the symbols its domain needs. Sorted, so
    two runs over one kernel produce the same signature."""
    reads, writes = nest_reads_writes(state, entry)
    arrays = sorted(set(reads) | set(writes))
    symbols = sorted({str(sym) for sym in entry.map.range.free_symbols} - set(arrays))
    return arrays + symbols


def kernel_source(state: SDFGState, sdfg: dace.SDFG, entry: nodes.MapEntry) -> str:
    """ONE kernel as a complete, runnable numpy module.

    This is the kernel's REPRESENTATION, as against :func:`kernel_body`, which is the compact excerpt
    the tree prints under a line. The difference matters: a body alone references loop variables that
    do not exist outside their headers, so it is a fragment -- not something an agent can run, and not
    something a translator can read. This returns the loop nest inside a ``def`` with a real signature,
    on top of a preamble that defines everything the body calls. Paste it in a file and it runs.

    That it RUNS is also what makes it checkable: emit, execute, compare against the SDFG. A
    representation nothing can execute cannot be shown to be correct.
    """
    return standalone_source(entry.map.label, kernel_args(state, entry), map_lines(state, sdfg, entry))


#: ``ReductionType`` -> how the tree spells it. Anything absent renders its enum name lowercased, so
#: an op with no infix spelling (``min_location``) still reads, and ``custom`` still says what it is.
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
    """Every reduction leaving this map, as ``<op> over <axes> -> <target>``.

    A WCR on the map's exit IS a tree reduction: the map declares its iterations independent, so the
    fold order is unspecified and a backend may use a register accumulator or an OpenMP ``reduction``
    clause. That is a structural fact about the kernel, and the agent should not have to read the body
    to find it.

    The reduced axes are the map parameters the OUTPUT subset does not mention -- a map over
    ``(i0, i1)`` writing ``C[i0]`` has collapsed ``i1``. ``normalize.NormalizeWCR`` /
    ``NormalizeWCRSource`` are what make this readable at all: without them a reduction can sit inside
    a nested SDFG or source from a tasklet, and there is no single edge to ask.
    """
    exit_node = state.exit_node(entry)
    params = set(entry.map.params)
    out: List[str] = []
    # IN-edges of the exit: `NormalizeWCRSource` guarantees a WCR sources from an AccessNode, so the
    # reduction rides `AccessNode -[wcr]-> MapExit`. The exit's OUT edges are the plain copy onward.
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
    """A map's iteration domain, ``i=0:N, j=0:M``. Normalized maps are zero-based and unit-stride, so a
    step only ever shows up when the caller skipped normalization -- and then it should show."""
    return ", ".join(f"{p}={render_range(r)}" for p, r in zip(entry.map.params, entry.map.range))


def loop_domain(loop: LoopRegion, defs: Dict[str, str]) -> str:
    """A loop's iteration domain in the same shape as a map's, or its condition when the loop is not a
    counted one (a ``while`` has no start/end to show). Only the condition goes through
    :func:`resolve_scalars` -- a domain is not an expression."""
    start = loop_analysis.get_init_assignment(loop)
    end = loop_analysis.get_loop_end(loop)
    stride = loop_analysis.get_loop_stride(loop)
    if loop.loop_variable and start is not None and end is not None:
        return f"{loop.loop_variable}={render_range((start, end, stride if stride is not None else 1))}"
    return resolve_scalars(loop.loop_condition.as_string, defs) if loop.loop_condition is not None else ""


#: ``str(end) -> simplify(end + 1)`` -- the same handful of map/loop bounds recur across a hundred-kernel
#: tree, so keying on the string form (cheap, and stable across structurally-equal sympy objects built by
#: different call sites) skips the sympy simplify+add for every repeat.
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


def describe_graph(sdfg: dace.SDFG, handle: Optional[Handle] = None, bodies: bool = False) -> str:
    """The SDFG as an ASCII tree for the agent. Each line is one block or kernel; the guides show
    nesting. ``handle(kind, obj)``, when given, returns the session id to stamp on that line.

    ``bodies=True`` prints what each leaf kernel COMPUTES, as numpy, under its line -- the second
    projection of the same SDFG. It is off by default because it costs an emit per kernel, and the
    structure alone is what a fusion decision needs.
    """
    lines: List[str] = [f"SDFG '{sdfg.label}'"]
    walk_regions(sdfg, "", lines, handle, interstate_definitions(sdfg), bodies)
    return "\n".join(lines)


def stamp(text: str, handle: Optional[Handle], kind: str, obj: object) -> str:
    """Prefix a line's body with its session id, when there is one to prefix."""
    return f"[{handle(kind, obj)}] {text}" if handle is not None else text


def walk_regions(cfg: Union[dace.SDFG, ControlFlowRegion], prefix: str, lines: List[str], handle: Optional[Handle],
                 defs: Dict[str, str], bodies: bool) -> None:
    """Render one CFG's blocks under ``prefix``, recursing. ``prefix`` carries the guides of every
    ancestor, so a child knows whether to draw a pipe or a blank beneath each of them."""
    blocks = in_order(cfg)
    for index, block in enumerate(blocks):
        last = index == len(blocks) - 1
        lines.append(prefix + (ELBOW if last else TEE) + stamp(block_line(block, defs), handle, "region", block))
        below = prefix + (BLANK if last else PIPE)
        if isinstance(block, SDFGState):
            walk_state(block, below, lines, handle, bodies)
        elif isinstance(block, ConditionalBlock):
            walk_branches(block, below, lines, handle, defs, bodies)
        elif isinstance(block, ControlFlowRegion):
            walk_regions(block, below, lines, handle, defs, bodies)


def walk_branches(block: ConditionalBlock, prefix: str, lines: List[str], handle: Optional[Handle],
                  defs: Dict[str, str], bodies: bool) -> None:
    """A conditional's branches. They are held in ``branches``, not as graph nodes, and the FIRST
    matching one wins -- so they are rendered in stored order, which is execution order."""
    for index, (condition, branch) in enumerate(block.branches):
        last = index == len(block.branches) - 1
        tag = "else" if condition is None else f"when {resolve_scalars(condition.as_string, defs)}"
        body = stamp(f"{branch.label}  {tag}", handle, "region", branch)
        lines.append(prefix + (ELBOW if last else TEE) + body)
        walk_regions(branch, prefix + (BLANK if last else PIPE), lines, handle, defs, bodies)


def walk_state(state: SDFGState, prefix: str, lines: List[str], handle: Optional[Handle], bodies: bool) -> None:
    """A state's kernels: every map nest, plus any library node (which is a kernel that never became a
    map). Nested scopes recurse, so an inner map is shown under the map that encloses it."""
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
            lines.append(pad + (ELBOW if last else TEE) + stamp(kernel_line(state, node), handle, "nest", node))
            if isinstance(node, nodes.MapEntry):
                if bodies:
                    lines.extend(below + BODY + line for line in kernel_body(state, state.sdfg, node, children))
                descend(node, below)

    descend(None, prefix)


def block_line(block: ControlFlowBlock, defs: Dict[str, str]) -> str:
    """One control-flow block's line: its canonical label, plus the domain or condition that says what
    it does. Nothing else -- a fact that holds for every block of a kind belongs in the skills, not on
    every row."""
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
    # No parallel/sequential column: a Map is data-parallel BY DEFINITION -- that is what makes it a
    # map rather than a loop -- so printing it on every line says the same thing every time, and a
    # single iteration is no exception. Where execution is genuinely forced sequential the construct
    # is a LoopRegion, which the tree already renders as `for` / `while`.
    return f"{node.map.label}  [{map_domain(node)}]{folds}  reads={reads} writes={writes}"
