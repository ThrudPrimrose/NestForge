# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Normalize an SDFG into the form the agent's text tree is projected from.

The tree (:mod:`nestforge.ir.introspect`) is the agent's whole view of the program, so every name it
prints has to be an identifier the agent can hand back, and every construct has to render one way
regardless of which frontend (Fortran, numpy, C, C++) produced it. Four properties give that, and
:func:`normalize_for_tree` establishes them in this order:

1. **No top-level nested SDFG.** A ``NestedSDFG`` outside every map scope is an opaque box in the
   middle of the tree: its states are real control flow the agent cannot see or fuse across. Widening
   its boundary memlets (``ExpandNestedSDFGInputs``) then inlining it (``InlineMultistateSDFG``)
   lifts that control flow into the tree. A nested SDFG INSIDE a map is left alone -- it is the body
   of a kernel, not structure.
2. **Canonical iteration domains.** Every map range and loop counter becomes ``0:trip:1`` -- dace's
   ``NormalizeLoopsAndMaps``. Steps are then positive and unit by construction, so the tree never has
   to render a descending or strided domain and the emitters never have to reason about step sign.
3. **Every computation sits inside a map.** A ``Tasklet`` outside all map scopes is enclosed in a
   single-iteration map, so "one map is one kernel" holds with no exceptions for the tree, the
   granularity lattice, or offloading to special-case. A ``LibraryNode`` is deliberately NOT wrapped:
   it already denotes a kernel, and burying it under a map scope would hide that.
4. **Canonical labels.** Every control-flow block and every map is renamed ``<kind><level>_<index>``,
   globally unique, so two runs over the same input produce the same names and a tree line names
   something the agent can refer back to. Maps are ``kernel`` -- one map is one kernel, including the
   wrap maps step 3 just added. Transient DATA and map parameters are renamed here too -- ``t<n>``
   for arrays, ``s<n>`` for scalars, ``i<n>`` for parameters. The frontend qualifies a transient with
   its whole module path (``npbench_..._build_up_b___tmp1``), which was most of the width of a tree
   line and carried nothing the agent can use.

Steps 1-3 act on this SDFG and its control-flow regions only; a ``NestedSDFG`` that survives step 1
is a kernel body, and the agent fuses kernels, not their insides. Step 4 is the exception and
descends everywhere, because a nameless map is a hole in the vocabulary wherever it sits.
"""
from __future__ import annotations

import copy
import heapq
import re
from typing import Dict, List, Optional, Tuple, Union

import dace
from dace import data as dt
from dace.sdfg import nodes
from dace.sdfg.replace import replace_dict
from dace.sdfg.state import (BreakBlock, ConditionalBlock, ContinueBlock, ControlFlowBlock, ControlFlowRegion,
                             LoopRegion, ReturnBlock, SDFGState)
from dace.transformation.interstate.expand_nested_sdfg_inputs import ExpandNestedSDFGInputs
from dace.transformation.interstate.multistate_inline import InlineMultistateSDFG
from dace.transformation.passes.canonicalize.normalize_loops_and_maps import NormalizeLoopsAndMaps
from dace.transformation.passes.normalize_wcr import NormalizeWCR
from dace.transformation.passes.normalize_wcr_source import NormalizeWCRSource

#: Iteration variable of a wrap map. One name for every wrap map in the tree: they are all ``0:1``, so
#: the variable is never read, and a shared name keeps the rendered domain identical everywhere.
WRAP_PARAM = "__nf_wrap"

#: A transient name that is already canonical: ``t<n>`` for an array, ``s<n>`` for a scalar.
CANONICAL_DATA = re.compile(r"[ts]\d+")


def in_order(graph: Union[ControlFlowRegion, SDFGState]) -> List:
    """A graph's nodes in topological order, ties broken by insertion order -- used for both a CFG
    (blocks) and a state (dataflow nodes).

    Kahn with the ready set kept in insertion order, rather than ``sdutil.dfs_topological_sort``,
    because the tie-break is the point: a topological order alone is not unique, and two orders give
    two label assignments for one program. That sort is deterministic too (it follows edge insertion
    order) but it is depth-first, so at a branch it numbers one arm to the bottom before starting the
    other -- with structured control flow, where a CFG is usually a chain, the two agree; where they
    differ, the earliest-ready node is the one the source put first. Nodes left over (a cycle has no
    source) keep insertion order too -- an unreachable or cyclic block still needs a name.
    """
    all_nodes = list(graph.nodes())
    if not all_nodes:
        return []
    rank = {id(n): i for i, n in enumerate(all_nodes)}
    indegree = {id(n): 0 for n in all_nodes}
    for edge in graph.edges():
        indegree[id(edge.dst)] += 1
    ready = [rank[id(n)] for n in all_nodes if indegree[id(n)] == 0]
    heapq.heapify(ready)
    ordered: List = []
    while ready:
        node = all_nodes[heapq.heappop(ready)]
        ordered.append(node)
        for edge in graph.out_edges(node):
            indegree[id(edge.dst)] -= 1
            if indegree[id(edge.dst)] == 0:
                heapq.heappush(ready, rank[id(edge.dst)])
    seen = {id(n) for n in ordered}
    return ordered + [n for n in all_nodes if id(n) not in seen]


# --- 1. no top-level nested SDFG -------------------------------------------------------------------


def top_level_nsdfgs(sdfg: dace.SDFG) -> List[Tuple[SDFGState, nodes.NestedSDFG]]:
    """Every ``NestedSDFG`` that sits outside all map scopes. One inside a map is a kernel body and is
    left alone."""
    out: List[Tuple[SDFGState, nodes.NestedSDFG]] = []
    for state in sdfg.all_states():
        sd = state.scope_dict()  # once per state: state.entry_node() rebuilds this per call
        out += [(state, node) for node in state.nodes() if isinstance(node, nodes.NestedSDFG) and sd[node] is None]
    return out


def inline_top_level_nsdfgs(sdfg: dace.SDFG) -> int:
    """Widen and inline every top-level nested SDFG, returning how many transformations that took.

    Returns 0 without touching anything when there is no top-level nested SDFG -- the common case once
    the agent has normalized once, and worth a cheap scan: both transformations otherwise pattern-match
    across every state before concluding the same thing.
    """
    if not top_level_nsdfgs(sdfg):
        return 0
    applied = sdfg.apply_transformations_repeated(ExpandNestedSDFGInputs,
                                                  options={"top_level_only": True},
                                                  validate=False)
    return applied + sdfg.apply_transformations_repeated(InlineMultistateSDFG, validate=False)


# --- 3. every computation inside a map -------------------------------------------------------------


def free_tasklets(state: SDFGState) -> List[nodes.Tasklet]:
    """Tasklets in ``state`` that sit outside every map scope. A ``LibraryNode`` is not a ``Tasklet``
    and so is never here -- by design: it already is a kernel."""
    sd = state.scope_dict()  # once per state: state.entry_node() rebuilds this per call
    return [n for n in state.nodes() if isinstance(n, nodes.Tasklet) and sd[n] is None]


def wrap_groups(state: SDFGState) -> List[List[nodes.Tasklet]]:
    """The free tasklets of ``state``, partitioned into the FEWEST groups each of which can become one
    map.

    Two free tasklets may share a map only if neither can reach the other: contracting a reachable
    pair into a single scope closes a cycle through whatever sits between them. So a group must be an
    antichain of the reachability order. Levelling each tasklet by the longest chain of free tasklets
    ending at it produces antichains, and by Mirsky's theorem the number of levels equals the longest
    chain -- which is a lower bound on any such partition, so this is minimal.

    The level is computed by one forward pass in topological order: every node carries the length of
    the longest free-tasklet chain reaching it, and a free tasklet extends it by one. Paths through
    map scopes and library nodes are counted like any other, which is what keeps a tasklet before a
    map and a tasklet after it in different groups.
    """
    free = {id(t) for t in free_tasklets(state)}
    if not free:
        return []
    depth: Dict[int, int] = {}
    groups: Dict[int, List[nodes.Tasklet]] = {}
    for node in in_order(state):
        reaching = max((depth[id(e.src)] for e in state.in_edges(node) if id(e.src) in depth), default=-1)
        if id(node) in free:
            depth[id(node)] = reaching + 1
            groups.setdefault(reaching + 1, []).append(node)
        else:
            depth[id(node)] = reaching
    return [groups[level] for level in sorted(groups)]


def wrap_group(state: SDFGState, group: List[nodes.Tasklet], name: str) -> None:
    """Enclose ``group`` in one single-iteration map.

    The map is scheduled ``Sequential``, which is a CODEGEN choice and not a claim about the
    computation: a map is data-parallel by definition, one iteration included. The schedule only stops
    codegen opening an OpenMP region around a single iteration. Nothing in the tree reports it.
    """
    entry, exit_node = state.add_map(name, {WRAP_PARAM: "0:1"}, schedule=dace.ScheduleType.Sequential)
    for tasklet in group:
        in_edges = list(state.in_edges(tasklet))
        out_edges = list(state.out_edges(tasklet))
        for edge in in_edges:
            state.remove_edge(edge)
            state.add_memlet_path(edge.src,
                                  entry,
                                  tasklet,
                                  memlet=copy.deepcopy(edge.data),
                                  src_conn=edge.src_conn,
                                  dst_conn=edge.dst_conn)
        for edge in out_edges:
            state.remove_edge(edge)
            state.add_memlet_path(tasklet,
                                  exit_node,
                                  edge.dst,
                                  memlet=copy.deepcopy(edge.data),
                                  src_conn=edge.src_conn,
                                  dst_conn=edge.dst_conn)
        # A tasklet with no data on one side still has to be held in the scope, or it floats out of the
        # map and the wrap achieves nothing.
        if not in_edges:
            state.add_nedge(entry, tasklet, dace.Memlet())
        if not out_edges:
            state.add_nedge(tasklet, exit_node, dace.Memlet())


def wrap_free_tasklets(sdfg: dace.SDFG) -> int:
    """Wrap every free tasklet in the SDFG, returning how many maps that took (0 when there were none,
    in which case nothing is touched). The names given here are placeholders --
    :func:`normalize_labels` renumbers every map, these included."""
    added = 0
    for state in sdfg.all_states():
        for group in wrap_groups(state):
            wrap_group(state, group, f"wrap_{added}")
            added += 1
    return added


# --- 4. canonical labels ---------------------------------------------------------------------------


def block_kind(block: ControlFlowBlock) -> str:
    """The tree keyword for a control-flow block -- the ``<kind>`` half of its canonical label.

    A ``LoopRegion`` splits by shape rather than by class: one carrying both an init and an update
    statement is a counted ``for``, anything else is a ``while``. That distinction is what the agent
    reads to know whether a trip count exists, and it is invisible in the class name.
    """
    if isinstance(block, LoopRegion):
        return "for" if block.init_statement is not None and block.update_statement is not None else "while"
    if isinstance(block, ConditionalBlock):
        return "if"
    if isinstance(block, ContinueBlock):
        return "continue"
    if isinstance(block, BreakBlock):
        return "break"
    if isinstance(block, ReturnBlock):
        return "return"
    if isinstance(block, SDFGState):
        return "state"
    return "block"


def normalize_labels(sdfg: dace.SDFG) -> None:
    """Rename every control-flow block and every map to ``<kind><level>_<index>``, globally unique.

    ``level`` is nesting depth: the root SDFG's own blocks are level 0, a state's top-level maps are
    one deeper than the state, and a map inside a map is deeper again -- so the level matches the
    indentation the tree prints the thing at.

    ``index`` runs per ``(kind, level)`` across the WHOLE SDFG, so five kernels at depth 3 are
    ``kernel3_0 .. kernel3_4`` and the numbering of one kind never depends on how many of another
    happen to sit beside it. Counting per level rather than per CFG is what makes the names unique:
    two sibling loops each hold a first state, and numbering within the CFG would call both
    ``state1_0``.
    """
    relabel_cfg(sdfg, 0, {})


def next_label(kind: str, level: int, counters: Dict[tuple, int]) -> str:
    """The next free ``<kind><level>_<index>``, advancing that kind's counter at that level."""
    index = counters.get((kind, level), 0)
    counters[(kind, level)] = index + 1
    return f"{kind}{level}_{index}"


def relabel_cfg(cfg: Union[dace.SDFG, ControlFlowRegion], level: int, counters: Dict[tuple, int]) -> None:
    """Relabel one CFG's blocks at ``level``, recursing into the regions and states among them.
    ``counters`` is the per-(kind, level) running index, shared across the whole traversal."""
    for block in in_order(cfg):
        block.label = next_label(block_kind(block), level, counters)
        if isinstance(block, SDFGState):
            relabel_state(block, level + 1, counters)
        elif isinstance(block, ConditionalBlock):
            # Branches live in ``_branches``, not in the graph, so the loop above never reaches them.
            for _, branch in block.branches:
                branch.label = next_label("block", level + 1, counters)
                relabel_cfg(branch, level + 2, counters)
        elif isinstance(block, ControlFlowRegion):
            relabel_cfg(block, level + 1, counters)


def rename_transient_data(sdfg: dace.SDFG) -> Dict[str, str]:
    """Rename transient data to ``t<n>`` (arrays) and ``s<n>`` (scalars), returning the mapping. Returns
    ``{}`` and touches nothing when everything transient is already canonical.

    Non-transients keep their names: those are the program's interface, and the boundary, the manifest
    and the emitted numpy signature all name them. So ``dt``, ``dx``, ``rho`` stay readable while
    ``npbench_benchmarks_cavity_flow_cavity_flow_dace_build_up_b___tmp0`` -- the frontend qualifying an
    internal temporary with its whole module path -- becomes ``s4``.

    **A name that is already canonical KEEPS its index.** Renumbering densely from zero would be
    simpler and is wrong twice over: dropping one transient shifts every later name, so a single
    fusion move renamed 21 arrays on cavity_flow (106ms of `replace_dict` per move), and -- worse --
    the id the agent read off the tree would silently point at a different array after any move. The
    tree is only a vocabulary if its words hold still.
    """
    targets = {n: ("s" if isinstance(desc, dt.Scalar) else "t") for n, desc in sdfg.arrays.items() if desc.transient}
    settled = {n for n, prefix in targets.items() if CANONICAL_DATA.fullmatch(n) and n[0] == prefix}
    taken = {prefix: {int(n[1:]) for n in settled if n[0] == prefix} for prefix in ("t", "s")}
    # A survivor already called ``t3`` would otherwise be clobbered by whatever is renamed to ``t3``.
    survivors = {n for n in sdfg.arrays if n not in targets} | set(sdfg.symbols)
    # ...and so would a target that is canonical-SHAPED but under the wrong prefix: a Scalar named ``t0``
    # is not settled (its prefix is ``s``), yet it still HOLDS the name ``t0`` until its own rename lands.
    # replace_dict renames the descriptor dict sequentially, so handing ``t0`` to someone else first
    # clobbered the holder and left the SDFG failing validation with `Array "t0" not found`.
    held = {n for n in targets if n not in settled}
    renames = {}
    for old, prefix in targets.items():
        if old in settled:
            continue
        index = 0
        while index in taken[prefix] or f"{prefix}{index}" in (survivors | held):
            index += 1
        taken[prefix].add(index)
        renames[old] = f"{prefix}{index}"
    if not renames:
        return {}
    # ONE replace_dict, not a rename per name -- but note it is simultaneous only for the GRAPH: the
    # descriptor dict is rewritten key by key (dace/sdfg/sdfg.py), so a mapping that hands out a name
    # another target still holds destroys that target's descriptor. ``held`` above is what keeps such a
    # mapping from being built in the first place.
    sdfg.replace_dict(renames)
    return renames


def enclosing_param_count(node: nodes.MapEntry, scope: Dict) -> int:
    """How many map parameters the ENCLOSING map chain of ``node`` already owns.

    A nested map's body reads its ancestors' parameters too, so its own names have to start where theirs
    stop -- numbering every map from ``i0`` inside its own scope gives an inner 1-D map the name its outer
    map already holds."""
    count, parent = 0, scope[node]
    while parent is not None:
        count += len(parent.map.params)
        parent = scope[parent]
    return count


def rename_map_params(sdfg: dace.SDFG) -> None:
    """Rename map parameters to ``i0, i1, ...``, numbered down the NESTING CHAIN rather than per scope.
    The frontend's ``__i0`` is the same name with leading underscores; a wrap map keeps
    :data:`WRAP_PARAM`, which says what it is.

    Numbered by depth because a map's scope subgraph contains its ancestors' parameters: an outer ``i``
    and an inner ``j`` both renamed to ``i0`` make the body read ``A[i0, i0]``, so the nest writes only its
    diagonal. It validates clean and returns a silent wrong answer (measured: ``B[0] == [1, 0, 0, 0]`` for
    ``B = A + 1``). Sibling maps are disjoint scopes and may reuse names freely.

    Two passes through a fresh temporary, because one pass cannot be safe in either direction: renaming an
    outer ``i`` to ``i0`` collides with an inner map ALREADY called ``i0``, and renaming the inner first
    collides with whatever the outer becomes. The temporaries are unique across the whole state, so
    neither pass can land on a live name."""
    for state in sdfg.all_states():
        scope = state.scope_dict()
        entries = [n for n in state.nodes() if isinstance(n, nodes.MapEntry) and WRAP_PARAM not in n.map.params]
        targets = {}
        for node in entries:
            base = enclosing_param_count(node, scope)
            wanted = [f"i{base + axis}" for axis in range(len(node.map.params))]
            if node.map.params != wanted:
                targets[node] = wanted
        if not targets:
            continue
        # pass 1: every renamed param to a name nothing in this state can be holding
        temps = {}
        for index, (node, wanted) in enumerate(targets.items()):
            temp = [f"__nf_param{index}_{axis}" for axis in range(len(wanted))]
            # ONE simultaneous substitution per scope, the same discipline as rename_transient_data: a
            # rename per pair lets the second overwrite the first when a later param holds an earlier
            # param's target.
            replace_dict(state.scope_subgraph(node), dict(zip(node.map.params, temp)))
            node.map.params = temp
            temps[node] = temp
        # pass 2: the temporaries to the final names, now that no live param can collide with one
        for node, wanted in targets.items():
            replace_dict(state.scope_subgraph(node), dict(zip(temps[node], wanted)))
            node.map.params = wanted


def relabel_state(state: SDFGState, level: int, counters: Dict[tuple, int]) -> None:
    """Name every map in ``state`` ``kernel<level>_<index>``, outermost first and one level deeper per
    enclosing map, and descend through any ``NestedSDFG`` into its blocks.

    The descent is what stops a map inside a kernel body from keeping a frontend name -- dace-fortran
    and the Python frontend both label inner maps after the SOURCE LINE they came from (``inner_9_4``),
    so an edit above them renames a kernel that did not change. Naming is cheap and harmless at any
    depth; wrapping (:func:`wrap_free_tasklets`) deliberately is not applied down here, where a trivial
    map around every statement would bloat the body the agent never fuses inside anyway.

    Setting ``Map.label`` renames the entry and the exit together -- they share the ``Map`` object,
    which is the thing being named.
    """
    children = state.scope_children()
    rank = {id(n): i for i, n in enumerate(in_order(state))}

    def descend(scope: Optional[nodes.MapEntry], depth: int) -> None:
        for node in sorted(children[scope], key=lambda n: rank.get(id(n), 0)):
            if isinstance(node, nodes.MapEntry):
                node.map.label = next_label("kernel", depth, counters)
                descend(node, depth + 1)
            elif isinstance(node, nodes.NestedSDFG):
                relabel_cfg(node.sdfg, depth, counters)

    descend(None, level)


def normalize_reductions(sdfg: dace.SDFG) -> None:
    """Put every reduction in one shape: the accumulation on a body-local transient, and the
    cross-iteration fold as a WCR on an ``AccessNode -> MapExit`` edge.

    That is what makes a reduction *recognizable* rather than merely present. Left alone, the frontend
    can put a masked reduction's WCR inside a nested SDFG, or source one from a tasklet, and there is
    then no single edge to ask what is being reduced over which axes -- so the tree cannot show it and
    the agent has to read the body to find out a kernel folds.
    """
    NormalizeWCR().apply_pass(sdfg, {})
    NormalizeWCRSource().apply_pass(sdfg, {})


# --- the pipeline ----------------------------------------------------------------------------------


def normalize_for_tree(sdfg: dace.SDFG) -> None:
    """Put ``sdfg`` in the tree's normal form, in place: no top-level nested SDFG, canonical domains,
    every computation in a map, canonical labels. Idempotent -- running it twice is running it once,
    which matters because the agent re-normalizes after each fusion move."""
    inline_top_level_nsdfgs(sdfg)
    normalize_reductions(sdfg)
    NormalizeLoopsAndMaps().apply_pass(sdfg, {})
    wrap_free_tasklets(sdfg)
    # Names LAST: a wrap map is a kernel like any other and has to be numbered with the rest, and the
    # transients the inline lifted out of a nested SDFG have to be numbered with the rest too.
    rename_transient_data(sdfg)
    rename_map_params(sdfg)
    normalize_labels(sdfg)
