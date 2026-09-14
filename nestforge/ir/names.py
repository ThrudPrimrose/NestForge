# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Normalize an SDFG into the canonical form the agent's text tree is projected from: no top-level
nested SDFG, ``0:trip:1`` iteration domains, every computation inside a map, and canonical
``<kind><level>_<index>`` labels for every block, map, transient, and map parameter."""

from __future__ import annotations

import copy
import heapq
import re
from typing import Dict, List, Optional, Tuple, Union

import dace
from dace import data as dt
from dace.sdfg import nodes
from dace.sdfg.replace import replace_dict
from dace.sdfg.state import (
    BreakBlock,
    ConditionalBlock,
    ContinueBlock,
    ControlFlowBlock,
    ControlFlowRegion,
    LoopRegion,
    ReturnBlock,
    SDFGState,
)
from dace.transformation.interstate.expand_nested_sdfg_inputs import ExpandNestedSDFGInputs
from dace.transformation.interstate.multistate_inline import InlineMultistateSDFG
from dace.transformation.passes.canonicalize.normalize_loops_and_maps import NormalizeLoopsAndMaps
from dace.transformation.passes.normalize_wcr import NormalizeWCR
from dace.transformation.passes.normalize_wcr_source import NormalizeWCRSource
from dace.utils import find_new_name

#: Iteration variable of a wrap map; shared by every wrap map since none of them ever read it.
WRAP_PARAM = "__nf_wrap"

#: A transient name that is already canonical: ``t<n>`` for an array, ``s<n>`` for a scalar.
CANONICAL_DATA = re.compile(r"[ts]\d+")


def in_order(graph: Union[ControlFlowRegion, SDFGState]) -> List:
    """A graph's nodes in topological order, ties broken by insertion order (Kahn, so two runs over
    the same program give the same label assignment)."""
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


# 1. no top-level nested SDFG


def top_level_nsdfgs(sdfg: dace.SDFG) -> List[Tuple[SDFGState, nodes.NestedSDFG]]:
    """Every ``NestedSDFG`` that sits outside all map scopes. One inside a map is a kernel body and is
    left alone."""
    out: List[Tuple[SDFGState, nodes.NestedSDFG]] = []
    for state in sdfg.all_states():
        sd = state.scope_dict()  # once per state: state.entry_node() rebuilds this per call
        out += [(state, node) for node in state.nodes() if isinstance(node, nodes.NestedSDFG) and sd[node] is None]
    return out


def inline_top_level_nsdfgs(sdfg: dace.SDFG) -> int:
    """Widen and inline every top-level nested SDFG, returning how many transformations that took;
    0 without touching anything when there is none (a cheap scan avoids two pattern-match sweeps)."""
    if not top_level_nsdfgs(sdfg):
        return 0
    applied = sdfg.apply_transformations_repeated(
        ExpandNestedSDFGInputs, options={"top_level_only": True}, validate=False
    )
    return applied + sdfg.apply_transformations_repeated(InlineMultistateSDFG, validate=False)


# 3. every computation inside a map


def free_tasklets(state: SDFGState) -> List[nodes.Tasklet]:
    """Tasklets in ``state`` that sit outside every map scope. A ``LibraryNode`` is not a ``Tasklet``
    and so is never here -- by design: it already is a kernel."""
    sd = state.scope_dict()  # once per state: state.entry_node() rebuilds this per call
    return [n for n in state.nodes() if isinstance(n, nodes.Tasklet) and sd[n] is None]


def wrap_groups(state: SDFGState) -> List[List[nodes.Tasklet]]:
    """The free tasklets of ``state``, partitioned into the FEWEST groups each of which can become one
    map: two tasklets share a group only if neither can reach the other (an antichain of the
    reachability order), found by levelling each tasklet by its longest free-tasklet chain depth."""
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
    """Enclose ``group`` in one single-iteration map."""
    # Sequential is a codegen choice only (a map is data-parallel by definition either way).
    entry, exit_node = state.add_map(name, {WRAP_PARAM: "0:1"}, schedule=dace.ScheduleType.Sequential)
    for tasklet in group:
        in_edges = list(state.in_edges(tasklet))
        out_edges = list(state.out_edges(tasklet))
        for edge in in_edges:
            state.remove_edge(edge)
            state.add_memlet_path(
                edge.src,
                entry,
                tasklet,
                memlet=copy.deepcopy(edge.data),
                src_conn=edge.src_conn,
                dst_conn=edge.dst_conn,
            )
        for edge in out_edges:
            state.remove_edge(edge)
            state.add_memlet_path(
                tasklet,
                exit_node,
                edge.dst,
                memlet=copy.deepcopy(edge.data),
                src_conn=edge.src_conn,
                dst_conn=edge.dst_conn,
            )
        # a tasklet with no data on one side still needs holding, or it floats out of the map
        if not in_edges:
            state.add_nedge(entry, tasklet, dace.Memlet())
        if not out_edges:
            state.add_nedge(tasklet, exit_node, dace.Memlet())


def wrap_free_tasklets(sdfg: dace.SDFG) -> int:
    """Wrap every free tasklet in the SDFG, returning how many maps that took (0 and untouched when
    there were none); names given here are placeholders, renumbered later with every other map."""
    added = 0
    for state in sdfg.all_states():
        for group in wrap_groups(state):
            wrap_group(state, group, f"wrap_{added}")
            added += 1
    return added


# 4. canonical labels


def block_kind(block: ControlFlowBlock) -> str:
    """The tree keyword for a control-flow block. A ``LoopRegion`` splits by shape, not class: one
    carrying both an init and an update statement is a counted ``for``, anything else a ``while``."""
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
    """Rename every control-flow block and every map to ``<kind><level>_<index>``, globally unique:
    ``index`` counts per ``(kind, level)`` across the whole SDFG, not per CFG. A library node keeps its
    label unless an earlier one already holds it."""
    relabel_cfg(sdfg, 0, {})
    unique_library_labels(sdfg)


def unique_library_labels(sdfg: dace.SDFG) -> None:
    """Rename each library node whose label an earlier one holds, nested SDFGs included."""
    taken: Dict[str, None] = {}
    for node, _ in sdfg.all_nodes_recursive():
        if not isinstance(node, nodes.LibraryNode):
            continue
        if node.label in taken:
            # LibraryNode sets name and label alike; keep them equal
            node.name = node.label = find_new_name(node.label, taken)
        taken[node.label] = None


def next_label(kind: str, level: int, counters: Dict[tuple, int]) -> str:
    """The next free ``<kind><level>_<index>``, advancing that kind's counter at that level."""
    index = counters.get((kind, level), 0)
    counters[(kind, level)] = index + 1
    return f"{kind}{level}_{index}"


def relabel_cfg(cfg: Union[dace.SDFG, ControlFlowRegion], level: int, counters: Dict[tuple, int]) -> None:
    """Relabel one CFG's blocks at ``level``, recursing into the regions and states among them."""
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
    """Rename transient data to ``t<n>``/``s<n>``, returning the mapping (``{}`` if already canonical).
    An already-canonical name KEEPS its index: renumbering would leave a tree id the agent already
    holds pointing at a different array."""
    targets = {n: ("s" if isinstance(desc, dt.Scalar) else "t") for n, desc in sdfg.arrays.items() if desc.transient}
    settled = {n for n, prefix in targets.items() if CANONICAL_DATA.fullmatch(n) and n[0] == prefix}
    taken = {prefix: {int(n[1:]) for n in settled if n[0] == prefix} for prefix in ("t", "s")}
    survivors = {n for n in sdfg.arrays if n not in targets} | set(sdfg.symbols)
    # a mis-prefixed canonical-shaped name (Scalar "t0") still HOLDS "t0" until its own rename lands
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
    sdfg.replace_dict(renames)
    return renames


def enclosing_param_count(node: nodes.MapEntry, scope: Dict) -> int:
    """How many map parameters the ENCLOSING map chain of ``node`` already owns."""
    count, parent = 0, scope[node]
    while parent is not None:
        count += len(parent.map.params)
        parent = scope[parent]
    return count


def rename_map_params(sdfg: dace.SDFG) -> None:
    """Rename map parameters to ``i0, i1, ...`` down the NESTING CHAIN, not per scope -- reusing an
    ancestor's name would alias reads and silently compute the wrong answer. Renamed via a fresh
    temporary in two passes, since renaming outer-then-inner (or the reverse) collides with the other."""
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
            # one simultaneous substitution per scope: a rename per pair could overwrite an earlier target
            replace_dict(state.scope_subgraph(node), dict(zip(node.map.params, temp)))
            node.map.params = temp
            temps[node] = temp
        # pass 2: the temporaries to the final names, now that no live param can collide with one
        for node, wanted in targets.items():
            replace_dict(state.scope_subgraph(node), dict(zip(temps[node], wanted)))
            node.map.params = wanted


def relabel_state(state: SDFGState, level: int, counters: Dict[tuple, int]) -> None:
    """Name every map in ``state`` ``kernel<level>_<index>``, outermost first and one level deeper per
    enclosing map, descending through any ``NestedSDFG`` so an inner map never keeps a frontend
    source-line name (``inner_9_4``) that would rename a kernel that did not change."""
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
    """Put every reduction in one shape: accumulation on a body-local transient, cross-iteration fold
    as a WCR on an ``AccessNode -> MapExit`` edge -- so there is one edge the tree can always ask."""
    NormalizeWCR().apply_pass(sdfg, {})
    NormalizeWCRSource().apply_pass(sdfg, {})


# the pipeline


def normalize_for_tree(sdfg: dace.SDFG) -> None:
    """Put ``sdfg`` in the tree's normal form, in place; idempotent, since the agent re-normalizes
    after each fusion move."""
    inline_top_level_nsdfgs(sdfg)
    normalize_reductions(sdfg)
    NormalizeLoopsAndMaps().apply_pass(sdfg, {})
    wrap_free_tasklets(sdfg)
    # names LAST: a wrap map and any transient the inline lifted out still need numbering with the rest
    rename_transient_data(sdfg)
    rename_map_params(sdfg)
    normalize_labels(sdfg)
