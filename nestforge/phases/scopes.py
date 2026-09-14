# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from typing import Callable, Dict, List, Tuple, Type, Union
import dace
from dace.sdfg import nodes
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion
from nestforge.ir.extract import NestNode
import copy
from typing import List, Tuple, Union
from nestforge.ir.emit_numpy import nest_to_numpy
from nestforge.ir.emit_yaml import manifest_dict
from nestforge.ir.extract import Boundary, extract_nest_to_sdfg
from nestforge.ir.libnode import ExternalCall, in_conn, out_conn
from dataclasses import dataclass
from dace.sdfg.state import ConditionalBlock, LoopRegion
from nestforge.ir.extract import NestNode, extract_nest_to_sdfg, whole_program_boundary

Strategy = Callable[[dace.SDFG], List[Tuple[dace.SDFG, NestNode]]]

REGISTRY: Dict[str, Strategy] = {}


def register_strategy(name: str, fn: Strategy) -> None:
    REGISTRY[name] = fn


def get_strategy(name: str) -> Strategy:
    if name not in REGISTRY:
        raise KeyError(f"unknown strategy {name!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[name]


def strategy_names() -> List[str]:
    return sorted(REGISTRY)


def top_level_map_entries(state: dace.SDFGState) -> List[nodes.MapEntry]:
    """MapEntry nodes at the top of a state's scope tree (not nested inside another map)."""
    return [n for n in state.scope_children()[None] if isinstance(n, nodes.MapEntry)]


def branch_states(region: ConditionalBlock) -> List[dace.SDFGState]:
    """Direct states of a conditional's branches, including nested conditionals but not nested loops."""
    states: List[dace.SDFGState] = []
    for _, branch in region.branches:
        for block in branch.nodes():
            if isinstance(block, dace.SDFGState):
                states.append(block)
            elif isinstance(block, ConditionalBlock):
                states.extend(branch_states(block))
    return states


def outer(sdfg: dace.SDFG) -> List[Tuple[dace.SDFG, NestNode]]:
    """Outermost nests of the root SDFG: top-level map-nests, loop regions, and conditional branches' maps."""
    refs: List[Tuple[dace.SDFG, NestNode]] = []
    for block in sdfg.nodes():
        if isinstance(block, LoopRegion):
            refs.append((sdfg, block))
        elif isinstance(block, dace.SDFGState):
            for me in top_level_map_entries(block):
                refs.append((sdfg, me))
        elif isinstance(block, ConditionalBlock):
            for state in branch_states(block):
                for me in top_level_map_entries(state):
                    refs.append((sdfg, me))
    return refs


_COMPUTE = (nodes.Tasklet, nodes.LibraryNode, nodes.NestedSDFG)


def direct_child_maps(state: dace.SDFGState, entry: nodes.MapEntry) -> List[nodes.MapEntry]:
    return [n for n in state.scope_children()[entry] if isinstance(n, nodes.MapEntry)]


def is_taskloop_map(state: dace.SDFGState, entry: nodes.MapEntry) -> bool:
    """A map whose body is *only* maps (no tasklet/library/nested compute) -- a scheduling wrapper."""
    kids = state.scope_children()[entry]
    has_map = any(isinstance(n, nodes.MapEntry) for n in kids)
    has_compute = any(isinstance(n, _COMPUTE) for n in kids)
    return has_map and not has_compute


def is_parallel_nest(node: NestNode) -> bool:
    """Whether an extracted nest is PARALLEL (a non-Sequential Map) or SEQUENTIAL (a LoopRegion)."""
    if isinstance(node, nodes.MapEntry):
        return node.map.schedule != dace.ScheduleType.Sequential
    return False  # LoopRegion (or anything non-Map): sequential


def is_taskloop_loop(loop: LoopRegion) -> bool:
    """A loop whose body is *only* maps: states with no free compute and no nested control flow."""
    if any(not isinstance(b, dace.SDFGState) for b in loop.nodes()):
        return False
    has_map = False
    for state in loop.nodes():
        for n in state.scope_children()[None]:
            if isinstance(n, _COMPUTE):
                return False
            if isinstance(n, nodes.MapEntry):
                has_map = True
    return has_map


def collect_skip_map(sdfg: dace.SDFG, state: dace.SDFGState, entry: nodes.MapEntry, refs: list) -> None:
    if is_taskloop_map(state, entry):
        for child in direct_child_maps(state, entry):
            collect_skip_map(sdfg, state, child, refs)
    else:
        refs.append((sdfg, entry))


def collect_skip_loop(sdfg: dace.SDFG, loop: LoopRegion, refs: list) -> None:
    if is_taskloop_loop(loop):
        for state in loop.nodes():
            for me in top_level_map_entries(state):
                collect_skip_map(sdfg, state, me, refs)
    else:
        refs.append((sdfg, loop))


def skip_taskloops(sdfg: dace.SDFG) -> List[Tuple[dace.SDFG, NestNode]]:
    """Skips pure taskloop wrappers (map/loop bodies holding only maps) and descends to the first compute nest."""
    refs: List[Tuple[dace.SDFG, NestNode]] = []
    for block in sdfg.nodes():
        if isinstance(block, LoopRegion):
            collect_skip_loop(sdfg, block, refs)
        elif isinstance(block, dace.SDFGState):
            for me in top_level_map_entries(block):
                collect_skip_map(sdfg, block, me, refs)
        elif isinstance(block, ConditionalBlock):
            for state in branch_states(block):
                for me in top_level_map_entries(state):
                    collect_skip_map(sdfg, state, me, refs)
    return refs


def region_has(region: Union[dace.SDFG, ControlFlowRegion], node_types: Tuple[Type, ...]) -> bool:
    """True if any state anywhere in a control-flow region holds a node of the given types."""
    return any(
        isinstance(n, node_types) for block in region.all_control_flow_blocks() if isinstance(block, dace.SDFGState)
        for n in block.nodes())


def innermost(sdfg: dace.SDFG) -> List[Tuple[dace.SDFG, NestNode]]:
    """Every innermost compute leaf (a map/loop with no map, loop, or NestedSDFG nested inside it), each once."""
    refs: List[Tuple[dace.SDFG, NestNode]] = []
    for sub in sdfg.all_sdfgs_recursive():
        for state in sub.states():
            for entry in [n for n in state.nodes() if isinstance(n, nodes.MapEntry)]:
                deeper = [
                    n for n in state.scope_subgraph(entry, include_entry=False).nodes()
                    if isinstance(n, (nodes.MapEntry, nodes.NestedSDFG))
                ]
                if not deeper:
                    refs.append((sub, entry))
        for region in sub.all_control_flow_regions():
            if not isinstance(region, LoopRegion):
                continue
            nested_loops = [
                r for r in region.all_control_flow_regions() if isinstance(r, LoopRegion) and r is not region
            ]
            if not nested_loops and not region_has(region, (nodes.MapEntry, nodes.NestedSDFG)):
                refs.append((sub, region))
    return refs


def empty_strategy_reason(sdfg: dace.SDFG) -> str:
    """Why a strategy found nothing: an honest empty kernel, or one whose only compute is a library node."""
    has_libnode = any(
        isinstance(n, nodes.LibraryNode) for sub in sdfg.all_sdfgs_recursive() for st in sub.states()
        for n in st.nodes())
    if has_libnode:
        return "only library-node compute (DaCe offloads it to its fastest library; no loop-nest to externalise)"
    return "no compute nest (strategy returned nothing)"


register_strategy("outer", outer)
register_strategy("skip-taskloops", skip_taskloops)
register_strategy("innermost", innermost)


def reference_sdfg(boundary: Boundary) -> "dace.SDFG":
    """Copy of the standalone SDFG with boundary arrays renamed to the node's connectors; an in-place
    array gets both an ``_in_`` and an ``_out_`` connector since one array carries two connectors."""
    ref = copy.deepcopy(boundary.standalone_sdfg)
    inplace = set(boundary.inputs) & set(boundary.outputs)
    for i in boundary.inputs:
        if i not in inplace:
            ref.replace(i, in_conn(i))
    for o in boundary.outputs:
        ref.replace(o, out_conn(o))
    for name in sorted(inplace):
        ref.add_datadesc(in_conn(name), copy.deepcopy(ref.arrays[out_conn(name)]))
    return ref


def replace_nsdfg_with_external(boundary: Boundary, name: str) -> ExternalCall:
    state = boundary.state
    nsdfg = boundary.nsdfg_node
    # Connectors are prefixed so they never collide with array/symbol names (a LibraryNode rule).
    ext = ExternalCall(name,
                       inputs={in_conn(i)
                               for i in boundary.inputs},
                       outputs={out_conn(o)
                                for o in boundary.outputs},
                       numpy_source=nest_to_numpy(boundary, fn_name=name),
                       config=manifest_dict(boundary, name),
                       standalone_sdfg=reference_sdfg(boundary))
    state.add_node(ext)
    # Fresh memlets per edge (never reuse subsets/memlets); remap connector names.
    for e in state.in_edges(nsdfg):
        state.add_edge(e.src, e.src_conn, ext, in_conn(e.dst_conn), copy.deepcopy(e.data))
    for e in state.out_edges(nsdfg):
        state.add_edge(ext, out_conn(e.src_conn), e.dst, e.dst_conn, copy.deepcopy(e.data))
    state.remove_node(nsdfg)
    return ext


def lower_nests_to_external_call(sdfg: dace.SDFG,
                                 strategy: Union[str,
                                                 Strategy] = "skip-taskloops") -> List[Tuple[ExternalCall, Boundary]]:
    """Lowers every nest ``strategy`` selects (default ``skip-taskloops``) into an ``ExternalCall`` node,
    returning ``[(call, boundary), ...]`` in extraction order."""
    strat = get_strategy(strategy) if isinstance(strategy, str) else strategy
    refs = strat(sdfg)
    out: List[Tuple[ExternalCall, Boundary]] = []
    for idx, (parent, node) in enumerate(refs):
        name = f"extcall_{idx}"
        boundary = extract_nest_to_sdfg(parent, node, name=name)
        ext = replace_nsdfg_with_external(boundary, name)
        out.append((ext, boundary))
    return out


#: A granularity maps an SDFG to the nests to externalize.
OffloadGranularity = Strategy

#: Default granularity: skip pure scheduling wrappers.
DEFAULT_GRANULARITY = "skip-taskloops"


def label_nest(node: NestNode) -> str:
    """A short human/agent-readable label for an offload candidate."""
    if isinstance(node, nodes.MapEntry):
        return f"map[{', '.join(node.map.params)}] over {node.map.range}"
    if isinstance(node, LoopRegion):
        return f"loop {node.label}"
    if isinstance(node, ConditionalBlock):
        return f"conditional {node.label} ({len(node.branches)} branches)"
    if isinstance(node, dace.SDFGState):
        return f"state {node.label}"
    raise TypeError(f"not an offload candidate: {type(node).__name__}")


@dataclass(slots=True)
class OffloadCandidate:
    """One nest a granularity would externalize, with its label and whether it may carry an OpenMP parallel scope."""
    parent_sdfg: dace.SDFG
    node: NestNode
    label: str
    parallel: bool


def offload_candidates(sdfg: dace.SDFG,
                       granularity: Union[str, OffloadGranularity] = DEFAULT_GRANULARITY) -> List[OffloadCandidate]:
    """The nests ``granularity`` would externalize, without mutating ``sdfg`` (detection only, not extraction)."""
    strat = get_strategy(granularity) if isinstance(granularity, str) else granularity
    return [OffloadCandidate(parent, node, label_nest(node), is_parallel_nest(node)) for parent, node in strat(sdfg)]


#: Offload unit granularity (paper Axis 2), coarse -> fine: a whole cfg block, a whole state, or a single map.
OFFLOAD_UNITS = ("cfg", "state", "map")


def state_has_compute(state: dace.SDFGState) -> bool:
    """Whether a state holds real compute (map, tasklet, library node, or nested SDFG); a connectorless
    tasklet (a precondition-trap guard state) does not count."""
    for node in state.nodes():
        if isinstance(node, nodes.Tasklet):
            if node.in_connectors or node.out_connectors:
                return True
        elif isinstance(node, (nodes.MapEntry, nodes.LibraryNode, nodes.NestedSDFG)):
            return True
    return False


def unit_refs(sdfg: dace.SDFG, unit: str) -> List[Tuple[dace.SDFG, NestNode]]:
    """The (parent-SDFG, node) pairs to externalize at one offloading UNIT level, recursive over nested SDFGs."""
    if unit == "map":
        return [(sub, me) for sub in sdfg.all_sdfgs_recursive() for st in sub.all_states()
                for me in top_level_map_entries(st)]
    if unit == "cfg":
        # top-level only (a nested region's parent isn't the SDFG); ConditionalBlock counts too, or a
        # branchy kernel reports zero cfg candidates instead of "not expressible at this unit".
        return [(sub, r) for sub in sdfg.all_sdfgs_recursive() for r in sub.nodes()
                if isinstance(r, (LoopRegion, ConditionalBlock))]
    if unit == "state":
        return [(sub, st) for sub in sdfg.all_sdfgs_recursive() for st in sub.all_states() if state_has_compute(st)]
    raise ValueError(f"unknown offload unit {unit!r}; known: {OFFLOAD_UNITS}")


def offload_unit_axis() -> List[str]:
    """The offloading-granularity axis, coarse -> fine."""
    return list(OFFLOAD_UNITS)


def offload_coarseness(unit: str) -> int:
    """Rank of an offloading unit, 0 = coarsest."""
    return OFFLOAD_UNITS.index(unit)


for unit_name in OFFLOAD_UNITS:  # each unit level is also a registered detection strategy
    register_strategy(unit_name, (lambda u: lambda sdfg: unit_refs(sdfg, u))(unit_name))

__all__ = [
    "OffloadGranularity",
    "DEFAULT_GRANULARITY",
    "OffloadCandidate",
    "offload_candidates",
    "label_nest",
    "OFFLOAD_UNITS",
    "offload_unit_axis",
    "offload_coarseness",
    "unit_refs",
    "state_has_compute",
    "register_strategy",
    "get_strategy",
    "strategy_names",
    "lower_nests_to_external_call",
    "extract_nest_to_sdfg",
    "whole_program_boundary",
]
