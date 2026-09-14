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

_REGISTRY: Dict[str, Strategy] = {}


def register_strategy(name: str, fn: Strategy) -> None:
    _REGISTRY[name] = fn


def get_strategy(name: str) -> Strategy:
    if name not in _REGISTRY:
        raise KeyError(f"unknown strategy {name!r}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def strategy_names() -> List[str]:
    return sorted(_REGISTRY)


def top_level_map_entries(state: dace.SDFGState) -> List[nodes.MapEntry]:
    """MapEntry nodes at the top of a state's scope tree (not nested inside another map)."""
    return [n for n in state.scope_children()[None] if isinstance(n, nodes.MapEntry)]


def branch_states(region: ConditionalBlock) -> List[dace.SDFGState]:
    """Direct states of every branch of a conditional, plus those of any conditional nested inside it.

    A kernel guarded by ``if k > 0:`` is a whole state graph the top-level walk cannot see: it is not an
    ``SDFGState`` and not a ``LoopRegion``, so a walk that tests only those two skipped the entire
    conditional and reported "no compute nest" for the kernel (foundation ``s162`` is exactly this).

    Deliberately NOT states inside a nested ``LoopRegion``: extraction carves a ``SubgraphView`` out of
    the parent SDFG's OWN nodes, so a loop that is not a direct node of the root cannot be pulled out --
    and lifting a map from inside it would leave its loop behind.
    """
    states: List[dace.SDFGState] = []
    for _, branch in region.branches:
        for block in branch.nodes():
            if isinstance(block, dace.SDFGState):
                states.append(block)
            elif isinstance(block, ConditionalBlock):
                states.extend(branch_states(block))
    return states


def outer(sdfg: dace.SDFG) -> List[Tuple[dace.SDFG, NestNode]]:
    """Outermost nests of the root SDFG: top-level map-nests + top-level CFG loop regions, plus the
    map-nests of a top-level conditional's branches (see :func:`branch_states`).

    Does not descend into nested SDFGs (those are already 'inside'); other strategies may.
    """
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
    """Whether an extracted nest is PARALLEL within the DaCe scope (its iterations are independent, so
    the emitted kernel may carry an OpenMP parallel scope) or SEQUENTIAL.

    A ``MapEntry`` is parallel unless its schedule is explicitly ``Sequential`` -- DaCe's ``LoopToMap``
    (run in the ``baseline``/``canonicalize`` build) only turns a *provably parallel* loop into a Map, so
    a Map is the parallel signal. A ``LoopRegion`` is a loop that stayed a loop (a loop-carried
    recurrence LoopToMap refused), hence sequential. A WCR reduction inside a parallel map is still
    parallel -- the OpenMP emitter carries it as a ``reduction(...)`` clause, not a serialization.
    """
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
    """Like :func:`outer`, but never externalise a pure *taskloop* wrapper.

    A map whose body is only maps, or a loop whose body is only maps, is a scheduling construct with
    no compute of its own -- offloading it buys nothing. Such wrappers are skipped and the search
    descends to the first compute-bearing nest inside them (the actual kernel).
    """
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
    """Every innermost nest -- a map or loop with no further nest inside it -- across all SDFGs.

    The vectorization-style unit, for both parallel and sequential compute leaves:

    * an **innermost map** has no map nested in its scope (a map cannot contain a loop region);
    * an **innermost loop** (``LoopRegion``) has no nested loop *and* no map inside -- if it held
      maps, those maps would be the innermost units, so the loop is a wrapper, not a leaf.

    A map or loop that wraps a ``NestedSDFG`` is *not* a leaf: the nested SDFG holds the real compute
    and is walked separately by ``all_sdfgs_recursive``, so selecting the wrapper too would offload
    the same compute twice. The two never overlap, so each compute leaf is returned exactly once.
    """
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
    """Why a strategy found no nest to externalise -- distinguishing an honestly EMPTY kernel from one
    whose only compute is a **library node**. We deliberately do NOT offload library nodes: DaCe expands
    each to its fastest available library (BLAS/LAPACK/argreduce/...), so externalising it to a naive
    numpy->C loop would only lose performance. Such a kernel is legitimately skipped -- but with a reason
    that says so, instead of the misleading 'no compute nest'.
    """
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
    """A copy of the standalone SDFG whose boundary arrays are renamed to the node's connectors,
    so the ``DaceReference`` nested-SDFG expansion lines up with the ``ExternalCall`` connectors.

    An in-place array is in BOTH ``inputs`` and ``outputs`` and so carries two connectors, but it is
    one array: the body is renamed to the ``_out_`` name only -- the same single pointer
    :func:`~nestforge.libnode.connector_for` hands the extern-C call for an in-place arg, and the
    parent wires both connectors to the one AccessNode, so ``_out_`` already holds the input values.
    ``_in_`` then carries only the read dependency, but a NestedSDFG connector must still resolve to
    a descriptor, so register one for it (renaming the body to ``_in_`` instead would leave the
    ``_out_`` connector undefined and fail NestedSDFG validation).
    """
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
    """Lower every nest the strategy selects into an ``ExternalCall`` node.

    Defaults to ``skip-taskloops``: offload the compute-bearing nests, not the pure map/loop
    scheduling wrappers around them.

    :returns: ``[(external_call_node, boundary), ...]`` in extraction order.
    """
    strat = get_strategy(strategy) if isinstance(strategy, str) else strategy
    refs = strat(sdfg)
    out: List[Tuple[ExternalCall, Boundary]] = []
    for idx, (parent, node) in enumerate(refs):
        name = f"extcall_{idx}"
        boundary = extract_nest_to_sdfg(parent, node, name=name)
        ext = replace_nsdfg_with_external(boundary, name)
        out.append((ext, boundary))
    return out


#: A Phase-2 offload granularity is a detection strategy: SDFG -> the nests to externalize.
OffloadGranularity = Strategy

#: The default granularity: top-level compute nests (outermost, skipping scheduling wrappers).
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
    """One nest a granularity would externalize -- the parent SDFG it lives in, the nest node, a
    label, and whether its emitted kernel may carry an OpenMP parallel scope (see
    :func:`nestforge.strategies.is_parallel_nest`)."""
    parent_sdfg: dace.SDFG
    node: NestNode
    label: str
    parallel: bool


def offload_candidates(sdfg: dace.SDFG,
                       granularity: Union[str, OffloadGranularity] = DEFAULT_GRANULARITY) -> List[OffloadCandidate]:
    """The nests ``granularity`` would externalize, WITHOUT mutating ``sdfg``.

    Detection is read-only -- extraction happens later in :func:`lower_nests_to_external_call`. Lets
    the agent see the offload set (and each nest's parallel/sequential nature) before committing.
    """
    strat = get_strategy(granularity) if isinstance(granularity, str) else granularity
    return [OffloadCandidate(parent, node, label_nest(node), is_parallel_nest(node)) for parent, node in strat(sdfg)]


#: Offloading granularity UNITS (paper Axis 2), COARSE -> FINE. The structural unit each external call
#: wraps, from the graph itself: a whole ``cfg`` (a ``LoopRegion`` or a ``ConditionalBlock``), a whole
#: ``state`` (an ``SDFGState`` and all the maps it holds), or a single ``map`` (one ``MapEntry`` within a
#: state). Coarser wraps more compute per call; finer isolates one map. A DISTINCT decision from fusion
#: granularity (Axis 1, :mod:`nestforge.granularity`) and COMPOSES with it: a ``map`` offload over the
#: atoms partition puts each statement-atom in its own external call. The coarsest endpoint (no
#: decomposition, the whole program as one unit) is :func:`whole_program_boundary`.
OFFLOAD_UNITS = ("cfg", "state", "map")


def state_has_compute(state: dace.SDFGState) -> bool:
    """Whether a state holds real compute (a map, tasklet, library node, or nested SDFG) -- not a bare
    copy/access-only state, which there is nothing to externalize.

    A tasklet counts only if it has connectors. DaCe's precondition traps (canonicalize's
    ``check_assumption_*``, the scatter-conflict guard) are connectorless CPP tasklets in their own
    state: they read and write nothing, so the state crosses no data and externalizing it yields a
    ``void f(void)`` nest -- an extern call that computes nothing but still links and times.
    """
    for node in state.nodes():
        if isinstance(node, nodes.Tasklet):
            if node.in_connectors or node.out_connectors:
                return True
        elif isinstance(node, (nodes.MapEntry, nodes.LibraryNode, nodes.NestedSDFG)):
            return True
    return False


def unit_refs(sdfg: dace.SDFG, unit: str) -> List[Tuple[dace.SDFG, NestNode]]:
    """The (parent-SDFG, node) pairs to externalize at one offloading UNIT level -- recursive over nested
    SDFGs. ``map`` = every top-level map-nest; ``cfg`` = every ``LoopRegion``; ``state`` = every
    compute-bearing state (externalized whole)."""
    if unit == "map":
        return [(sub, me) for sub in sdfg.all_sdfgs_recursive() for st in sub.all_states()
                for me in top_level_map_entries(st)]
    if unit == "cfg":
        # top-level blocks only: extract_cfg_nest needs the block's parent to BE the SDFG
        # (SubgraphView(parent_sdfg, [block])), so a region nested inside another one is not a cfg unit.
        # A ConditionalBlock counts: it outlines whole, branches included, and skipping it left a branchy
        # kernel reporting ZERO cfg candidates -- "nothing to offload" where the truth was "not expressible".
        return [(sub, r) for sub in sdfg.all_sdfgs_recursive() for r in sub.nodes()
                if isinstance(r, (LoopRegion, ConditionalBlock))]
    if unit == "state":
        return [(sub, st) for sub in sdfg.all_sdfgs_recursive() for st in sub.all_states() if state_has_compute(st)]
    raise ValueError(f"unknown offload unit {unit!r}; known: {OFFLOAD_UNITS}")


def offload_unit_axis() -> List[str]:
    """The offloading-granularity axis, coarse -> fine (Axis 2). ``offload_candidates(sdfg, unit)`` previews
    a unit; ``lower_nests_to_external_call(sdfg, unit)`` commits it (each unit is a registered strategy)."""
    return list(OFFLOAD_UNITS)


def offload_coarseness(unit: str) -> int:
    """Rank of an offloading unit, 0 = coarsest (``cfg``). Lets a sweep order the axis and a search step one
    rung finer/coarser."""
    return OFFLOAD_UNITS.index(unit)


for _unit in OFFLOAD_UNITS:  # each unit level is also a detection strategy, so the existing lowering path works
    register_strategy(_unit, (lambda u: lambda sdfg: unit_refs(sdfg, u))(_unit))

__all__ = [
    "OffloadGranularity",
    "DEFAULT_GRANULARITY",
    "OffloadCandidate",
    "offload_candidates",
    "label_nest",
    # offloading granularity axis (Axis 2): cfg / state / map units
    "OFFLOAD_UNITS",
    "offload_unit_axis",
    "offload_coarseness",
    "unit_refs",
    "state_has_compute",
    # registry (from nestforge.strategies)
    "register_strategy",
    "get_strategy",
    "strategy_names",
    # commit + coarsest-granularity surface (re-exported)
    "lower_nests_to_external_call",
    "extract_nest_to_sdfg",
    "whole_program_boundary",
]
