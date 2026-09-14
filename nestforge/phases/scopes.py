# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union

import dace
from dace.libraries.standard.helper import GPU_RESIDENT_STORAGES
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion

from nestforge.ir.emit_numpy import nest_to_numpy
from nestforge.ir.emit_yaml import manifest_dict
from nestforge.ir.extract import Boundary, NestNode, extract_nest_to_sdfg, find_state_of_node
from nestforge.ir.introspect import nest_reads_writes
from nestforge.ir.libnode import ExternalCall, in_conn, out_conn


def top_level_map_entries(state: dace.SDFGState) -> List[nodes.MapEntry]:
    """MapEntry nodes at the top of a state's scope tree (not nested inside another map)."""
    return [n for n in state.scope_children()[None] if isinstance(n, nodes.MapEntry)]


def is_parallel_nest(node: NestNode) -> bool:
    """Whether an extracted nest is PARALLEL (a non-Sequential Map) or SEQUENTIAL (a LoopRegion)."""
    if isinstance(node, nodes.MapEntry):
        return node.map.schedule != dace.ScheduleType.Sequential
    return False  # LoopRegion (or anything non-Map): sequential


def parallel_top_level_maps(sdfg: dace.SDFG) -> List[Tuple[dace.SDFG, nodes.MapEntry]]:
    """Phase 2's scope candidates: one scope per parallel top-level map, anywhere in the SDFG's
    control flow (including inside a loop region)."""
    return [
        (sdfg, entry)
        for state in sdfg.all_states()
        for entry in top_level_map_entries(state)
        if is_parallel_nest(entry)
    ]


def label_nest(node: Union[nodes.MapEntry, LoopRegion]) -> str:
    """A short human/agent-readable label for a map-nest or loop-nest."""
    if isinstance(node, nodes.MapEntry):
        return f"map[{', '.join(node.map.params)}] over {node.map.range}"
    if isinstance(node, LoopRegion):
        return f"loop {node.label}"
    raise TypeError(f"not a nest: {type(node).__name__}")


@dataclass(slots=True)
class OffloadCandidate:
    """One parallel top-level map phase 2 would externalize, with its label."""

    parent_sdfg: dace.SDFG
    node: nodes.MapEntry
    label: str
    parallel: bool


def offload_candidates(sdfg: dace.SDFG) -> List[OffloadCandidate]:
    """The scopes phase 2 would externalize, without mutating ``sdfg`` (detection only, not extraction)."""
    return [
        OffloadCandidate(parent, node, label_nest(node), is_parallel_nest(node))
        for parent, node in parallel_top_level_maps(sdfg)
    ]


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


def kernel_connector(prefixed: Callable[[str], str], conn: Optional[str]) -> Optional[str]:
    """``prefixed(conn)``, or ``None`` for an ordering edge (empty memlet, no connector), so it never becomes
    ``_in_None``."""
    return None if conn is None else prefixed(conn)


def replace_nsdfg_with_external(boundary: Boundary, name: str) -> ExternalCall:
    state = boundary.state
    nsdfg = boundary.nsdfg_node
    # Connectors are prefixed so they never collide with array/symbol names (a LibraryNode rule).
    ext = ExternalCall(
        name,
        inputs={in_conn(i) for i in boundary.inputs},
        outputs={out_conn(o) for o in boundary.outputs},
        numpy_source=nest_to_numpy(boundary, fn_name=name),
        config=manifest_dict(boundary, name),
        standalone_sdfg=reference_sdfg(boundary),
    )
    state.add_node(ext)
    # Fresh memlets per edge (never reuse subsets/memlets); remap connector names.
    for e in state.in_edges(nsdfg):
        state.add_edge(e.src, e.src_conn, ext, kernel_connector(in_conn, e.dst_conn), copy.deepcopy(e.data))
    for e in state.out_edges(nsdfg):
        state.add_edge(ext, kernel_connector(out_conn, e.src_conn), e.dst, e.dst_conn, copy.deepcopy(e.data))
    state.remove_node(nsdfg)
    return ext


def is_host_length1_array(desc: dace.data.Data) -> bool:
    return (
        isinstance(desc, dace.data.Array)
        and not isinstance(desc, dace.data.View)
        and desc.total_size == 1
        and desc.storage not in GPU_RESIDENT_STORAGES
    )


def host_length1_inputs(sdfg: dace.SDFG, entry: nodes.MapEntry) -> List[str]:
    """Read-only inputs of the nest at ``entry`` that are length-1 arrays in host memory."""
    state = find_state_of_node(sdfg, entry)
    reads, writes = nest_reads_writes(state, entry)
    return [name for name in reads if name not in writes and is_host_length1_array(state.sdfg.arrays[name])]


def refuse_host_length1_inputs(refs: List[Tuple[dace.SDFG, nodes.MapEntry]]) -> None:
    """A host kernel takes a scalar input by value, so a length-1 array standing in for one is refused
    rather than converted; only a device pointer (a GPU-resident length-1 array) may carry one."""
    offenders = [(entry.map.label, name) for parent, entry in refs for name in host_length1_inputs(parent, entry)]
    if offenders:
        raise ValueError(
            f"nest inputs {offenders} are length-1 arrays in host memory; declare each as a Scalar, which "
            "crosses the kernel boundary by value"
        )


def lower_nests_to_external_call(sdfg: dace.SDFG) -> List[Tuple[ExternalCall, Boundary]]:
    """Lowers every parallel top-level map into an ``ExternalCall`` node, returning
    ``[(call, boundary), ...]`` in extraction order. Refuses before extracting anything if a nest
    reads a host length-1 array (:func:`refuse_host_length1_inputs`)."""
    refs = parallel_top_level_maps(sdfg)
    refuse_host_length1_inputs(refs)
    out: List[Tuple[ExternalCall, Boundary]] = []
    for idx, (parent, node) in enumerate(refs):
        name = f"extcall_{idx}"
        boundary = extract_nest_to_sdfg(parent, node, name=name)
        ext = replace_nsdfg_with_external(boundary, name)
        out.append((ext, boundary))
    return out


__all__ = [
    "OffloadCandidate",
    "offload_candidates",
    "label_nest",
    "parallel_top_level_maps",
    "top_level_map_entries",
    "is_parallel_nest",
    "lower_nests_to_external_call",
]
