# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Extract any loop-nest (CFG ``LoopRegion``) or map-nest (``MapEntry``) into a standalone SDFG, via
DaCe's ``nest_state_subgraph``/``nest_sdfg_subgraph`` outliners. :class:`Boundary` records the
in/out data and symbols and keeps a handle on the placed node for a later ``ExternalCall`` swap."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import dace
from dace import symbolic
from dace.sdfg import nodes
from dace.sdfg.graph import SubgraphView
from dace.sdfg.state import ConditionalBlock, LoopRegion, SDFGState
from dace.sdfg.type_inference import infer_expr_type
from dace.transformation import helpers

CfgNest = Union[LoopRegion, ConditionalBlock]
NestNode = Union[nodes.MapEntry, CfgNest]


@dataclass(slots=True)
class Boundary:
    """The interface of an extracted nest, in the order the arena/libnode will use."""

    inputs: List[str]
    outputs: List[str]
    symbols: List[str]
    nsdfg_node: nodes.NestedSDFG  # placed in the parent; the replacement anchor
    state: SDFGState
    standalone_sdfg: dace.SDFG  # detached, independently compilable copy of the nest
    parent_sdfg: dace.SDFG = field(repr=False, default=None)


def detach(sdfg: dace.SDFG) -> dace.SDFG:
    """Deep-copy an outlined nested SDFG and cut its parent links so it stands alone."""
    det = copy.deepcopy(sdfg)
    det.parent = None
    det.parent_sdfg = None
    det.parent_nsdfg_node = None
    det.reset_cfg_list()
    return det


def find_state_of_node(sdfg: dace.SDFG, node: nodes.Node) -> SDFGState:
    """Return the ``SDFGState`` in ``sdfg`` that contains ``node``."""
    for state in sdfg.states():
        if node in state.nodes():
            return state
    raise ValueError(f"node {node} not found in any state of SDFG {sdfg.label}")


def boundary_from_nsdfg(nsdfg_node: nodes.NestedSDFG, state: SDFGState, parent_sdfg: dace.SDFG) -> Boundary:
    inputs = sorted(nsdfg_node.in_connectors.keys())
    outputs = sorted(nsdfg_node.out_connectors.keys())
    symbols = sorted(str(s) for s in nsdfg_node.symbol_mapping.keys())
    return Boundary(
        inputs=inputs,
        outputs=outputs,
        symbols=symbols,
        nsdfg_node=nsdfg_node,
        state=state,
        standalone_sdfg=detach(nsdfg_node.sdfg),
        parent_sdfg=parent_sdfg,
    )


def extract_map_nest(parent_sdfg: dace.SDFG, map_entry: nodes.MapEntry, name: Optional[str] = None) -> Boundary:
    """Outline a whole map scope (entry..exit + body) into a standalone SDFG. ``full_data=True`` nests
    whole boundary arrays, not the accessed sub-range (else DaCe shrinks the connector and breaks the
    generated C signature)."""
    state = find_state_of_node(parent_sdfg, map_entry)
    subgraph = state.scope_subgraph(map_entry, include_entry=True, include_exit=True)
    nsdfg_node = helpers.nest_state_subgraph(parent_sdfg, state, subgraph, name=name or "nest", full_data=True)
    return boundary_from_nsdfg(nsdfg_node, state, parent_sdfg)


def assignment_dtype(sdfg: dace.SDFG, rhs: str) -> dace.dtypes.typeclass:
    """dtype of an interstate assignment's RHS, inferred from ``sdfg``'s symbol/array tables; falls
    back to ``int64`` (a float staged across an edge must not be silently truncated to it)."""
    table = {s: t for s, t in sdfg.symbols.items()}
    table.update({name: desc.dtype for name, desc in sdfg.arrays.items()})
    try:
        inferred = infer_expr_type(rhs, table)
    except Exception:  # inference walks arbitrary expression ASTs; an untypeable RHS keeps the default
        return dace.int64
    return inferred if isinstance(inferred, dace.dtypes.typeclass) else dace.int64


def nest_defined_symbol_dtypes(sdfg: dace.SDFG, region: CfgNest) -> Dict[str, dace.dtypes.typeclass]:
    """Every symbol defined inside the nest (loop variables plus interstate-edge assignment targets),
    mapped to the dtype it should be declared with."""
    dtypes: Dict[str, dace.dtypes.typeclass] = {}
    for b in [region, *region.all_control_flow_blocks()]:
        if isinstance(b, LoopRegion) and b.loop_variable and b.init_statement:
            dtypes[b.loop_variable] = dace.int64
    for e in region.all_interstate_edges():
        for target, rhs in e.data.assignments.items():
            if target not in dtypes:
                dtypes[target] = assignment_dtype(sdfg, str(rhs))
    return dtypes


def trip_count_symbols(sdfg: dace.SDFG) -> set:
    """Symbols that can change how much work ``sdfg`` does: loop init/condition/update statements, map
    ranges, and interstate conditions (not assignments, which carry a value but never gate whether it
    runs). Recurses into NestedSDFGs, translating each inner name back through ``symbol_mapping``."""
    syms = set()
    for block in sdfg.all_control_flow_blocks():
        if isinstance(block, LoopRegion):
            for stmt in (block.init_statement, block.loop_condition, block.update_statement):
                if stmt is not None:
                    syms.update(str(s) for s in stmt.get_free_symbols())
    for state in sdfg.states():
        for node in state.nodes():
            if isinstance(node, nodes.MapEntry):
                syms.update(str(s) for s in node.map.range.free_symbols)
            elif isinstance(node, nodes.NestedSDFG):
                for inner in trip_count_symbols(node.sdfg):
                    bound_to = node.symbol_mapping.get(inner)
                    if bound_to is None:
                        syms.add(inner)  # not remapped: the parent knows it under the same name
                    else:
                        syms.update(str(s) for s in symbolic.pystr_to_symbolic(bound_to).free_symbols)
    for edge in sdfg.all_interstate_edges():
        syms.update(str(s) for s in edge.data.condition.get_free_symbols())
    return syms


def extract_cfg_nest(parent_sdfg: dace.SDFG, region: CfgNest, name: Optional[str] = None) -> Boundary:
    """Outline one control-flow block -- a ``LoopRegion`` or a ``ConditionalBlock`` with all its branches
    -- into a standalone SDFG; the coarsest of the three offload units."""
    # pre-declare with the INFERRED dtype: int64 by fiat would truncate a float staged across an edge.
    for s, dtype in nest_defined_symbol_dtypes(parent_sdfg, region).items():
        if s not in parent_sdfg.symbols:
            parent_sdfg.add_symbol(s, dtype)
    subgraph = SubgraphView(parent_sdfg, [region])
    inner_state = helpers.nest_sdfg_subgraph(parent_sdfg, subgraph)
    nsdfg_node = next(n for n in inner_state.nodes() if isinstance(n, nodes.NestedSDFG))
    # nest_sdfg_subgraph takes no name; apply it here or same-kernel nests collide in the build cache.
    if name:
        nsdfg_node.sdfg.name = name
    return boundary_from_nsdfg(nsdfg_node, inner_state, parent_sdfg)


def extract_nest_to_sdfg(parent_sdfg: dace.SDFG, node: NestNode, name: Optional[str] = None) -> Boundary:
    """Extract any map-nest or loop-nest into a standalone SDFG.

    :returns: a :class:`Boundary`; the standalone SDFG is ``boundary.nsdfg_node.sdfg``."""
    if isinstance(node, nodes.MapEntry):
        return extract_map_nest(parent_sdfg, node, name=name)
    if isinstance(node, (LoopRegion, ConditionalBlock)):
        return extract_cfg_nest(parent_sdfg, node, name=name)
    raise TypeError(
        f"cannot extract node of type {type(node).__name__}; expected MapEntry, LoopRegion, or ConditionalBlock"
    )


def whole_program_boundary(sdfg: dace.SDFG) -> Boundary:
    """A :class:`Boundary` wrapping the WHOLE (un-split) kernel SDFG, for tools that take the entire
    program instead of one extracted nest. ``inputs``/``outputs`` come from the read/write sets,
    restricted to non-transient arrays (transients are the kernel's own scratch)."""
    detached = detach(sdfg)
    read, write = detached.read_and_write_sets()
    arrays = {n for n, desc in detached.arrays.items() if not desc.transient}
    inputs = sorted(a for a in arrays if a in read)
    outputs = sorted(a for a in arrays if a in write)
    symbols = [a for a in detached.arglist() if a not in detached.arrays]
    return Boundary(
        inputs=inputs,
        outputs=outputs,
        symbols=symbols,
        nsdfg_node=None,
        state=None,
        standalone_sdfg=detached,
        parent_sdfg=None,
    )
