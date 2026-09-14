# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Extract any loop-nest (CFG ``LoopRegion``) or map-nest (``MapEntry``) into a standalone SDFG.

One primitive unifies DaCe's two outliners:
  * ``MapEntry``   -> ``nest_state_subgraph(sdfg, state, state.scope_subgraph(entry))``
  * ``LoopRegion`` -> ``nest_sdfg_subgraph(sdfg, subgraph)``
Both produce a ``NestedSDFG`` whose ``.sdfg`` is the standalone SDFG. :class:`Boundary` records the
in/out data and symbols and keeps a handle on the placed node so a later pass can swap it for an
``ExternalCall``.
"""
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

#: A control-flow block outlined whole: a loop, or a branch and all its alternatives.
CfgNest = Union[LoopRegion, ConditionalBlock]
NestNode = Union[nodes.MapEntry, CfgNest, SDFGState]


@dataclass(slots=True)
class Boundary:
    """The interface of an extracted nest, in the order the arena/libnode will use."""
    inputs: List[str]  # data read by the nest (NestedSDFG in-connectors)
    outputs: List[str]  # data written by the nest (NestedSDFG out-connectors)
    symbols: List[str]  # free symbols (sizes) the nest depends on
    nsdfg_node: nodes.NestedSDFG  # the NestedSDFG placed in the parent (replacement anchor)
    state: SDFGState  # parent state holding ``nsdfg_node``
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
    return Boundary(inputs=inputs,
                    outputs=outputs,
                    symbols=symbols,
                    nsdfg_node=nsdfg_node,
                    state=state,
                    standalone_sdfg=detach(nsdfg_node.sdfg),
                    parent_sdfg=parent_sdfg)


def extract_map_nest(parent_sdfg: dace.SDFG, map_entry: nodes.MapEntry, name: Optional[str] = None) -> Boundary:
    """Outline a whole map scope (entry..exit + body) into a standalone SDFG.

    ``full_data=True`` nests the entire boundary arrays, not just the accessed sub-range -- otherwise
    DaCe shrinks+rebases each connector to its accessed slice (e.g. a ``[1:N-1]`` stencil would pass a
    size-``N-2`` buffer), breaking the generated C signature.
    """
    state = find_state_of_node(parent_sdfg, map_entry)
    subgraph = state.scope_subgraph(map_entry, include_entry=True, include_exit=True)
    nsdfg_node = helpers.nest_state_subgraph(parent_sdfg, state, subgraph, name=name or "nest", full_data=True)
    return boundary_from_nsdfg(nsdfg_node, state, parent_sdfg)


def extract_state_nest(parent_sdfg: dace.SDFG, state: SDFGState, name: Optional[str] = None) -> Boundary:
    """Outline a WHOLE state (all its maps/tasklets as one unit) into a standalone SDFG -- the ``state``
    offloading granularity (coarser than one map, finer than a control-flow region). ``full_data=True``
    nests entire boundary arrays so the emitted C signature is not shrunk to accessed sub-ranges (same
    reason as :func:`extract_map_nest`)."""
    subgraph = SubgraphView(state, state.nodes())
    nsdfg_node = helpers.nest_state_subgraph(parent_sdfg, state, subgraph, name=name or "nest", full_data=True)
    return boundary_from_nsdfg(nsdfg_node, state, parent_sdfg)


def nest_defined_symbols(region: CfgNest) -> set:
    """Symbols DEFINED inside the nest: every loop variable plus every inter-state-edge assignment
    target. Mirrors the ``ndefined_symbols`` set ``helpers.nest_sdfg_subgraph`` builds."""
    syms = set()
    for b in [region, *region.all_control_flow_blocks()]:
        if isinstance(b, LoopRegion) and b.loop_variable and b.init_statement:
            syms.add(b.loop_variable)
    for e in region.all_interstate_edges():
        syms.update(e.data.assignments.keys())
    return syms


def assignment_dtype(sdfg: dace.SDFG, rhs: str) -> dace.dtypes.typeclass:
    """The dtype of an inter-state assignment's right-hand side, inferred from ``sdfg``'s symbol and array
    tables; ``int64`` when it cannot be inferred.

    A loop variable is an integer, but an assignment target is not necessarily one: canonicalization stages
    a float read (``delta = A[i] * 0.5``) as exactly such a symbol, and declaring it int64 truncates every
    value that crosses the nest boundary -- a wrong answer, not a crash. Inference only ever replaces the
    int64 assumption when it succeeds, so an expression dace cannot type keeps the previous behaviour."""
    table = {s: t for s, t in sdfg.symbols.items()}
    table.update({name: desc.dtype for name, desc in sdfg.arrays.items()})
    try:
        inferred = infer_expr_type(rhs, table)
    except Exception:  # inference walks arbitrary expression ASTs; an untypeable RHS keeps the default
        return dace.int64
    return inferred if isinstance(inferred, dace.dtypes.typeclass) else dace.int64


def nest_defined_symbol_dtypes(sdfg: dace.SDFG, region: CfgNest) -> Dict[str, dace.dtypes.typeclass]:
    """Every symbol :func:`nest_defined_symbols` reports, mapped to the dtype it should be declared with:
    ``int64`` for a loop variable, the inferred type of the assigned expression for an assignment target."""
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
    """Symbols that can change HOW MUCH work ``sdfg`` does: loop init/condition/update statements, map
    ranges, and inter-state-edge conditions -- named as ``sdfg`` itself sees them.

    Interstate ASSIGNMENTS are excluded: they carry a value along the iteration but don't decide whether
    it happens; a condition does.

    Nested SDFGs are RECURSED INTO, with each inner name translated back through the NestedSDFG's
    ``symbol_mapping`` -- skipping this would silently report a buried bound as unsizeable.

    A symbol absent from this set (and from every array shape) can't make a loop zero-trip or a buffer
    empty, so binding it arbitrarily keeps the amount of work intact -- safe to validate against (see
    :func:`nestforge.tsvc.sample_sizes`).
    """
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
    """Outline one control-flow block -- a ``LoopRegion`` or a ``ConditionalBlock`` with all its branches --
    into a standalone SDFG (M1). This is the ``cfg`` offloading unit, the coarsest of the three.

    Nothing here is loop-shaped: the outliner is a ``SubgraphView`` over the single block, and the symbol
    pre-declaration walks ``all_control_flow_blocks``/``all_interstate_edges``, which every control-flow
    block answers. A conditional therefore outlines whole, branches included, and the emitter renders it
    as an ``if``/``else`` around the branch bodies.
    """
    # pre-declare each nest-defined symbol: nest_sdfg_subgraph KeyErrors otherwise. The dtype is INFERRED,
    # not int64 by fiat -- nest_sdfg_subgraph's own type-inference fallback reads whatever is registered
    # here, so a float staged across an interstate edge would be locked to int64 and truncated silently.
    for s, dtype in nest_defined_symbol_dtypes(parent_sdfg, region).items():
        if s not in parent_sdfg.symbols:
            parent_sdfg.add_symbol(s, dtype)
    subgraph = SubgraphView(parent_sdfg, [region])
    inner_state = helpers.nest_sdfg_subgraph(parent_sdfg, subgraph)
    # find the NestedSDFG node in the state nest_sdfg_subgraph returned.
    nsdfg_node = next(n for n in inner_state.nodes() if isinstance(n, nodes.NestedSDFG))
    # nest_sdfg_subgraph takes no name (unlike nest_state_subgraph), so apply it here: two loop nests
    # of one kernel would otherwise share DaCe's default label and collide in the build cache.
    if name:
        nsdfg_node.sdfg.name = name
    return boundary_from_nsdfg(nsdfg_node, inner_state, parent_sdfg)


def extract_nest_to_sdfg(parent_sdfg: dace.SDFG, node: NestNode, name: Optional[str] = None) -> Boundary:
    """Extract any map-nest or loop-nest into a standalone SDFG.

    :returns: a :class:`Boundary`; the standalone SDFG is ``boundary.nsdfg_node.sdfg``.
    """
    if isinstance(node, nodes.MapEntry):
        return extract_map_nest(parent_sdfg, node, name=name)
    if isinstance(node, (LoopRegion, ConditionalBlock)):
        return extract_cfg_nest(parent_sdfg, node, name=name)
    if isinstance(node, SDFGState):
        return extract_state_nest(parent_sdfg, node, name=name)
    raise TypeError(f"cannot extract node of type {type(node).__name__}; expected MapEntry, LoopRegion, "
                    "ConditionalBlock or SDFGState")


def whole_program_boundary(sdfg: dace.SDFG) -> Boundary:
    """A :class:`Boundary` wrapping the WHOLE (un-split) kernel SDFG -- the whole-program lane, where each
    external tool gets the entire program instead of one extracted nest.

    No extraction, no parent: ``standalone_sdfg`` is a detached copy of the whole SDFG, and the
    replacement handles are ``None`` (this lane never swaps a libnode). ``inputs``/``outputs`` come from
    the SDFG's read/write sets, restricted to NON-transient arrays -- transients are the kernel's own
    scratch, not part of the caller interface."""
    detached = detach(sdfg)
    read, write = detached.read_and_write_sets()
    arrays = {n for n, desc in detached.arrays.items() if not desc.transient}
    inputs = sorted(a for a in arrays if a in read)
    outputs = sorted(a for a in arrays if a in write)
    symbols = [a for a in detached.arglist() if a not in detached.arrays]
    return Boundary(inputs=inputs,
                    outputs=outputs,
                    symbols=symbols,
                    nsdfg_node=None,
                    state=None,
                    standalone_sdfg=detached,
                    parent_sdfg=None)
