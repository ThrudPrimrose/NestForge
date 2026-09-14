# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The Phase-2 agent's fusion tool surface: enumerate the legal fusion moves on an SDFG and apply one.

Three arms, all validated single-pair DaCe transformations:
  * ``fuse-loops``          -- two adjacent same-range sequential ``LoopRegion`` s (``FuseLoops``)
  * ``fuse-map-vertical``   -- a producer map feeding a consumer map through a transient (``MapFusionVertical``)
  * ``fuse-map-horizontal`` -- two sibling maps over the same range (``MapFusionHorizontal``)

Every move is legality-gated by the transformation's own ``can_be_applied_to``; the agent only ever sees
moves that are semantics-preserving. :func:`apply_fusion` commits one, re-verifying immediately before
apply (the map-fusion transforms assume a fresh ``can_be_applied`` and node references go stale after any
mutation -- so the agent loop is: enumerate -> apply one -> re-enumerate).

Fission (the other half of the Phase-2 lever) lives separately; this module is the fusion half.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import chain
from typing import Dict, Iterator, List, Optional, Type

import dace
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.transformation.dataflow.map_fusion_horizontal import MapFusionHorizontal
from dace.transformation.dataflow.map_fusion_vertical import MapFusionVertical
from dace.transformation.interstate.loop_fusion import LoopFusion as FuseLoops

from nestforge.extract import find_state_of_node


@dataclass(slots=True)
class FusionMove:
    """One legal fusion the agent may apply. ``where`` maps the transformation's ``PatternNode`` names to
    the matched nodes (the ``apply_to`` / ``can_be_applied_to`` keyword arguments)."""
    kind: str
    where: Dict[str, nodes.Node]
    xform: Type = field(repr=False)

    def label(self) -> str:
        return f"{self.kind}({', '.join(str(n) for n in self.where.values())})"


def iter_loop_fusion_moves(sdfg: dace.SDFG) -> Iterator[FusionMove]:
    """Adjacent ``LoopRegion`` pairs FuseLoops accepts (single sequencing edge, same range, legal), yielded
    lazily so :func:`first_fusion` can stop at the first without scanning the rest."""
    for cfg in sdfg.all_control_flow_regions(recursive=True):
        for first in cfg.nodes():
            if not isinstance(first, LoopRegion):
                continue
            out = cfg.out_edges(first)
            if len(out) != 1:
                continue
            second = out[0].dst
            if isinstance(second, LoopRegion) and second is not first and \
                    FuseLoops.can_be_applied_to(sdfg, first=first, second=second):
                yield FusionMove("fuse-loops", {"first": first, "second": second}, FuseLoops)


def loop_fusion_moves(sdfg: dace.SDFG) -> List[FusionMove]:
    return list(iter_loop_fusion_moves(sdfg))


def iter_vertical_map_moves(sdfg: dace.SDFG) -> Iterator[FusionMove]:
    """Producer->consumer map pairs through a TRANSIENT intermediate that MapFusionVertical accepts. A
    non-transient intermediate is a real output -- fusing it away would drop a live result -- so only
    transients are offered. Lazy (see :func:`first_fusion`)."""
    for state in sdfg.all_states():
        for node in state.nodes():
            if not (isinstance(node, nodes.AccessNode) and sdfg.arrays[node.data].transient):
                continue
            producers = [e.src for e in state.in_edges(node) if isinstance(e.src, nodes.MapExit)]
            consumers = [e.dst for e in state.out_edges(node) if isinstance(e.dst, nodes.MapEntry)]
            for mx in producers:
                for me in consumers:
                    if MapFusionVertical.can_be_applied_to(sdfg, first_map_exit=mx, array=node, second_map_entry=me):
                        yield FusionMove("fuse-map-vertical", {
                            "first_map_exit": mx,
                            "array": node,
                            "second_map_entry": me
                        }, MapFusionVertical)


def vertical_map_moves(sdfg: dace.SDFG) -> List[FusionMove]:
    return list(iter_vertical_map_moves(sdfg))


def iter_horizontal_map_moves(sdfg: dace.SDFG) -> Iterator[FusionMove]:
    """Sibling map pairs (same scope, parallel, same range) that MapFusionHorizontal accepts. Lazy: the
    O(entries^2) ``can_be_applied_to`` scan stops at the first legal pair when consumed via
    :func:`first_fusion`."""
    for state in sdfg.all_states():
        scope = state.scope_dict()
        entries = [n for n in state.nodes() if isinstance(n, nodes.MapEntry)]
        for i, first in enumerate(entries):
            for second in entries[i + 1:]:
                if scope[first] is not scope[second]:
                    continue
                if MapFusionHorizontal.can_be_applied_to(sdfg,
                                                         first_parallel_map_entry=first,
                                                         second_parallel_map_entry=second):
                    yield FusionMove("fuse-map-horizontal", {
                        "first_parallel_map_entry": first,
                        "second_parallel_map_entry": second
                    }, MapFusionHorizontal)


def horizontal_map_moves(sdfg: dace.SDFG) -> List[FusionMove]:
    return list(iter_horizontal_map_moves(sdfg))


def enumerate_fusions(sdfg: dace.SDFG) -> List[FusionMove]:
    """Every legal fusion move on ``sdfg`` right now, across all three arms. The agent picks one, applies it
    (:func:`apply_fusion`), and re-enumerates -- applying a fusion invalidates the other moves' node
    references."""
    return loop_fusion_moves(sdfg) + vertical_map_moves(sdfg) + horizontal_map_moves(sdfg)


def first_fusion(sdfg: dace.SDFG) -> Optional[FusionMove]:
    """The first legal fusion move, in the SAME order :func:`enumerate_fusions` lists them (loop, then
    vertical, then horizontal) -- i.e. exactly ``enumerate_fusions(sdfg)[0]`` when one exists, else
    ``None``. Stops at the first hit instead of materializing every arm: the greedy granularity policies
    (:func:`nestforge.granularity.fuse_first_k`, :func:`nestforge.granularity.fusion_depth`,
    :func:`nestforge.feedback.default_fuse_step`) apply one move then re-scan, so they only ever use
    ``moves[0]`` -- building the whole list (notably the O(entries^2) horizontal ``can_be_applied_to``
    sweep) just to discard all but the first is wasted work."""
    return next(chain(iter_loop_fusion_moves(sdfg), iter_vertical_map_moves(sdfg), iter_horizontal_map_moves(sdfg)),
                None)


def apply_fusion(sdfg: dace.SDFG, move: FusionMove) -> None:
    """Commit one fusion move. Re-verifies (``verify=True``) immediately before applying -- the map-fusion
    transforms assume a fresh ``can_be_applied`` -- so pass only a move from a CURRENT
    :func:`enumerate_fusions` on this exact SDFG state."""
    move.xform.apply_to(sdfg, verify=True, annotate=False, save=False, **move.where)


STATE_BARRIER = ("nests are in different states -- a State boundary is a control-flow dependency, and map "
                 "fusion never crosses one. It is not permanent: merge the enclosing regions first "
                 "(fuse_regions / list_region_fusions, i.e. StateFusion) and these nests become fusable.")


def can_fuse(sdfg: dace.SDFG, first: nodes.Node, second: nodes.Node) -> str:
    """Diagnose whether ``first`` and ``second`` may fuse: ``"yes"`` if a legal arm applies, else a one-line
    reason. Same gates as :func:`enumerate_fusions`, so a ``"yes"`` here is exactly a move
    :func:`apply_fusion` accepts. Shared by the agent and the deterministic path -- the agent reads the
    reason and picks its next move (fission, align granularity, or fuse the states)."""
    if isinstance(first, LoopRegion) and isinstance(second, LoopRegion):
        return fuse_loops_reason(sdfg, first, second)
    if isinstance(first, nodes.MapEntry) and isinstance(second, nodes.MapEntry):
        return fuse_maps_reason(sdfg, first, second)
    return ("cannot fuse a map-nest with a loop-nest directly -- bring both to the same granularity first "
            "(fission the loop to maps, or keep both as loops).")


def fuse_loops_reason(sdfg: dace.SDFG, first: LoopRegion, second: LoopRegion) -> str:
    if first.parent_graph is not second.parent_graph:
        return ("loops are in different control-flow regions (a control-flow dependency separates them); "
                "fuse the ENCLOSING loops first (a fuse-loops move one level up), then these become "
                "siblings -- cannot fuse across the region boundary directly.")
    cfg = first.parent_graph
    out = cfg.out_edges(first)
    if len(out) != 1 or out[0].dst is not second:
        return ("loops are not adjacent: they must be joined by exactly one sequencing edge (first -> "
                "second) with nothing between.")
    if FuseLoops.can_be_applied_to(sdfg, first=first, second=second):
        return "yes"
    return "blocked by FuseLoops: different iteration ranges, or a loop-carried dependency between the two."


def fuse_maps_reason(sdfg: dace.SDFG, first: nodes.MapEntry, second: nodes.MapEntry) -> str:
    state = find_state_of_node(sdfg, first)
    if find_state_of_node(sdfg, second) is not state:
        return STATE_BARRIER
    # Producer -> transient -> consumer is VERTICAL, in whichever order the data flows. A transient path
    # (even between two top-level maps, which are also scope-siblings) means the pair is not horizontal.
    vertical = vertical_reason(sdfg, state, first, second) or vertical_reason(sdfg, state, second, first)
    if vertical is not None:
        return vertical
    scope = state.scope_dict()  # no data path -> HORIZONTAL (independent siblings, same range)
    if scope[first] is not scope[second]:
        return "maps are in different scopes (one is nested inside the other) with no shared data; not a fusion pair."
    if MapFusionHorizontal.can_be_applied_to(sdfg, first_parallel_map_entry=first, second_parallel_map_entry=second):
        return "yes"
    if first.map.range != second.map.range:
        return (f"different map ranges: {first.map.range} vs {second.map.range} -- horizontal fusion needs "
                "the same range.")
    return "blocked by MapFusionHorizontal: not both parallel-compatible, or a data dependency links them."


def vertical_reason(sdfg: dace.SDFG, state: dace.SDFGState, producer: nodes.MapEntry,
                    consumer: nodes.MapEntry) -> Optional[str]:
    """``"yes"``/reason if ``producer`` feeds ``consumer`` through a transient (vertical fusion), else
    ``None`` when no such data path exists (so the caller can try the other direction, then horizontal)."""
    exit_p = state.exit_node(producer)
    # EVERY intermediate is examined, not just the first: ``vertical_map_moves`` offers a move when ANY
    # transient intermediate applies, so returning on the first one would report "live output" for a pair
    # that list_fusions still offers via another (transient) array -- can_fuse and enumerate_fusions
    # disagreeing, and the agent steered away from a legal fusion.
    reasons: List[str] = []
    for e in state.out_edges(exit_p):
        arr = e.dst
        if not isinstance(arr, nodes.AccessNode) or not any(oe.dst is consumer for oe in state.out_edges(arr)):
            continue
        if not sdfg.arrays[arr.data].transient:
            reasons.append(f"intermediate '{arr.data}' is a live output (non-transient); fusing would drop a result")
            continue
        if MapFusionVertical.can_be_applied_to(sdfg, first_map_exit=exit_p, array=arr, second_map_entry=consumer):
            return "yes"  # one applicable intermediate is enough -- that IS the move enumerate offers
        reasons.append(f"blocked by MapFusionVertical on '{arr.data}': shape or dependency mismatch")
    if not reasons:
        return None  # no data path at all: let the caller try the other direction, then horizontal
    return "; ".join(reasons) + " -- cannot fuse."
