# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import chain
from typing import Dict, Iterator, List, Optional, Tuple, Type, Union

import dace
import sympy
from dace.sdfg import nodes
from dace.sdfg.performance_evaluation import total_volume, work_depth
from dace.sdfg.state import LoopRegion, SDFGState
from dace.transformation.dataflow.map_fission import MapFission
from dace.transformation.dataflow.map_fusion_horizontal import MapFusionHorizontal
from dace.transformation.dataflow.map_fusion_vertical import MapFusionVertical
from dace.transformation.interstate.loop_fusion import LoopFusion as FuseLoops
from dace.transformation.interstate.state_fusion import StateFusion
from dace.transformation.passes.canonicalize import canonicalize, stage_labels
from dace.transformation.passes.canonicalize.split_statements import SplitStatements
from dace.transformation.passes.loop_fission import LoopFission

from nestforge.ir.extract import detach, extract_cfg_nest, extract_map_nest, find_state_of_node
from nestforge.ir.names import inline_top_level_nsdfgs
from nestforge.phases.normalize import FUSE_STAGE, Targets


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
            if (
                isinstance(second, LoopRegion)
                and second is not first
                and FuseLoops.can_be_applied_to(sdfg, first=first, second=second)
            ):
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
                        yield FusionMove(
                            "fuse-map-vertical",
                            {"first_map_exit": mx, "array": node, "second_map_entry": me},
                            MapFusionVertical,
                        )


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
            for second in entries[i + 1 :]:
                if scope[first] is not scope[second]:
                    continue
                if MapFusionHorizontal.can_be_applied_to(
                    sdfg, first_parallel_map_entry=first, second_parallel_map_entry=second
                ):
                    yield FusionMove(
                        "fuse-map-horizontal",
                        {"first_parallel_map_entry": first, "second_parallel_map_entry": second},
                        MapFusionHorizontal,
                    )


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
    return next(
        chain(iter_loop_fusion_moves(sdfg), iter_vertical_map_moves(sdfg), iter_horizontal_map_moves(sdfg)), None
    )


def apply_fusion(sdfg: dace.SDFG, move: FusionMove) -> None:
    """Commit one fusion move. Re-verifies (``verify=True``) immediately before applying -- the map-fusion
    transforms assume a fresh ``can_be_applied`` -- so pass only a move from a CURRENT
    :func:`enumerate_fusions` on this exact SDFG state."""
    move.xform.apply_to(sdfg, verify=True, annotate=False, save=False, **move.where)


STATE_BARRIER = (
    "nests are in different states -- a State boundary is a control-flow dependency, and map "
    "fusion never crosses one. It is not permanent: merge the enclosing regions first "
    "(fuse_regions / list_region_fusions, i.e. StateFusion) and these nests become fusable."
)


def can_fuse(sdfg: dace.SDFG, first: nodes.Node, second: nodes.Node) -> str:
    """Diagnose whether ``first`` and ``second`` may fuse: ``"yes"`` if a legal arm applies, else a one-line
    reason. Same gates as :func:`enumerate_fusions`, so a ``"yes"`` here is exactly a move
    :func:`apply_fusion` accepts. Shared by the agent and the deterministic path -- the agent reads the
    reason and picks its next move (fission, align granularity, or fuse the states)."""
    if isinstance(first, LoopRegion) and isinstance(second, LoopRegion):
        return fuse_loops_reason(sdfg, first, second)
    if isinstance(first, nodes.MapEntry) and isinstance(second, nodes.MapEntry):
        return fuse_maps_reason(sdfg, first, second)
    return (
        "cannot fuse a map-nest with a loop-nest directly -- bring both to the same granularity first "
        "(fission the loop to maps, or keep both as loops)."
    )


def fuse_loops_reason(sdfg: dace.SDFG, first: LoopRegion, second: LoopRegion) -> str:
    if first.parent_graph is not second.parent_graph:
        return (
            "loops are in different control-flow regions (a control-flow dependency separates them); "
            "fuse the ENCLOSING loops first (a fuse-loops move one level up), then these become "
            "siblings -- cannot fuse across the region boundary directly."
        )
    cfg = first.parent_graph
    out = cfg.out_edges(first)
    if len(out) != 1 or out[0].dst is not second:
        return (
            "loops are not adjacent: they must be joined by exactly one sequencing edge (first -> "
            "second) with nothing between."
        )
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
        return (
            f"different map ranges: {first.map.range} vs {second.map.range} -- horizontal fusion needs the same range."
        )
    return "blocked by MapFusionHorizontal: not both parallel-compatible, or a data dependency links them."


def vertical_reason(
    sdfg: dace.SDFG, state: dace.SDFGState, producer: nodes.MapEntry, consumer: nodes.MapEntry
) -> Optional[str]:
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


def fission_to_statements(sdfg: dace.SDFG) -> int:
    """Explode ``sdfg`` to STATEMENT granularity in place, where a statement is one GLOBAL output written
    from N global inputs, with local temps recomputed (never materialized to a buffer). Returns the number
    of fission steps applied. The inverse of Phase-1's max-fuse; the agent then fuses back up
    (:mod:`nestforge.fusion_arms`) to the chosen granularity.

    Steps: ``SplitStatements(split_maps=True)`` (a straight-line map with several global outputs -> one
    flat map per output, shared local recomputed; a fission-blocking NestedSDFG replicated per output),
    then ``LoopFission`` (sequential loops), then :func:`fission_multi_output_maps` (the remaining
    NestedSDFG-bodied maps -- dependent / indirection).

    NOT ``apply_transformations_repeated(MapFission)``: that splits a map per TASKLET and materializes the
    local temps to size-N arrays (``{t=x*2; A=t+1}`` -> two maps + a buffer ``t``). Statement granularity
    is one map per global output precisely because that is the finest split that keeps a local a scalar.
    """

    applied = 0
    applied += SplitStatements(split_maps=True).apply_pass(sdfg, {}) or 0
    applied += LoopFission().apply_pass(sdfg, {}) or 0
    applied += fission_multi_output_maps(sdfg)
    return applied


def fission_multi_output_maps(sdfg: dace.SDFG) -> int:
    """Fission only the maps NOT yet at statement granularity: a top-level map still writing >=2 distinct
    global outputs (the NestedSDFG-bodied dependent / indirection maps ``SplitStatements`` left). A flat
    single-output map is already a statement and is left ALONE -- MapFission would split its tasklet chain
    and materialize the locals. Returns the number of MapFission applications."""
    applied = 0
    while True:
        target = None
        for state in sdfg.all_states():
            sd = state.scope_dict()
            for entry in [n for n in state.nodes() if isinstance(n, nodes.MapEntry) and sd[n] is None]:
                exit_node = state.exit_node(entry)
                global_outs = {e.data.data for e in state.in_edges(exit_node) if e.data is not None and e.data.data}
                if len(global_outs) < 2:
                    continue
                # expr_index=1 wants a single NestedSDFG body; expr_index=0 the multi-component form.
                bodies = [
                    n for n in state.scope_subgraph(entry, False, False).nodes() if isinstance(n, nodes.NestedSDFG)
                ]
                if len(bodies) == 1 and MapFission.can_be_applied_to(
                    sdfg, expr_index=1, map_entry=entry, nested_sdfg=bodies[0]
                ):
                    target = (1, {"map_entry": entry, "nested_sdfg": bodies[0]})
                elif MapFission.can_be_applied_to(sdfg, expr_index=0, map_entry=entry):
                    target = (0, {"map_entry": entry})
                if target is not None:
                    break
            if target is not None:
                break
        if target is None:
            break
        expr_index, kwargs = target
        MapFission.apply_to(sdfg, expr_index=expr_index, **kwargs)
        applied += 1
    return applied


def map_fission_moves(sdfg: dace.SDFG) -> List[Tuple[nodes.MapEntry, nodes.NestedSDFG]]:
    """``(map_entry, nested_sdfg)`` pairs ``MapFission`` can split (a map whose nested-SDFG body has
    independent output groups) -- the single-pair fission move for fine agent control. Each pair is applied
    with ``MapFission.apply_to(sdfg, expr_index=1, map_entry=me, nested_sdfg=nsdfg)``; the validated body is
    carried in the move so the caller applies the same pair that was checked."""
    moves: List[Tuple[nodes.MapEntry, nodes.NestedSDFG]] = []
    for state in sdfg.all_states():
        for node in state.nodes():
            if not isinstance(node, nodes.MapEntry):
                continue
            # A map entry reaches its body over one edge per connector, so dedup before checking.
            body = dict.fromkeys(e.dst for e in state.out_edges(node) if isinstance(e.dst, nodes.NestedSDFG))
            for nsdfg in body:
                # expr_index=1 is MapFission's map-with-nested-SDFG pattern. The default 0 is the
                # map-with-subgraph pattern, whose match drops `nested_sdfg` and rejects a lone
                # NestedSDFG body as a single component -- i.e. exactly the maps enumerated here.
                if MapFission.can_be_applied_to(sdfg, expr_index=1, map_entry=node, nested_sdfg=nsdfg):
                    moves.append((node, nsdfg))
    return moves


@dataclass(slots=True)
class FissionMove:
    """One legal single-pair split from :func:`map_fission_moves`: ``map_entry``'s nested-SDFG body,
    split at ``nested_sdfg`` into its independent output groups."""

    map_entry: nodes.MapEntry
    nested_sdfg: nodes.NestedSDFG

    def label(self) -> str:
        return f"fission-map({self.map_entry}): splits nested body {self.nested_sdfg} into independent output groups"


def enumerate_map_fissions(sdfg: dace.SDFG) -> List[FissionMove]:
    """:func:`map_fission_moves` wrapped as labeled :class:`FissionMove` objects, mirroring how
    :func:`enumerate_fusions` wraps its arms for the agent-facing API."""
    return [FissionMove(entry, nsdfg) for entry, nsdfg in map_fission_moves(sdfg)]


def apply_map_fission(sdfg: dace.SDFG, move: FissionMove) -> None:
    """Commit one map-fission split (from a CURRENT :func:`enumerate_map_fissions`). ``expr_index=1`` is
    MapFission's map-with-nested-SDFG pattern -- see :func:`map_fission_moves`."""
    MapFission.apply_to(
        sdfg,
        expr_index=1,
        verify=True,
        annotate=False,
        save=False,
        map_entry=move.map_entry,
        nested_sdfg=move.nested_sdfg,
    )


@dataclass(slots=True)
class RegionMove:
    """One legal region merge. ``where`` maps the transformation's ``PatternNode`` names to the matched
    control-flow blocks."""

    kind: str
    where: Dict[str, object]
    xform: Type = field(repr=False)

    def label(self) -> str:
        return f"{self.kind}({', '.join(str(b) for b in self.where.values())})"


def enumerate_region_fusions(sdfg: dace.SDFG) -> List[RegionMove]:
    """Every legal region merge right now: the adjacent ``SDFGState`` pairs ``StateFusion`` accepts -- the
    merge that dissolves the map barrier so cross-state maps become fusable. Enumerated across every
    control-flow region (recursive), mirroring :func:`nestforge.fusion_arms.enumerate_fusions`.

    Loop-region merges ride the fusion arms instead (fuse the enclosing loops). Applying one move stales
    the rest -- re-enumerate after each."""
    return [
        RegionMove("fuse-states", {"first_state": edge.src, "second_state": edge.dst}, StateFusion)
        for cfg in sdfg.all_control_flow_regions(recursive=True)
        for edge in cfg.edges()
        if isinstance(edge.src, SDFGState)
        and isinstance(edge.dst, SDFGState)
        and edge.src is not edge.dst
        and StateFusion.can_be_applied_to(sdfg, first_state=edge.src, second_state=edge.dst)
    ]


def apply_region_fusion(sdfg: dace.SDFG, move: RegionMove) -> None:
    """Commit one region merge (from a CURRENT :func:`enumerate_region_fusions`). Re-verifies legality
    before applying, same as :func:`nestforge.fusion_arms.apply_fusion`."""
    move.xform.apply_to(sdfg, verify=True, annotate=False, save=False, **move.where)


CACHE_MODEL = "map_perfect_loop_none"


@dataclass(slots=True, frozen=True)
class ScopeMetrics:
    """Symbolic cost of one scope; ``oi`` is ``work / bytes``, ``None`` when no counted byte moves."""

    work: sympy.Expr
    depth: sympy.Expr
    bytes: sympy.Expr
    oi: Optional[sympy.Expr]

    def suffix(self) -> str:
        if self.oi is None:
            oi = "-"
        else:
            oi = f"{float(self.oi):.4g}" if self.oi.is_number else str(self.oi)
        return f"work={self.work} depth={self.depth} bytes={self.bytes} OI={oi}"


def standalone_scope(sdfg: dace.SDFG, node: Union[nodes.MapEntry, LoopRegion]) -> dace.SDFG:
    twin_sdfg = detach(sdfg)
    if isinstance(node, nodes.MapEntry):
        state = find_state_of_node(sdfg, node)
        if state.entry_node(node) is not None:
            raise TypeError(f"map {node} is nested in another map; metrics are per top-level map")
        twin_state = twin_sdfg.states()[sdfg.states().index(state)]
        return extract_map_nest(twin_sdfg, twin_state.node(state.node_id(node))).standalone_sdfg
    if isinstance(node, LoopRegion) and node.parent_graph is sdfg:
        return extract_cfg_nest(twin_sdfg, twin_sdfg.nodes()[sdfg.nodes().index(node)]).standalone_sdfg
    raise TypeError(f"{node} is neither a top-level map of a state nor a loop at the top of the SDFG")


def scope_metrics(sdfg: dace.SDFG, node: Union[nodes.MapEntry, LoopRegion]) -> ScopeMetrics:
    """Work, depth, bytes moved and operational intensity of one scope, analyzed on a detached copy.

    :param sdfg: The program holding ``node``; never mutated.
    :param node: A top-level ``MapEntry`` of a state, or a ``LoopRegion`` block of ``sdfg`` itself.
    :returns: The metrics over ``sdfg``'s symbols, bytes under :data:`CACHE_MODEL`.
    """
    scope = standalone_scope(sdfg, node)
    work, depth = work_depth.analyze_sdfg(scope, {}, work_depth.get_tasklet_work_depth, [], False)
    read, write = total_volume.analyze_sdfg(scope, cache_model=CACHE_MODEL)
    moved = dace.symbolic.simplify(read + write)
    oi = dace.symbolic.simplify(work / moved) if moved != 0 else None
    return ScopeMetrics(work, depth, moved, oi)


def post_fusion_stages(targets: Targets) -> List[str]:
    """Canonicalization stages after :data:`FUSE_STAGE`; they run once the granularity is chosen."""
    labels = stage_labels(targets.canon_target)
    return labels[labels.index(FUSE_STAGE) + 1 :]


def full_fusion(sdfg: dace.SDFG, targets: Targets) -> dace.SDFG:
    """Deterministic phase-1 default: canonicalization's own fusion stage, then the post-fusion stages.

    :param sdfg: A normalized SDFG (phase 0 output), possibly with re-inlined kernels.
    :param targets: Picks the canonicalization preset.
    :returns: The same SDFG, fused.
    """
    # Map fusion never descends into a NestedSDFG, and a kernel re-inlined for feedback arrives nested.
    inline_top_level_nsdfgs(sdfg)
    return canonicalize(sdfg, target=targets.canon_target, stages=[FUSE_STAGE, *post_fusion_stages(targets)])


def finish_schedule(sdfg: dace.SDFG, targets: Targets) -> dace.SDFG:
    """Run the post-fusion stages after a hand-chosen (agent or human) granularity."""
    return canonicalize(sdfg, target=targets.canon_target, stages=post_fusion_stages(targets))
