# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Session: the consolidated agent-facing API over the 4-phase optimizer.

A stateless model cannot hold a live SDFG node across turns -- a reference goes stale the moment any
transformation mutates the graph, and a stale reference is a silently-wrong move, not an error. The
:class:`Session` closes that gap: the SDFG lives here (server-side), and the agent names graph objects
only by **epoch-stamped string ids**.

The contract:

  * A ``list_*`` call mints ids at the session's current epoch and returns plain data (labels, read/write
    sets, reasons) -- never a node.
  * A mutating call (:meth:`fuse`, :meth:`fission_all`, :meth:`externalize`) bumps the
    epoch and clears every handle, then returns the fresh graph. The tree it returns carries ids again --
    minted at the NEW epoch by :meth:`describe` -- so the agent can act on what it was just handed
    without a re-list; it is only the ids it held from BEFORE the call that are now stale.
  * An id from a past epoch no longer resolves: :meth:`resolve` raises :class:`StaleHandle` telling the
    agent to re-list. That is the safety net -- the "enumerate -> apply one -> re-enumerate" discipline the
    arms require, enforced by the id scheme instead of by the agent remembering to do it.

Every method is one agent tool; every return value is JSON-serializable, so a transport layer (MCP, RPC)
is a thin wrapper over this class and holds no graph state of its own. The same calls drive the
deterministic path -- a strategy is just a scripted policy over ``list_fusions`` / ``fuse``.
"""
from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import dace
from dace.sdfg import nodes
from dace.sdfg.state import ControlFlowBlock, ControlFlowRegion, LoopRegion, SDFGState

from nestforge.build.arena import Cell, run_arena
from nestforge.ir.extract import Boundary, detach, extract_map_nest, find_state_of_node
from nestforge.phases.feedback import run_feedback_loop
from dace.sdfg.state import ConditionalBlock

from nestforge.phases.schedule import FusionMove, apply_fusion, can_fuse, enumerate_fusions, fission_to_statements
from nestforge.ir.introspect import describe_graph, kernel_body, kernel_source, nest_reads_writes
from nestforge.phases.scopes import DEFAULT_GRANULARITY, label_nest, lower_nests_to_external_call, offload_candidates
from nestforge.phases.schedule import RegionMove, apply_region_fusion, enumerate_region_fusions
from nestforge.phases.scopes import is_parallel_nest, top_level_map_entries
from nestforge.corpus.translate import Prepared, emit_sources, prepare

#: kernel_source(lang=...) -> (numpyto --target, generated-source extension). The C backend emits the
#: WHOLE C-family from ONE ``--target c`` run -- a ``.c`` AND a ``.cpp`` -- so bare C++ is the ``.cpp`` of
#: the plain ``c`` target, not ``cpp_omp`` (which would inject OpenMP the agent did not ask for).
LANG_LOWERING = {
    "c": ("c", ".c"),
    "cpp": ("c", ".cpp"),
    "fortran": ("fortran", ".f90"),
}


class StaleHandle(KeyError):
    """Raised when an id from a past epoch is used -- the graph moved under it; re-list and retry."""


def unit_reads_writes(parent: dace.SDFG, node: Union[nodes.MapEntry, LoopRegion]) -> Tuple[List[str], List[str]]:
    """``(reads, writes)`` for one nest, whichever kind it is (map needs its state; loop reads its own
    read/write sets). Thin adapter over :func:`introspect.nest_reads_writes`."""
    container = find_state_of_node(parent, node) if isinstance(node, nodes.MapEntry) else parent
    return nest_reads_writes(container, node)


class Session:
    """Server-side owner of one SDFG and the id registry the agent drives it through."""

    __slots__ = ("sdfg", "name", "epoch", "handles", "work_dir", "prepared")

    def __init__(self, sdfg: dace.SDFG, name: Optional[str] = None, work_dir: Optional[str] = None) -> None:
        self.sdfg = sdfg
        self.name = name or sdfg.label
        self.epoch = 0
        self.handles: Dict[str, object] = {}  # id -> node/move/nest, valid only at self.epoch
        self.work_dir = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="nfsession_"))
        self.prepared: Dict[str, Prepared] = {}  # nest id -> its emitted numpy/yaml package (Phase 3)

    # --- handle machinery -------------------------------------------------------------------------

    def mint(self, kind: str, obj: object) -> str:
        """Register ``obj`` under a fresh epoch-stamped id and return it."""
        hid = f"e{self.epoch}:{kind}:{len(self.handles)}"
        self.handles[hid] = obj
        return hid

    def resolve(self, hid: str, kind: Optional[str] = None) -> object:
        """The object ``hid`` names, or raise. A past-epoch id is a :class:`StaleHandle` (re-list); a
        wrong-``kind`` id is a plain error (the agent mixed a move id with a region id)."""
        if hid not in self.handles:
            # Only a WELL-FORMED past-epoch id is stale ("re-list and retry" is then the right advice).
            # A malformed id is a caller bug, and reporting it as stale would send the agent into a
            # re-list/retry loop that can never succeed.
            stamp = hid.split(":", 1)[0] if ":" in hid else ""
            if not re.fullmatch(r"e\d+", stamp):
                raise KeyError(f"malformed id {hid!r}; expected 'e<epoch>:<kind>:<n>' from a list_* call")
            if stamp != f"e{self.epoch}":
                raise StaleHandle(f"id {hid!r} is from a past epoch (now e{self.epoch}); re-list and retry")
            raise KeyError(f"unknown id {hid!r}")
        obj = self.handles[hid]
        if kind is not None and hid.split(":", 2)[1] != kind:
            raise KeyError(f"id {hid!r} is not a {kind} handle")
        return obj

    def bump(self) -> None:
        """A mutation happened: advance the epoch and drop every handle (all now stale)."""
        self.epoch += 1
        self.handles = {}
        # Nest ids carry the epoch, so every memoized Prepared is now unreachable -- resolve() rejects the
        # id before prepare_nest ever consults the cache. Holding them would pin whole emitted packages
        # (and, after :meth:`feedback` swaps the graph, nodes of an SDFG nobody references any more).
        self.prepared = {}

    # --- Phase 0: see the graph -------------------------------------------------------------------

    def describe(self, bodies: bool = False) -> str:
        """The SDFG as a TEXT tree, every line stamped with the id that acts on it. Read-only; safe at
        any epoch. See :meth:`region_tree` for the same tree as structured data.

        ``bodies=True`` also prints what each leaf kernel COMPUTES, as numpy -- the second projection
        of the same SDFG, for when a fusion choice turns on the arithmetic and not just the shape.

        The ids on the nest lines are the SAME handles :meth:`can_fuse` and :meth:`fuse` resolve, so the
        agent reads a line and acts on it without joining this view against a separate
        :meth:`list_nests` call by eyeballing labels.
        """
        return describe_graph(self.sdfg, handle=self.tree_handle, bodies=bodies)

    def tree_handle(self, kind: str, obj: object) -> str:
        """The id :meth:`describe` stamps on one line. A nest gets a real minted handle, because the
        fusion calls resolve it. A region gets the stable descriptive id :meth:`region_id` hands out --
        no method resolves a ``region`` kind, so minting one would grow the registry on a read-only
        call and hand back an id that raises on the kind guard."""
        return self.mint("nest", obj) if kind == "nest" else self.region_id(obj)

    def region_tree(self) -> dict:
        """The control-flow REGION tree as nested data, one id per container. A ``region`` is a control-flow
        container (``SDFGState`` / ``LoopRegion`` / ``ConditionalBlock`` / ``ControlFlowRegion``) -- the box a
        nest lives in, NOT the nest itself. A ``SDFGState`` is a ``barrier`` and lists the nests it holds;
        the others recurse via ``children``. Two nests fuse only inside one region, so this is the map the
        agent reads before deciding whether a fuse needs a region merge first (:meth:`list_region_fusions`)."""
        return self.region_node(self.sdfg)

    # --- Level 1: region structure (merge the containers) -----------------------------------------

    def list_region_fusions(self) -> List[dict]:
        """Legal region merges right now -- adjacent ``SDFGState`` pairs ``StateFusion`` accepts, each with an
        id for :meth:`fuse_regions`. This is the level ABOVE nest fusion: merging two states so the maps that
        were barred by the state boundary become fusable siblings. (Loop-region merges ride :meth:`fuse` --
        fuse the enclosing loops.)"""
        out: List[dict] = []
        for move in enumerate_region_fusions(self.sdfg):
            hid = self.mint("regmove", move)
            out.append({"id": hid, "kind": move.kind, "label": move.label()})
        return out

    def fuse_regions(self, move_id: str) -> dict:
        """Commit one region merge (from a current :meth:`list_region_fusions`) -- dissolve a state barrier so
        the nests inside can then fuse. Bumps the epoch (all prior ids stale); returns the fresh region tree."""
        move: RegionMove = self.resolve(move_id, "regmove")
        apply_region_fusion(self.sdfg, move)
        self.bump()
        return self.region_tree()

    def kernel_body(self, nest_id: str, form: str = "point") -> List[str]:
        """What one kernel COMPUTES, as numpy statements -- the second projection of the SDFG, for a
        single nest rather than the whole tree.

        ``form="point"`` (the only one built) is scalar and indexed by the kernel's own domain
        variables, which the tree already prints on the kernel line. A reduction renders FOLDED there
        by construction -- ``C[i0] = C[i0] + s1`` -- because an explicit accumulate is the only point
        rendering there is; ``declared`` (``np.sum(...)``) is a whole-array spelling and arrives with
        the slice form.

        Returns the statements, or a single ``<not emitted: ...>`` line when the numpy projection
        cannot express the nest -- a fact the agent needs rather than an exception it cannot act on.
        A ``LoopRegion`` nest has no map body and is refused: it is a loop the parallelizer left
        alone, and its contents are the kernels inside it.
        """
        if form != "point":
            raise ValueError(f"form={form!r} is not built yet; only 'point' exists (slice form is BK4)")
        state, nest = self.map_nest(nest_id)
        return kernel_body(state, self.sdfg, nest, state.scope_children())

    def kernel_source(self, nest_id: str, lang: str = "python") -> str:
        """ONE kernel as a complete, RUNNABLE module -- its representation, not an excerpt.

        ``kernel_body`` returns the statements the tree prints under a line; those reference loop
        variables that exist only inside their headers, so they are a fragment. This returns the whole
        thing: preamble, ``def`` with a real signature, the loop nest. Pure numpy -- no injected
        namespace -- so it runs where it is pasted and a translator can read it without being told what
        ``int_floor`` means.

        ``lang`` in ``{"c", "cpp", "fortran"}`` lowers that SAME numpy through the numpy translator
        (``prepare`` -> ``numpyto``) rather than through a second, divergent emitter: C and C++ are one
        ``--target c`` emit (the ``.c`` and the ``.cpp`` of it), Fortran is ``--target fortran``. The
        translator, not nest-forge, spells the intrinsics -- nest-forge only hands it ``np.<op>``.
        """
        state, nest = self.map_nest(nest_id)  # cheap + refuses a LoopRegion before any emit work
        if lang == "python":
            return kernel_source(state, self.sdfg, nest)
        if lang not in LANG_LOWERING:
            raise ValueError(f"lang={lang!r} is not a kernel language; expected one of "
                             f"'python', {', '.join(repr(k) for k in LANG_LOWERING)}")
        target, ext = LANG_LOWERING[lang]
        prep = self.prepare_bare_nest(nest)
        gen = self.work_dir / prep.name / target
        sources = emit_sources(prep, gen, target=target)
        hit = next((p for p in sources if str(p).endswith(ext)), None)
        if hit is None:  # translator ran but emitted no file for this language -- a real gap, not an excerpt
            raise RuntimeError(f"numpyto emitted no {ext} source for {prep.name!r} (target={target}); "
                               f"got {[str(p) for p in sources]}")
        return Path(hit).read_text()

    def prepare_bare_nest(self, nest: nodes.MapEntry) -> Prepared:
        """The numpy+yaml package for a NON-externalized nest -- extract it on a DETACHED copy so the
        live SDFG is unchanged, then reuse the same ``prepare`` (nest_to_numpy + manifest) the Phase-3
        path uses. Distinct from :meth:`prepare_nest`, which resolves an already-externalized ``extnest``
        handle; this one takes a bare map so :meth:`kernel_source` can lower a kernel the agent has only
        looked at, not offloaded."""
        return prepare(self.nest_boundary_copy(nest), nest.map.label, self.work_dir / nest.map.label)

    def nest_boundary_copy(self, nest: nodes.MapEntry) -> Boundary:
        """A :class:`Boundary` for ``nest`` extracted on a detached copy of the SDFG. ``extract_map_nest``
        outlines the map in place, so it must NOT run on the live graph for a read-only projection. The
        twin map in the copy is found by position (deepcopy preserves state and node order), so this
        needs no node identity across the copy."""
        state = find_state_of_node(self.sdfg, nest)
        state_i = list(self.sdfg.all_states()).index(state)
        node_i = list(state.nodes()).index(nest)
        work = detach(self.sdfg)
        twin_state = list(work.all_states())[state_i]
        twin = list(twin_state.nodes())[node_i]
        return extract_map_nest(work, twin, name=nest.map.label)

    def map_nest(self, nest_id: str) -> Tuple[SDFGState, nodes.MapEntry]:
        """Resolve a nest id to ``(state, MapEntry)``, or refuse. A ``LoopRegion`` has no map body: it
        is a loop the parallelizer left alone, and its kernels are the nests inside it."""
        nest = self.resolve(nest_id, "nest")
        if not isinstance(nest, nodes.MapEntry):
            raise TypeError(f"{nest_id} is a {type(nest).__name__}, which has no map body; "
                            "its kernels are the nests inside it")
        return find_state_of_node(self.sdfg, nest), nest

    # --- Level 2: nest fusion (fuse maps / loops within a region) ----------------------------------

    def list_nests(self) -> List[dict]:
        """Every nest (top-level map-nest or loop-nest) with a fresh id, label, parallel flag, and read/write
        sets. Ids feed :meth:`can_fuse`; the sets tell the agent what each nest computes. These are the units
        Level-2 fusion operates on -- distinct from the offload candidates of :meth:`list_offload_candidates`."""
        out: List[dict] = []
        for container, nest in fusion_units(self.sdfg):
            reads, writes = nest_reads_writes(container, nest)
            hid = self.mint("nest", nest)
            out.append({
                "id": hid,
                "kind": "map" if isinstance(nest, nodes.MapEntry) else "loop",
                "label": label_nest(nest),
                "parallel": is_parallel_nest(nest),
                "reads": reads,
                "writes": writes,
            })
        return out

    def can_fuse(self, first_id: str, second_id: str) -> str:
        """``"yes"`` if the two nests may fuse, else a one-line reason -- the same gate :meth:`fuse` applies,
        so ``"yes"`` is exactly an applicable move. Diagnostic: works on any pair, legal or not. When the
        blocker is a region boundary (two maps in different states), the reason says to merge the enclosing
        regions first (:meth:`list_region_fusions` / :meth:`fuse_regions`)."""
        first = self.resolve(first_id, "nest")
        second = self.resolve(second_id, "nest")
        return can_fuse(self.sdfg, first, second)

    def list_fusions(self) -> List[dict]:
        """Every legal fusion move right now (all three arms), each with an id for :meth:`fuse`. This is
        ``can_fuse == "yes"`` enumerated -- the agent applies one, then re-lists (the epoch bump forces it)."""
        out: List[dict] = []
        for move in enumerate_fusions(self.sdfg):
            hid = self.mint("move", move)
            out.append({"id": hid, "kind": move.kind, "label": move.label()})
        return out

    def fuse(self, move_id: str) -> str:
        """Commit one fusion move (from a current :meth:`list_fusions`). Bumps the epoch -- all prior ids go
        stale -- and returns the fresh graph tree."""
        move: FusionMove = self.resolve(move_id, "move")
        apply_fusion(self.sdfg, move)
        self.bump()
        return self.describe()

    def fission_all(self) -> str:
        """Explode the whole SDFG to statement granularity (the inverse of max-fuse), then re-fuse up to
        taste. Bumps the epoch; returns the fresh tree."""
        fission_to_statements(self.sdfg)
        self.bump()
        return self.describe()

    # --- Phase 2: externalize (hand a nest to the next phase) --------------------------------------

    def list_offload_candidates(self, granularity: str = DEFAULT_GRANULARITY) -> List[dict]:
        """Non-mutating preview: what ``granularity`` WOULD externalize -- each candidate nest labeled,
        parallel/sequential, with its read/write sets. The Phase-2 analog of :meth:`list_fusions`. A
        DISTINCT decision from fusion: which nests leave as library calls, not how nests are fused."""
        out: List[dict] = []
        for cand in offload_candidates(self.sdfg, granularity):
            reads, writes = unit_reads_writes(cand.parent_sdfg, cand.node)
            hid = self.mint("cand", cand)
            out.append({
                "id": hid,
                "label": cand.label,
                "parallel": cand.parallel,
                "reads": reads,
                "writes": writes,
            })
        return out

    def externalize(self, granularity: str = DEFAULT_GRANULARITY) -> List[dict]:
        """Commit Phase 2: swap every selected nest for an ``ExternalCall`` (still runs, numpy-reference
        fallback, so the SDFG stays bit-exact). Bumps the epoch, then mints an id per externalized nest at
        the new epoch -- each carrying the boundary's read (``inputs``) / write (``outputs``) sets, the
        interface Phase 3 compiles against."""
        lowered = lower_nests_to_external_call(self.sdfg, granularity)
        if lowered:
            self.bump()  # only a REAL change costs the agent its handles; a no-op must not strand them
        out: List[dict] = []
        for ext, boundary in lowered:
            hid = self.mint("extnest", (ext, boundary))
            out.append({
                "id": hid,
                "name": ext.name,
                "reads": list(boundary.inputs),
                "writes": list(boundary.outputs),
                "symbols": list(boundary.symbols),
            })
        return out

    # --- Phase 3: optimize a nest's kernel (two evaluation modes) ----------------------------------

    def nest_boundary(self, nest_id: str) -> dict:
        """The externalized nest's interface -- read (``inputs``) / write (``outputs``) / ``symbols`` in
        boundary order, plus the kernel name. What an agent authoring a Mode-A kernel compiles against; the
        boundary order is also the order :meth:`set_kernel`'s ``abi_order`` must match."""
        ext, boundary = self.resolve(nest_id, "extnest")
        return {
            "name": ext.name,
            "inputs": list(boundary.inputs),
            "outputs": list(boundary.outputs),
            "symbols": list(boundary.symbols),
            "boundary_order": list(boundary.inputs) + list(boundary.outputs) + list(boundary.symbols),
        }

    def emit_reference(self, nest_id: str) -> str:
        """Write the nest's numpy reference (the correctness oracle) and return its path. Both modes
        validate against it."""
        return str(self.prepare_nest(nest_id).numpy_path)

    def set_kernel(self, nest_id: str, lib_path: str, symbol: str, abi_order: List[str], fp_mode: str = "") -> dict:
        """Mode A -- point the nest at an agent-authored compiled kernel: a ``.a`` (static, linked into the
        parent ``.so``) or ``.so`` exposing ``symbol`` as an ``extern "C"`` entry. Sets leaf fields only
        (no topology change, so the epoch is unchanged and ids stay valid).

        ``abi_order`` is the SILENT-BREAK field: it must be the arg order the compiled symbol expects, not
        the manifest/role order. Returned alongside ``boundary_order`` so the agent can check them; a
        mismatch is undiagnosed ABI corruption, never an error."""
        ext, boundary = self.resolve(nest_id, "extnest")
        ext.lib_path = lib_path
        ext.symbol = symbol
        ext.abi_order = list(abi_order)
        # Load-bearing: ExternalCall.default_implementation is "DaceReference", so the three leaf fields
        # above are INERT on their own -- expansion would emit the numpy reference, the agent's kernel would
        # never be called, and Mode A would report the reference's time as the agent's result. Correct
        # output plus a plausible number is exactly the failure no assertion catches (differential.py:78
        # carries the same line for the same reason).
        ext.implementation = "ExternCall"
        if fp_mode:
            ext.fp_mode = fp_mode
        return {
            "nest": ext.name,
            "implementation": ext.implementation,
            "abi_order": list(ext.abi_order),
            "boundary_order": list(boundary.inputs) + list(boundary.outputs) + list(boundary.symbols),
        }

    def sweep(self, nest_id: str, sizes: Dict[str, int], reps: int = 100, seed: int = 0) -> dict:
        """Mode B -- sweep ``compiler x fp-mode`` over the framework-emitted C kernel, validate each cell
        bit-exact vs the numpy oracle, time it, and return the winning cell per fp-mode. The framework does
        the building; the agent only supplies the code (or lets the framework emit it)."""
        ext, boundary = self.resolve(nest_id, "extnest")
        prep = self.prepare_nest(nest_id)
        gen = self.work_dir / prep.name / "c"
        sources = emit_sources(prep, gen, target="c")
        c_source = next((Path(p) for p in sources if str(p).endswith(".c")), None)
        if c_source is None:  # bare StopIteration here would surface as an opaque tool error
            raise ValueError(f"emitting {prep.name!r} as C produced no .c source (got {[str(p) for p in sources]}); "
                             "nothing to sweep")
        res = run_arena(prep, boundary, c_source, self.work_dir / prep.name / "arena", sizes, reps=reps, seed=seed)
        return {fp: cell_summary(cell) for fp, cell in res.winners.items()}

    # --- Phase 4: feed measurements back ----------------------------------------------------------

    def feedback(self, measure: Callable, max_rounds: int = 8) -> dict:
        """Re-adjust granularity (re-fuse move by move) until measured time plateaus. ``measure`` is the
        framework's build+time step (supplied by the transport, never the model) -- Phase 4 is driven by
        MEASUREMENT, so it cannot be a pure-data tool. Bumps the epoch (granularity changed)."""
        res = run_feedback_loop(self.sdfg, measure, max_rounds=max_rounds)
        # run_feedback_loop MUTATES the graph it is given and keeps searching past the best round, so
        # self.sdfg is left at whatever granularity the last (rejected) round produced. Adopting
        # res.sdfg -- the snapshot taken when the best was set -- is what makes the reported best_us
        # describe the graph the session then hands to emit/sweep/build. The bump() below is what makes
        # the swap safe: every id into the old graph goes stale in the same step.
        self.sdfg = res.sdfg
        self.bump()
        best = res.best
        return {
            "rounds": res.rounds,
            "best_name": best.proposal.name if best is not None else None,
            "best_us": best.median_us if best is not None else None,
        }

    def prepare_nest(self, nest_id: str) -> Prepared:
        """The nest's numpy+yaml package (:func:`translate.prepare`), memoized per id so emit and sweep
        share one."""
        if nest_id not in self.prepared:
            ext, boundary = self.resolve(nest_id, "extnest")
            self.prepared[nest_id] = prepare(boundary, ext.name, self.work_dir / ext.name)
        return self.prepared[nest_id]

    # --- region-tree walk (used by region_tree / fuse_regions) ------------------------------------

    def region_id(self, block: ControlFlowBlock) -> str:
        """A STABLE, purely descriptive id for a control-flow container in :meth:`region_tree`.

        Deliberately not a minted handle: ``region_tree`` is a read view, and no method resolves a
        ``region`` kind. Minting here would grow ``self.handles`` on every inspection call and hand back ids
        that raise on the kind-guard if an agent tried to use one. The label already names the block
        uniquely within its SDFG, which is all a tree reader needs.
        """
        return f"region:{block.label}"

    def region_node(self, cfg: Union[dace.SDFG, ControlFlowRegion]) -> dict:
        """One control-flow region as structured data: an id, its label/type, and its child blocks."""
        return {
            "id": self.region_id(cfg),
            "label": cfg.label,
            "type": type(cfg).__name__,
            "children": [self.block_node(block) for block in cfg.nodes()],
        }

    def block_node(self, block: ControlFlowBlock) -> dict:
        """One block inside a region: a State (a barrier, listing its nests) or a nested region (recurse)."""
        if isinstance(block, SDFGState):
            nests = []
            for entry in top_level_map_entries(block):
                reads, writes = nest_reads_writes(block, entry)
                nests.append({
                    "label": label_nest(entry),
                    "parallel": is_parallel_nest(entry),
                    "reads": reads,
                    "writes": writes,
                })
            return {
                "id": self.region_id(block),
                "label": block.label,
                "type": "SDFGState",
                "barrier": True,
                "nests": nests
            }
        if isinstance(block, ConditionalBlock):
            branches = [{
                "condition": "else" if cond is None else f"when {cond.as_string}",
                "region": self.region_node(branch),
            } for cond, branch in block.branches]
            return {
                "id": self.region_id(block),
                "label": block.label,
                "type": "ConditionalBlock",
                "selector": "stays in core SDFG",
                "branches": branches
            }
        if isinstance(block, (LoopRegion, ControlFlowRegion)):
            node = self.region_node(block)
            node["sequential"] = isinstance(block, LoopRegion)
            return node
        return {"id": self.region_id(block), "label": block.label, "type": type(block).__name__}


def cell_summary(cell: Cell) -> dict:
    """The agent-relevant fields of an arena :class:`Cell` -- correctness + time + who won."""
    return {
        "compiler": cell.compiler,
        "fp_mode": cell.fp_mode,
        "ok": cell.ok,
        "maxdiff": cell.maxdiff,
        "time_us": cell.time_us,
        "error": cell.error,
    }


def fusion_units(sdfg: dace.SDFG) -> List[Tuple[object, Union[nodes.MapEntry, LoopRegion]]]:
    """The fusable units of ``sdfg`` as ``(container, node)`` pairs: every loop-nest (recursive) and every
    top-level map-nest (per state) -- exactly the node kinds :func:`can_fuse` accepts, matching what
    :func:`introspect.describe_graph` walks. ``container`` is the ``SDFGState`` holding a map (what
    :func:`introspect.nest_reads_writes` needs) or the SDFG itself for a loop (which carries its own
    states, so the container is ignored there). Carrying it here is why :meth:`Session.list_nests` needs
    no ``find_state_of_node`` -- the state is already known at the point each map is collected."""
    units: List[Tuple[object, Union[nodes.MapEntry, LoopRegion]]] = []
    for cfg in sdfg.all_control_flow_regions(recursive=True):
        for node in cfg.nodes():
            if isinstance(node, LoopRegion):
                units.append((sdfg, node))
    for state in sdfg.all_states():
        for entry in top_level_map_entries(state):
            units.append((state, entry))
    return units
