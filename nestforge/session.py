# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Session: the one API over every phase, shared by the deterministic driver, humans and agents.

The SDFG lives here; callers name graph objects by epoch-stamped string ids. Any mutation bumps the
epoch, so an id from before it raises :class:`StaleHandle` instead of acting on a moved graph.
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import dace
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion, SDFGState

from nestforge.build.toolchain import discover_toolchains
from nestforge.corpus.translate import Prepared, emit_sources, prepare
from nestforge.ir.extract import Boundary, detach, extract_map_nest, find_state_of_node
from nestforge.ir.introspect import describe_graph, kernel_body, kernel_source, nest_reads_writes
from nestforge.phases.feedback import run_feedback_loop
from nestforge.phases.kernel import KernelSource, schedule_kernel, use_kernel_library
from nestforge.phases.normalize import Targets, normalize
from nestforge.phases.offload import offload
from nestforge.phases.schedule import (
    FissionMove,
    FusionMove,
    RegionMove,
    apply_fusion,
    apply_map_fission,
    apply_region_fusion,
    can_fuse,
    enumerate_fusions,
    enumerate_map_fissions,
    enumerate_region_fusions,
    finish_schedule,
    fission_to_statements,
    full_fusion,
    scope_metrics,
)
from nestforge.phases.scopes import (
    is_parallel_nest,
    label_nest,
    lower_nests_to_external_call,
    offload_candidates,
    top_level_map_entries,
)
from nestforge.phases.variants import VariantCell, enumerate_variants, select_variant

#: kernel_source language -> (translator target, generated file suffix). C and C++ come from one C emit.
LANG_LOWERING = {"c": ("c", ".c"), "cpp": ("c", ".cpp"), "fortran": ("fortran", ".f90")}


class StaleHandle(KeyError):
    """An id from a past epoch: the graph changed under it; list again and retry."""


class Session:
    """Owner of one program SDFG and the ids callers drive it through."""

    __slots__ = ("sdfg", "name", "targets", "epoch", "handles", "work_dir", "prepared", "kernel_sources")

    def __init__(
        self,
        sdfg: dace.SDFG,
        targets: Optional[Targets] = None,
        name: Optional[str] = None,
        work_dir: Optional[str] = None,
    ) -> None:
        self.sdfg = sdfg
        self.targets = targets if targets is not None else Targets()
        self.name = name or sdfg.label
        self.epoch = 0
        self.handles: Dict[str, object] = {}
        self.work_dir = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="nfsession_"))
        self.prepared: Dict[str, Prepared] = {}
        self.kernel_sources: Dict[str, KernelSource] = {}

    # Ids

    def mint(self, kind: str, obj: object) -> str:
        hid = f"e{self.epoch}:{kind}:{len(self.handles)}"
        self.handles[hid] = obj
        return hid

    def resolve(self, hid: str, kind: Optional[str] = None) -> object:
        """The object ``hid`` names; raises :class:`StaleHandle` for a well-formed id from a past epoch."""
        if hid not in self.handles:
            stamp = hid.split(":", 1)[0] if ":" in hid else ""
            if not re.fullmatch(r"e\d+", stamp):
                raise KeyError(f"malformed id {hid!r}; expected 'e<epoch>:<kind>:<n>' from a list call")
            if stamp != f"e{self.epoch}":
                raise StaleHandle(f"id {hid!r} is from a past epoch (now e{self.epoch}); list again and retry")
            raise KeyError(f"unknown id {hid!r}")
        if kind is not None and hid.split(":", 2)[1] != kind:
            raise KeyError(f"id {hid!r} is not a {kind} handle")
        return self.handles[hid]

    def bump(self) -> None:
        self.epoch += 1
        self.handles = {}
        self.prepared = {}
        self.kernel_sources = {}

    # Phase 0: normalize

    def normalize(self) -> str:
        """Canonicalize up to the fusion stage for the session's targets; returns the new tree."""
        normalize(self.sdfg, self.targets)
        self.bump()
        return self.describe()

    # Phase 1: inter-kernel schedule

    def describe(self, bodies: bool = False, metrics: bool = False) -> str:
        """The program as a text tree; nest lines carry :meth:`can_fuse` ids, and scope metrics if ``metrics``."""
        return describe_graph(
            self.sdfg, handle=self.tree_handle, bodies=bodies, metrics=self.metrics_suffix if metrics else None
        )

    def metrics_suffix(self, entry: nodes.MapEntry) -> str:
        return scope_metrics(self.sdfg, entry).suffix()

    def tree_handle(self, kind: str, obj: object) -> str:
        return self.mint("nest", obj) if kind == "nest" else f"region:{obj.label}"

    def list_nests(self) -> List[dict]:
        """Every map-nest and loop-nest with an id, label, parallel flag and read/write sets."""
        out: List[dict] = []
        for container, nest in fusion_units(self.sdfg):
            reads, writes = nest_reads_writes(container, nest)
            out.append(
                {
                    "id": self.mint("nest", nest),
                    "kind": "map" if isinstance(nest, nodes.MapEntry) else "loop",
                    "label": label_nest(nest),
                    "parallel": is_parallel_nest(nest),
                    "reads": reads,
                    "writes": writes,
                }
            )
        return out

    def can_fuse(self, first_id: str, second_id: str) -> str:
        """``"yes"`` or a one-line reason; the same gate :meth:`fuse` applies."""
        return can_fuse(self.sdfg, self.resolve(first_id, "nest"), self.resolve(second_id, "nest"))

    def list_fusions(self) -> List[dict]:
        return [{"id": self.mint("move", m), "kind": m.kind, "label": m.label()} for m in enumerate_fusions(self.sdfg)]

    def fuse(self, move_id: str) -> str:
        move: FusionMove = self.resolve(move_id, "move")
        apply_fusion(self.sdfg, move)
        self.bump()
        return self.describe()

    def list_region_fusions(self) -> List[dict]:
        """Adjacent state pairs that may merge, so nests in them can fuse afterwards."""
        return [
            {"id": self.mint("regmove", m), "kind": m.kind, "label": m.label()}
            for m in enumerate_region_fusions(self.sdfg)
        ]

    def fuse_regions(self, move_id: str) -> str:
        move: RegionMove = self.resolve(move_id, "regmove")
        apply_region_fusion(self.sdfg, move)
        self.bump()
        return self.describe()

    def fission_all(self) -> str:
        """Split the program to statement granularity."""
        fission_to_statements(self.sdfg)
        self.bump()
        return self.describe()

    def list_fissions(self) -> List[dict]:
        """Every legal single-pair map-fission split right now, each naming which nest and where it splits."""
        return [{"id": self.mint("fission", m), "label": m.label()} for m in enumerate_map_fissions(self.sdfg)]

    def fission(self, move_id: str) -> str:
        move: FissionMove = self.resolve(move_id, "fission")
        apply_map_fission(self.sdfg, move)
        self.bump()
        return self.describe()

    def full_fusion(self) -> str:
        """Deterministic default: canonicalization's fusion stage and the stages after it."""
        full_fusion(self.sdfg, self.targets)
        self.bump()
        return self.describe()

    def finish_schedule(self) -> str:
        """Run the post-fusion stages after a hand-chosen granularity."""
        finish_schedule(self.sdfg, self.targets)
        self.bump()
        return self.describe()

    def kernel_body(self, nest_id: str) -> List[str]:
        state, nest = self.map_nest(nest_id)
        return kernel_body(state, self.sdfg, nest, state.scope_children())

    def kernel_source(self, nest_id: str, lang: str = "python") -> str:
        """One nest as a runnable module: ``python`` (NumPy), ``c``, ``cpp`` or ``fortran``."""
        state, nest = self.map_nest(nest_id)
        if lang == "python":
            return kernel_source(state, self.sdfg, nest)
        if lang not in LANG_LOWERING:
            raise ValueError(f"lang={lang!r}; expected 'python' or one of {sorted(LANG_LOWERING)}")
        target, ext = LANG_LOWERING[lang]
        prep = prepare(self.nest_boundary_copy(nest), nest.map.label, self.work_dir / nest.map.label)
        sources = emit_sources(prep, self.work_dir / prep.name / target, target=target)
        hit = next((p for p in sources if str(p).endswith(ext)), None)
        if hit is None:
            raise RuntimeError(f"translator emitted no {ext} source for {prep.name!r}; got {[str(p) for p in sources]}")
        return Path(hit).read_text()

    def nest_boundary_copy(self, nest: nodes.MapEntry) -> Boundary:
        """Extract ``nest`` from a detached copy, so a read-only rendering leaves the live graph alone."""
        state = find_state_of_node(self.sdfg, nest)
        state_index = list(self.sdfg.all_states()).index(state)
        node_index = list(state.nodes()).index(nest)
        work = detach(self.sdfg)
        twin_state = list(work.all_states())[state_index]
        return extract_map_nest(work, list(twin_state.nodes())[node_index], name=nest.map.label)

    def map_nest(self, nest_id: str) -> Tuple[SDFGState, nodes.MapEntry]:
        nest = self.resolve(nest_id, "nest")
        if not isinstance(nest, nodes.MapEntry):
            raise TypeError(f"{nest_id} is a {type(nest).__name__}; its kernels are the nests inside it")
        return find_state_of_node(self.sdfg, nest), nest

    # Phase 2: scope definition

    def list_scope_candidates(self) -> List[dict]:
        """The parallel top-level maps phase 2 would extract, without mutating."""
        out: List[dict] = []
        for cand in offload_candidates(self.sdfg):
            container = find_state_of_node(cand.parent_sdfg, cand.node)
            reads, writes = nest_reads_writes(container, cand.node)
            out.append(
                {
                    "id": self.mint("cand", cand),
                    "label": cand.label,
                    "parallel": cand.parallel,
                    "reads": reads,
                    "writes": writes,
                }
            )
        return out

    def define_scopes(self) -> List[dict]:
        """Replace every parallel top-level map with an ``ExternalCall`` kernel; returns kernel ids."""
        lowered = lower_nests_to_external_call(self.sdfg)
        if lowered:
            self.bump()
        return [
            {
                "id": self.mint("kernel", (ext, boundary)),
                "name": ext.name,
                "reads": list(boundary.inputs),
                "writes": list(boundary.outputs),
                "symbols": list(boundary.symbols),
            }
            for ext, boundary in lowered
        ]

    def kernel_boundary(self, kernel_id: str) -> dict:
        """The kernel's interface; ``boundary_order`` is the argument order a library must accept."""
        ext, boundary = self.resolve(kernel_id, "kernel")
        return {
            "name": ext.name,
            "inputs": list(boundary.inputs),
            "outputs": list(boundary.outputs),
            "symbols": list(boundary.symbols),
            "boundary_order": [*boundary.inputs, *boundary.outputs, *boundary.symbols],
        }

    def emit_reference(self, kernel_id: str) -> str:
        """Write the kernel's NumPy oracle and return its path."""
        return str(self.prepare_kernel(kernel_id).numpy_path)

    def prepare_kernel(self, kernel_id: str) -> Prepared:
        if kernel_id not in self.prepared:
            ext, boundary = self.resolve(kernel_id, "kernel")
            self.prepared[kernel_id] = prepare(boundary, ext.name, self.work_dir / ext.name)
        return self.prepared[kernel_id]

    # Phase 3: offload

    def offload(self) -> dict:
        """Give every kernel a device and insert the host/device copies. With a GPU target the graph
        changes, so every earlier id goes stale and the kernels come back under fresh ids."""
        kernels = [(hid, obj) for hid, obj in self.handles.items() if hid.split(":", 2)[1] == "kernel"]
        placement = offload(self.sdfg, self.targets)
        if self.targets.gpu:
            self.bump()
            kernels = [(self.mint("kernel", obj), obj) for _, obj in kernels]
        return {
            "kernels": [
                {"id": hid, "name": ext.name, "device": placement.devices[ext.name]} for hid, (ext, _) in kernels
            ],
            "copies": [list(pair) for pair in placement.copies],
        }

    # Phase 4: optimize kernels

    def optimize_kernel(self, kernel_id: str) -> dict:
        """Apply the default kernel schedule and generate its source with one C entry."""
        ext, boundary = self.resolve(kernel_id, "kernel")
        src = schedule_kernel(ext, boundary, self.targets, self.work_dir / ext.name / "kernel")
        self.kernel_sources[kernel_id] = src
        return {"kernel": ext.name, "symbol": src.symbol, "abi_order": list(src.abi_order)}

    def set_kernel(self, kernel_id: str, lib_path: str, symbol: str, abi_order: List[str], fp_mode: str = "") -> dict:
        """Point a kernel at a compiled library exposing ``symbol``; ``abi_order`` must match its signature."""
        ext, boundary = self.resolve(kernel_id, "kernel")
        use_kernel_library(ext, Path(lib_path), symbol, abi_order)
        if fp_mode:
            ext.fp_mode = fp_mode
        return {
            "kernel": ext.name,
            "abi_order": list(ext.abi_order),
            "boundary_order": [*boundary.inputs, *boundary.outputs, *boundary.symbols],
        }

    # Phase 5: sweep configurations

    def sweep_configurations(
        self, kernel_id: str, sizes: Dict[str, int], reps: int = 10, compilers: Optional[List[str]] = None
    ) -> dict:
        """Build and time the kernel's variants, link the fastest correct one, and summarize the sweep.

        :param sizes: Value of every symbol the kernel needs, used for validation and timing.
        :param compilers: Toolchain names to keep (``gcc``, ``clang``, ...); all discovered ones when ``None``.
        """
        if kernel_id not in self.kernel_sources:
            self.optimize_kernel(kernel_id)
        src = self.kernel_sources[kernel_id]
        ext, _ = self.resolve(kernel_id, "kernel")
        toolchains = [tc for tc in discover_toolchains() if compilers is None or tc.name in compilers]
        result = select_variant(
            src,
            self.prepare_kernel(kernel_id),
            sizes,
            reps,
            enumerate_variants(toolchains),
            self.work_dir / ext.name / "variants",
        )
        winner = result.winner
        if winner is not None and result.library is not None:
            use_kernel_library(ext, result.library, result.symbol, result.abi_order)
            ext.fp_mode = winner.variant.fp_mode
        return {
            "kernel": ext.name,
            "cells": len(result.cells),
            "collapsed": list(result.collapsed),
            "winner": winner.variant.label if winner is not None else None,
            **winner_config(winner),
        }

    # Feedback

    def feedback(self, measure: Callable, max_rounds: int = 8) -> dict:
        """Re-fuse move by move until the measured time stops improving; keeps the best granularity."""
        res = run_feedback_loop(self.sdfg, measure, max_rounds=max_rounds)
        self.sdfg = res.sdfg
        self.bump()
        best = res.best
        return {
            "rounds": res.rounds,
            "best_name": best.name if best is not None else None,
            "best_us": best.median_us if best is not None else None,
        }


def winner_config(winner: Optional[VariantCell]) -> dict:
    """The configuration phase 5 chose: compiler, FP mode, cost model, flags and measured time, or all ``None``."""
    if winner is None:
        return dict.fromkeys(("compiler", "fp_mode", "cost_model", "flags", "time_us"))
    variant = winner.variant
    return {
        "compiler": Path(variant.compiler).name,
        "fp_mode": variant.fp_mode,
        "cost_model": variant.cost_model,
        "flags": list(variant.flags),
        "time_us": winner.verdict.time_us,
    }


def fusion_units(sdfg: dace.SDFG) -> List[Tuple[object, Union[nodes.MapEntry, LoopRegion]]]:
    """``(container, nest)`` for every loop-nest and every top-level map-nest, as :func:`can_fuse` accepts."""
    regions = sdfg.all_control_flow_regions(recursive=True)
    loops = [(sdfg, node) for cfg in regions for node in cfg.nodes() if isinstance(node, LoopRegion)]
    maps = [(state, entry) for state in sdfg.all_states() for entry in top_level_map_entries(state)]
    return loops + maps
