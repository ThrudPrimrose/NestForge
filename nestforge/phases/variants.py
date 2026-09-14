# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Configuration sweep: build a kernel per compiler x FP mode x cost model, keep the fastest correct build."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from nestforge.build import flags
from nestforge.build.arena import make_inputs, run_oracle
from nestforge.build.dedup import collapse, representatives, variant_key
from nestforge.build.toolchain import Toolchain
from nestforge.corpus.translate import Prepared
from nestforge.phases.kernel import (
    KernelSource,
    KernelVerdict,
    at_rung,
    build_kernel_library,
    failed_verdict,
    measure_kernel,
)


@dataclass(frozen=True, slots=True)
class Variant:
    """One sweep cell: a C++ compiler and the compile flags its axes compose to."""

    compiler: str
    fp_mode: str
    cost_model: str
    flags: Tuple[str, ...]

    @property
    def label(self) -> str:
        return f"{Path(self.compiler).name}:{self.fp_mode}:{self.cost_model}"


@dataclass(slots=True)
class VariantCell:
    """A variant's verdict; ``same_as`` names the cell whose identical artifact was measured instead."""

    variant: Variant
    verdict: KernelVerdict
    archive: Optional[Path] = None
    same_as: str = ""


@dataclass(slots=True)
class VariantResult:
    """Every cell, the collapsed groups, and the fastest correct cell with the entry it links through."""

    cells: List[VariantCell]
    collapsed: List[str]
    winner: Optional[VariantCell]
    symbol: str
    abi_order: List[str]

    @property
    def library(self) -> Optional[Path]:
        return self.winner.archive if self.winner is not None else None


def enumerate_variants(toolchains: Sequence[Toolchain]) -> List[Variant]:
    """Every compiler x FP mode x cost model cell the toolchains support, one per distinct flag set."""
    variants: Dict[Tuple[str, str, Tuple[str, ...]], Variant] = {}
    for tc in toolchains:
        if tc.cxx is None:
            continue
        # flag_matrix already dedups a cost model the family has no knob for onto its default flags
        for fp_mode, cost_model, composed in flags.flag_matrix(tc.fp_family, "c"):
            variants.setdefault(
                (tc.cxx, fp_mode, tuple(composed)), Variant(tc.cxx, fp_mode, cost_model, tuple(composed))
            )
    return list(variants.values())


def build_variants(
    src: KernelSource, variants: Sequence[Variant], out_dir: Path
) -> Tuple[Dict[str, VariantCell], Dict[str, str]]:
    """``(cells by id, artifact key by id)``; a failed build is a cell without a key."""
    cells: Dict[str, VariantCell] = {}
    keys: Dict[str, str] = {}
    for index, variant in enumerate(variants):
        cell_id = f"{index}:{variant.label}"
        try:
            archive = build_kernel_library(src, variant.compiler, list(variant.flags), out_dir / f"v{index}")
        except RuntimeError as err:
            cells[cell_id] = VariantCell(variant, failed_verdict(variant.fp_mode, str(err)))
            continue
        cells[cell_id] = VariantCell(variant, failed_verdict(variant.fp_mode, "not measured"), archive)
        # an artifact that cannot be inspected gets a unique key: failing to read it means measuring it
        keys[cell_id] = variant_key(archive.with_suffix(".so")) or f"unkeyed:{cell_id}"
    return cells, keys


def select_variant(
    src: KernelSource, prep: Prepared, sizes: Dict[str, int], reps: int, variants: Sequence[Variant], out_dir: Path
) -> VariantResult:
    """Build ``variants`` of ``src`` under ``out_dir``, time each distinct artifact once against the NumPy oracle
    (gating every cell at its own FP rung), and return all cells with the fastest correct one as winner."""
    inputs = make_inputs(src.boundary, sizes)
    oracle = run_oracle(prep, src.boundary, inputs, sizes)
    cells, keys = build_variants(src, variants, out_dir)
    for members in collapse(keys).values():
        head = cells[members[0]]
        archive = head.archive
        assert archive is not None, f"{members[0]} has an artifact key but no archive"
        head.verdict = measure_kernel(archive, src, inputs, oracle, sizes, reps, head.variant.fp_mode)
        for twin_id in members[1:]:
            twin = cells[twin_id]
            twin.verdict, twin.same_as = at_rung(head.verdict, twin.variant.fp_mode), members[0]
    correct = [cell for cell in cells.values() if cell.verdict.ok]
    winner = min(correct, key=lambda cell: cell.verdict.time_us) if correct else None
    return VariantResult(list(cells.values()), representatives(keys)[1], winner, src.symbol, list(src.abi_order))
