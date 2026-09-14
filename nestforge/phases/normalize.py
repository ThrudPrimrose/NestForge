# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Phase 0: normalize a program to the canonical parallel form, leaving the final fusion to phase 1."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import dace
from dace.transformation.passes.canonicalize import canonicalize, stage_labels
from dace.transformation.passes.symbol_propagation import SymbolPropagation

FUSE_STAGE = "fuse"


@dataclass(frozen=True, slots=True)
class Targets:
    """Devices the program may run on. The CPU is always a target; the GPU is opt-in."""
    gpu: bool = False

    @property
    def canon_target(self) -> str:
        return "gpu" if self.gpu else "cpu"


def normalization_stages(targets: Targets) -> List[str]:
    labels = stage_labels(targets.canon_target)
    return labels[:labels.index(FUSE_STAGE)]


def normalize(sdfg: dace.SDFG, targets: Targets) -> dace.SDFG:
    """Canonicalizes ``sdfg`` in place up to the fusion stage and returns it."""
    # The frontend binds derived loop bounds to fresh interstate symbols that extraction cannot pass in.
    SymbolPropagation().apply_pass(sdfg, {})
    return canonicalize(sdfg, target=targets.canon_target, stages=normalization_stages(targets))
