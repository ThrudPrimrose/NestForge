# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""nest-forge: extract DaCe loop-/map-nests and offload them to external compilers via an arena.

Two pieces of the optarena dependency are surfaced natively: :mod:`nestforge.translator`
(the numpy -> C/C++/Fortran translator) and :mod:`nestforge.corpus` (the npbench/polybench kernel
corpus). Everything else is nest-forge's own.
"""

# Must precede any ``dace.transformation.interstate`` import: extended's
# passes.canonicalize -> vectorization -> interstate cycle only resolves when ``passes`` loads first.
import dace.transformation.passes  # noqa: F401

from nestforge.corpus import CorpusKernel, dace_kernel_names, iter_dace_kernels
from nestforge.extract import Boundary, extract_nest_to_sdfg
from nestforge.fusion import (FusionMove, FusionStrategy, apply_fusion, enumerate_fusions, fission_to_statements,
                              fusion_strategy_names, get_fusion_strategy, maximal_fusion, map_fission_moves,
                              register_fusion_strategy)
from nestforge.feedback import (AgenticOptimizer, FeedbackResult, Outcome, best_outcome, default_fuse_step, improved,
                                run_agent_loop, run_feedback_loop)
from nestforge.offload import (DEFAULT_GRANULARITY, OffloadCandidate, OffloadGranularity, label_nest,
                               lower_nests_to_external_call, offload_candidates, strategy_names, whole_program_boundary)
# `optimize` (the phase-3 commit function) is deliberately NOT re-exported: the name would bind over the
# `nestforge.optimize` SUBMODULE, so `nestforge.optimize.optimization_choices` would raise AttributeError
# on a function. Reach it as its three sibling phases are reached -- `from nestforge.optimize import optimize`.
from nestforge.optimize import (DEFAULT_OPT_MODE, BuildOptions, DaceOptimizer, ExternalOptimizer, Optimizer, Proposal,
                                OPT_MODES, optimization_choices)
from nestforge.run import NestReport, ProgramReport, optimize_program
from nestforge.strategies import Strategy, get_strategy, outer, register_strategy
# `translate` (the function) is withheld for the reason `optimize` is, above: the name would bind
# over the `nestforge.translate` SUBMODULE. Reach it as `from nestforge.translator import translate`.
from nestforge.translator import BenchSpec

__all__ = [
    "Boundary",
    "extract_nest_to_sdfg",
    # Phase 1: fusion granularity
    "FusionStrategy",
    "register_fusion_strategy",
    "get_fusion_strategy",
    "fusion_strategy_names",
    "maximal_fusion",
    "FusionMove",
    "enumerate_fusions",
    "apply_fusion",
    "fission_to_statements",
    "map_fission_moves",
    # Phase 2: offload granularity
    "OffloadGranularity",
    "DEFAULT_GRANULARITY",
    "OffloadCandidate",
    "offload_candidates",
    "label_nest",
    "Strategy",
    "outer",
    "register_strategy",
    "get_strategy",
    "strategy_names",
    "lower_nests_to_external_call",
    "whole_program_boundary",
    # Phase 3: per-nest optimization ("optimize" itself stays module-scoped -- see the import above)
    "DEFAULT_OPT_MODE",
    "optimization_choices",
    "Optimizer",
    "Proposal",
    "DaceOptimizer",
    "ExternalOptimizer",
    "OPT_MODES",
    "BuildOptions",
    # Phase 4: measurement feedback loop
    "run_feedback_loop",
    "FeedbackResult",
    "default_fuse_step",
    "best_outcome",
    "improved",
    "Outcome",
    "AgenticOptimizer",
    "run_agent_loop",
    # the executing entry point: hand it a kernel, get measured winners
    "optimize_program",
    "ProgramReport",
    "NestReport",
    # native optarena surfaces
    "BenchSpec",
    "CorpusKernel",
    "iter_dace_kernels",
    "dace_kernel_names",
]
