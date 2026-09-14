# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Offloading granularity UNITS (paper Axis 2): the structural unit each external call wraps -- cfg / state
/ map, coarse -> fine. Unit set, no compile: candidate selection per unit, whole-state extraction, that
lowering each unit yields a valid SDFG, and -- the check that actually binds -- that it still computes the
same VALUES, run through the emitted numpy. Composition with Axis 1 (fusion granularity) is checked too --
finer fusion exposes more map-units."""
import numpy as np
import dace
from dace import symbolic

from nestforge.ir.emit_numpy import load_emitted, nest_to_numpy, scratch_arrays
from nestforge.phases.scopes import (OFFLOAD_UNITS, offload_candidates, offload_coarseness, offload_unit_axis)
from nestforge.ir.extract import extract_state_nest
from nestforge.phases.scopes import lower_nests_to_external_call
from nestforge.granularity import fuse_first_k
from nestforge.phases.scopes import top_level_map_entries

N = dace.symbol('N')


@dace.program
def two_map(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    T = np.empty_like(A)  # 1 compute state, 2 maps, no control-flow region
    for i in dace.map[0:N]:
        T[i] = A[i] + B[i]
    for i in dace.map[0:N]:
        C[i] = T[i] * 2.0


@dace.program
def recur(A: dace.float64[N], B: dace.float64[N]):
    for i in range(1, N):  # loop-carried -> stays a LoopRegion (a cfg unit)
        B[i] = B[i - 1] + A[i]


@dace.program
def branchy(A: dace.float64[N], B: dace.float64[N], flag: dace.int64):
    if flag > 0:  # a ConditionalBlock: the whole branch, both alternatives, is one cfg unit
        for i in dace.map[0:N]:
            B[i] = A[i] * 2.0
    else:
        for i in dace.map[0:N]:
            B[i] = A[i] + 1.0


def run_lowered(boundary, name, sizes, **args):
    """Run one lowered nest's emitted numpy. Every non-scalar transient is a caller-allocated buffer
    (the C-style contract), so anything the boundary names and the caller did not supply is scratch,
    allocated from its own descriptor rather than assumed size-1."""
    mod = load_emitted(nest_to_numpy(boundary, fn_name=name), name)
    sdfg = boundary.standalone_sdfg
    arrays = sdfg.arrays
    named = set(boundary.inputs) | set(boundary.outputs) | set(scratch_arrays(sdfg))
    scratch = {
        s:
        np.zeros(tuple(int(symbolic.evaluate(d, sizes)) for d in arrays[s].shape),
                 dtype=arrays[s].dtype.as_numpy_dtype())
        for s in named if s not in args
    }
    getattr(mod, name)(**args, **sizes, **scratch)


def test_axis_is_coarse_to_fine():
    assert offload_unit_axis() == ["cfg", "state", "map"]
    assert [offload_coarseness(u) for u in OFFLOAD_UNITS] == [0, 1, 2]  # cfg coarsest, map finest


def test_unit_selection_on_flat_kernel():
    sdfg = two_map.to_sdfg(simplify=True)
    assert len(offload_candidates(sdfg, "map")) == 2  # two maps
    assert len(offload_candidates(sdfg, "state")) == 1  # one compute state wrapping both
    assert len(offload_candidates(sdfg, "cfg")) == 0  # no control-flow region in a flat kernel


def test_cfg_unit_selects_a_control_flow_region():
    sdfg = recur.to_sdfg(simplify=True)
    cands = offload_candidates(sdfg, "cfg")
    assert len(cands) == 1
    assert "loop" in cands[0].label  # a control-flow region, not a state or map


def test_state_extraction_yields_the_state_interface():
    sdfg = two_map.to_sdfg(simplify=True)
    state = next(st for st in sdfg.states() if top_level_map_entries(st))
    boundary = extract_state_nest(sdfg, state)
    assert set(boundary.inputs) == {"A", "B"}  # the state reads A, B
    assert boundary.outputs == ["C"]  # writes C; T is an internal transient, not on the boundary


def test_lowering_each_unit_keeps_the_sdfg_valid():
    for unit, expected in [("map", 2), ("state", 1)]:
        sdfg = two_map.to_sdfg(simplify=True)
        lowered = lower_nests_to_external_call(sdfg, unit)
        assert len(lowered) == expected
        sdfg.validate()  # the numpy-reference fallback keeps the lowered SDFG valid


def test_lowering_cfg_unit_keeps_the_sdfg_valid():
    # the cfg lowering path (extract_cfg_nest via the unit strategy) -- distinct from map/state.
    sdfg = recur.to_sdfg(simplify=True)
    lowered = lower_nests_to_external_call(sdfg, "cfg")
    assert len(lowered) == 1  # the one LoopRegion externalized
    sdfg.validate()


def test_cfg_unit_selects_a_conditional_block():
    """A ConditionalBlock is a cfg unit: it outlines whole, branches included. Skipped, a branchy kernel
    reported ZERO cfg candidates -- indistinguishable from "this kernel has nothing to offload"."""
    sdfg = branchy.to_sdfg(simplify=True)
    cands = offload_candidates(sdfg, "cfg")
    assert len(cands) == 1
    assert "conditional" in cands[0].label and "2 branches" in cands[0].label
    assert not cands[0].parallel  # a branch is not a parallel nest, whatever its bodies are


def test_each_unit_computes_the_same_values_as_the_kernel():
    """THE granularity check: lowering at a unit must not change what the program computes. Validity is
    not the property that matters -- ``validate()`` passes on a nest that emits the wrong extent or drops
    a write, and every earlier unit test asserted only that."""
    rng = np.random.default_rng(0)
    A, B = rng.random(16), rng.random(16)
    want = A + B, (A + B) * 2.0  # T, then C

    for unit, count in (("map", 2), ("state", 1)):
        sdfg = two_map.to_sdfg(simplify=True)
        calls = lower_nests_to_external_call(sdfg, unit)
        assert len(calls) == count
        C = np.zeros(16)
        T = np.zeros(16)
        for i, (ext, boundary) in enumerate(calls):
            args = {k: v for k, v in dict(A=A, B=B, C=C, T=T).items() if k in boundary.inputs + boundary.outputs}
            run_lowered(boundary, f"u_{unit}_{i}", dict(N=16), **args)
        assert np.allclose(C, want[1]), f"{unit} unit changed the result"


def test_the_conditional_unit_computes_both_branches_correctly():
    """The branch is offloaded as ONE call, so the emitted kernel owns the selection: run it both ways."""
    sdfg = branchy.to_sdfg(simplify=True)
    calls = lower_nests_to_external_call(sdfg, "cfg")
    assert len(calls) == 1
    _ext, boundary = calls[0]
    A = np.arange(8, dtype=np.float64)
    for i, (flag, want) in enumerate(((1, A * 2.0), (0, A + 1.0))):
        B = np.zeros(8)
        run_lowered(boundary, f"cond_{i}", dict(N=8), A=A, B=B, flag=flag)
        assert np.allclose(B, want), f"flag={flag}"


def test_composes_with_fusion_granularity():
    # a fine (map) offload sees at least as many units at the atoms partition as at maximal fusion.
    atoms = two_map.to_sdfg(simplify=True)
    fuse_first_k(0)(atoms)
    maximal = two_map.to_sdfg(simplify=True)
    fuse_first_k(99)(maximal)
    assert len(offload_candidates(atoms, "map")) >= len(offload_candidates(maximal, "map"))


def test_a_precondition_guard_state_is_not_an_offload_unit():
    """DaCe's traps (``check_assumption_*``) are connectorless CPP tasklets in their own state: they
    read and write nothing. Counting them as compute externalized the guard state as a nest crossing
    no data, which emits ``void extcall_N_fp64(void)`` -- an extern call that computes nothing yet
    still links and gets timed, so it would have entered the tables as a measurement."""
    from dace.transformation.passes.canonicalize.assume_symbols_nonnegative import insert_assumption_guards
    from nestforge.phases.scopes import state_has_compute, unit_refs

    @dace.program
    def scaled(a: dace.float64[N], b: dace.float64[N]):
        for i in dace.map[0:N]:
            b[i] = a[i] * 2.0

    sdfg = scaled.to_sdfg(simplify=True)
    assert insert_assumption_guards(sdfg) == 1, "test is vacuous without a guard state"
    guard = next(s for s in sdfg.states() if s.label == "_assume_nonneg_syms")
    assert guard.number_of_nodes() == 1, guard.nodes()

    assert not state_has_compute(guard), "a connectorless trap tasklet is not compute"
    assert guard not in [st for _sub, st in unit_refs(sdfg, "state")]
    assert [st for _sub, st in unit_refs(sdfg, "state")], "the real compute state must still be a unit"


def test_a_tasklet_with_connectors_still_counts_as_compute():
    """The narrowing must not swallow ordinary single-tasklet states."""
    from nestforge.phases.scopes import state_has_compute

    sdfg = dace.SDFG("plain")
    sdfg.add_array("a", [N], dace.float64)
    state = sdfg.add_state()
    tasklet = state.add_tasklet("t", {"inp"}, {"out"}, "out = inp + 1.0")
    state.add_edge(state.add_read("a"), None, tasklet, "inp", dace.Memlet("a[0]"))
    state.add_edge(tasklet, "out", state.add_write("a"), None, dace.Memlet("a[0]"))
    assert state_has_compute(state)
