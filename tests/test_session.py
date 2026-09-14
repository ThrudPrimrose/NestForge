# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Session: the epoch-stamped-id safety layer over the 4-phase API, and the three distinct decision axes
it exposes -- region structure (Level 1), nest fusion (Level 2), and offload (Phase 2). These tests cover
the layer Session ADDS -- id minting, the stale-handle guard on every mutation, kind-checking, the
region/nest distinction, and that each tool returns plain JSON-able data (never a live node). The wrapped
transforms have their own tests; here we only prove Session drives them safely.
"""
import numpy as np
import pytest
import dace

from nestforge.phases.normalize import Targets
from nestforge.phases.scopes import top_level_map_entries
from nestforge.session import Session, StaleHandle

N = dace.symbol('N')


@dace.program
def vertical_pair(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    T = np.empty_like(A)  # transient -> the two maps fuse vertically
    for i in dace.map[0:N]:
        T[i] = A[i] + B[i]
    for i in dace.map[0:N]:
        C[i] = T[i] * 2.0


@dace.program
def two_indep(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N], D: dace.float64[N]):
    for i in dace.map[0:N]:
        C[i] = A[i] * 2.0
    for i in dace.map[0:N]:
        D[i] = B[i] * 3.0


@dace.program
def live_and_transient(A: dace.float64[N], B: dace.float64[N], live_out: dace.float64[N], C: dace.float64[N]):
    T = np.empty_like(A)  # transient producer output
    for i in dace.map[0:N]:
        T[i] = A[i] + B[i]
        live_out[i] = A[i] * 3.0  # a second, NON-transient producer output
    for i in dace.map[0:N]:
        C[i] = T[i] * 2.0 + live_out[i]  # consumer reads both


def make_session():
    return Session(vertical_pair.to_sdfg(simplify=True))


def barred_session():
    # simplify=False keeps each map in its own state -> a state barrier between the two nests.
    return Session(two_indep.to_sdfg(simplify=False))


def top_level_map_count(sdfg) -> int:
    return sum(len(top_level_map_entries(state)) for state in sdfg.all_states())


# --- Level 2: nest fusion + the id/epoch safety layer ---------------------------------------------


def test_list_nests_is_plain_data():
    s = make_session()
    tree = s.describe()
    assert tree.startswith("SDFG ") and "[e0:nest:" in tree  # a tree, with acting ids on its nest rows
    nests = s.list_nests()
    assert len(nests) == 2
    # epoch-0 stamped nest ids. Not pinned to :0/:1 -- describe() mints the same handles onto its
    # tree lines, so the counter's absolute value depends on what was inspected first.
    assert all(n["id"].startswith("e0:nest:") for n in nests)
    assert all(s.resolve(n["id"], "nest") is not None for n in nests)
    assert ({"A", "B"}, {"T"}) in [(set(n["reads"]), set(n["writes"])) for n in nests]
    for n in nests:  # JSON-able: only str/bool/list, never a node
        assert isinstance(n["label"], str) and isinstance(n["parallel"], bool)


def test_can_fuse_uses_nest_ids():
    s = make_session()
    a, b = (n["id"] for n in s.list_nests())
    assert s.can_fuse(a, b) == "yes"


def test_fuse_bumps_epoch_and_stales_prior_ids():
    s = make_session()
    nests = s.list_nests()
    moves = s.list_fusions()
    assert len(moves) == 1 and moves[0]["kind"] == "fuse-map-vertical"
    assert s.epoch == 0
    s.fuse(moves[0]["id"])
    assert s.epoch == 1
    with pytest.raises(StaleHandle):  # a nest id from epoch 0 no longer resolves
        s.can_fuse(nests[0]["id"], nests[1]["id"])


def test_fission_all_bumps_epoch():
    s = make_session()
    stale = [m["id"] for m in s.list_fusions()]  # mint some epoch-0 handles
    s.fission_all()
    assert s.epoch == 1
    # Every epoch-0 handle is gone. The dict is not EMPTY: fission_all returns the fresh tree, and
    # describe() stamps live epoch-1 ids on it so the agent can act on what it was just handed.
    assert not any(hid.startswith("e0:") for hid in s.handles)
    for hid in stale:
        with pytest.raises(StaleHandle):
            s.resolve(hid)


def test_resolve_rejects_wrong_kind():
    s = make_session()
    nest_id = s.list_nests()[0]["id"]
    with pytest.raises(KeyError):  # a nest id is not a move id
        s.fuse(nest_id)


def test_unknown_id_at_current_epoch_is_not_stale():
    s = make_session()
    s.list_nests()
    with pytest.raises(KeyError) as ei:  # current-epoch but nonexistent -> plain unknown, not StaleHandle
        s.can_fuse("e0:nest:99", "e0:nest:0")
    assert not isinstance(ei.value, StaleHandle)


# --- Level 1: region structure (containers) + the merge-first ordering rule -----------------------


def test_cross_state_nests_are_blocked_and_name_the_region_merge():
    s = barred_session()
    a, b = (n["id"] for n in s.list_nests())
    reason = s.can_fuse(a, b)
    assert "different states" in reason and "merge the enclosing regions" in reason
    assert s.list_region_fusions(), "a state-fusion move must exist to unblock them"


def test_region_fusion_unblocks_cross_state_nest_fusion():
    s = barred_session()
    assert not s.list_fusions(), "nothing fuses while the state barrier stands"
    while s.list_region_fusions():  # merge regions to a fixed point (agent picks; here we drain)
        s.fuse_regions(s.list_region_fusions()[0]["id"])
    assert s.list_fusions(), "with the states merged, the two maps are now fusable"


def test_fuse_regions_bumps_epoch_and_stales_prior_ids():
    s = barred_session()
    moves = s.list_region_fusions()
    assert moves and moves[0]["kind"] == "fuse-states"
    epoch = s.epoch
    s.fuse_regions(moves[0]["id"])
    assert s.epoch == epoch + 1
    with pytest.raises(StaleHandle):  # the region-move id is now stale
        s.fuse_regions(moves[0]["id"])


# --- Phase 2/3: offload is a distinct axis from fusion --------------------------------------------


def test_offload_candidates_are_distinct_from_nest_fusion():
    s = make_session()
    cands = s.list_scope_candidates()
    assert cands and all(c["id"].startswith("e0:cand:") for c in cands)
    assert all("reads" in c and "writes" in c for c in cands)


def test_define_scopes_mints_kernels_at_new_epoch_with_boundary_sets():
    s = make_session()
    kernels = s.define_scopes()
    assert s.epoch == 1
    assert len(kernels) == 2
    assert all(k["id"].startswith("e1:kernel:") for k in kernels)
    producer = next(k for k in kernels if k["writes"] == ["T"])
    assert producer["reads"] == ["A", "B"] and producer["symbols"] == ["N"]


def test_kernel_boundary_exposes_abi_order_target():
    s = make_session()
    kernel_id = s.define_scopes()[0]["id"]
    info = s.kernel_boundary(kernel_id)
    # boundary_order = inputs + outputs + symbols; set_kernel's abi_order is checked against it
    assert info["boundary_order"] == info["inputs"] + info["outputs"] + info["symbols"]


def test_set_kernel_sets_leaf_fields_without_bumping_epoch():
    s = make_session()
    kernel_id = s.define_scopes()[0]["id"]
    epoch = s.epoch
    out = s.set_kernel(kernel_id, "/abs/libk.a", "k", ["A", "B", "T", "N"])
    assert s.epoch == epoch  # a leaf-field write, ids stay valid
    assert out["abi_order"] == ["A", "B", "T", "N"]
    assert s.kernel_boundary(kernel_id)  # same id still resolves


def test_set_kernel_selects_the_extern_call_expansion():
    """The three leaf fields are inert without this: ExternalCall defaults to DaceReference, so expansion
    would emit the numpy reference and Mode A would time the FRAMEWORK's kernel while reporting it as the
    agent's. The outputs would still be correct and the number still plausible -- nothing else catches it."""
    s = make_session()
    kernel_id = s.define_scopes()[0]["id"]
    ext, _ = s.resolve(kernel_id, "kernel")
    # None, not "DaceReference": dace leaves the field unset and falls back to default_implementation at
    # expand time. Either way the agent's kernel is not the one that runs, so the guard is on "not chosen".
    assert ext.implementation != "ExternCall", "fixture already selects the expansion; test would be vacuous"
    s.set_kernel(kernel_id, "/abs/libk.a", "k", ["A", "B", "T", "N"])
    assert ext.implementation == "ExternCall"


def test_emit_reference_writes_numpy_oracle(tmp_path):
    s = Session(vertical_pair.to_sdfg(simplify=True), work_dir=str(tmp_path))
    kernel_id = s.define_scopes()[0]["id"]
    path = s.emit_reference(kernel_id)
    assert path.endswith(".py")
    with open(path) as f:
        assert "def " in f.read()


def test_malformed_id_is_not_reported_as_stale():
    # a garbage id reported as StaleHandle sends the agent into a re-list/retry loop that can never succeed;
    # it must be flagged as a caller bug instead.
    session = Session(vertical_pair.to_sdfg(simplify=True))
    with pytest.raises(KeyError, match="malformed id"):
        session.resolve("not-an-id")
    with pytest.raises(KeyError, match="malformed id"):
        session.resolve("x:move:0")


def test_noop_define_scopes_does_not_strand_handles():
    # define_scopes with a granularity that selects NO nest changes nothing, so it must not bump the
    # epoch -- bumping would silently invalidate every move id the agent had already enumerated.
    session = Session(vertical_pair.to_sdfg(simplify=True))
    moves = session.list_fusions()
    epoch_before = session.epoch
    assert session.define_scopes("cfg") == []  # a flat kernel has no LoopRegion to define a scope over
    assert session.epoch == epoch_before
    if moves:
        session.resolve(moves[0]["id"], "move")  # the agent's ids survive a no-op


# --- Phase 0/1: normalize -> full_fusion -> fission_all, structural checks -------------------------


def test_normalize_then_full_fusion_bumps_epoch_each_time_and_reduces_top_level_maps():
    sdfg = live_and_transient.to_sdfg(simplify=True)
    frontend_maps = top_level_map_count(sdfg)
    session = Session(sdfg, targets=Targets())
    session.normalize()
    assert session.epoch == 1
    session.full_fusion()
    assert session.epoch == 2
    assert top_level_map_count(session.sdfg) < frontend_maps


def test_fission_all_after_full_fusion_increases_nest_count():
    session = Session(live_and_transient.to_sdfg(simplify=True), targets=Targets())
    session.normalize()
    session.full_fusion()
    fused_nest_count = len(session.list_nests())
    session.fission_all()
    assert len(session.list_nests()) > fused_nest_count
