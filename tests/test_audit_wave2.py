# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Second audit wave: the remaining findings from the full-repo review. Unit set, no compile.

Covers the contract/robustness bugs that silently mislead rather than crash -- ``can_fuse`` disagreeing
with ``enumerate_fusions``, a documented FP level crashing flag composition, a runtime linked without an
rpath, a rank>=2 size-1 buffer indexed as a sub-array, and extern-call ABI args with no connector.
"""
import re
from pathlib import Path

import numpy as np
import pytest
import dace

import nestforge

from nestforge.build import arena
from nestforge.build.toolchain import LIBOMP, OpenMPRuntime
from nestforge.ir.emit_libnode import scalar_elem
from nestforge.phases.schedule import can_fuse, enumerate_fusions
from nestforge.ir.libnode import ExternLibEnv, ExternalCall, proto_and_call
from nestforge.build import flags
from nestforge.phases.scopes import top_level_map_entries

N = dace.symbol('N')


@dace.program
def live_and_transient(A: dace.float64[N], B: dace.float64[N], live_out: dace.float64[N], C: dace.float64[N]):
    T = np.empty_like(A)  # transient intermediate
    for i in dace.map[0:N]:
        T[i] = A[i] + B[i]
        live_out[i] = A[i] * 3.0  # a NON-transient result of the same producer map
    for i in dace.map[0:N]:
        C[i] = T[i] * 2.0 + live_out[i]  # consumer reads BOTH intermediates


def map_pairs(sdfg):
    for state in sdfg.all_states():
        entries = top_level_map_entries(state)
        for first in entries:
            for second in entries:
                if first is not second:
                    yield first, second


def test_can_fuse_agrees_with_enumerate_fusions():
    # THE contract: can_fuse == "yes" exactly when an applicable move exists. vertical_reason used to return
    # on the FIRST intermediate, so a live (non-transient) output could mask a legal move that
    # enumerate_fusions still offered via the transient -- the agent told "cannot fuse" about a listed move.
    sdfg = live_and_transient.to_sdfg(simplify=True)
    listed = enumerate_fusions(sdfg)
    for first, second in map_pairs(sdfg):
        verdict = can_fuse(sdfg, first, second)
        if verdict == "yes":
            assert listed, "can_fuse said yes but enumerate_fusions offered nothing"
        else:
            assert isinstance(verdict, str) and verdict  # always an explaining reason, never a bare False


def test_live_output_does_not_mask_a_transient_fusion():
    # the specific shape: if any move is offered for the producer/consumer pair, can_fuse must not report the
    # live output as the blocker.
    sdfg = live_and_transient.to_sdfg(simplify=True)
    assert enumerate_fusions(sdfg), "fixture must produce a fusable pair, else it tests nothing"
    verdicts = [can_fuse(sdfg, a, b) for a, b in map_pairs(sdfg)]
    assert any(v == "yes" for v in verdicts), f"a move is offered but no pair says yes: {verdicts}"


def test_lane_flags_accepts_every_documented_fp_level():
    # ExternalOptimizer/Proposal document fp_mode as FP_LEVELS, but anything except 'strict-ieee' fell into
    # the REDUCED table and raised KeyError during optimizer construction.
    for level in flags.FP_LEVELS:
        out, reason = flags.lane_flags("gnu", level, "none", "c")
        assert out is not None, f"{level} declined: {reason}"


def test_lane_flags_declines_an_unknown_fp_mode():
    out, reason = flags.lane_flags("gnu", "not-a-mode", "none", "c")
    assert out is None and "unknown fp_mode" in reason  # declines like any unsupported axis, never KeyError


def test_openmp_link_flags_carry_an_rpath_for_a_pinned_dir():
    # -L satisfies the LINKER only; without -rpath the built .so has DT_NEEDED and no RUNPATH, so the
    # ctypes.CDLL right after the build fails to find libomp.
    runtime = OpenMPRuntime(name=LIBOMP.name, soname=LIBOMP.soname, lib_dir="/opt/llvm/lib")
    linked = runtime.link_flags("clang")
    assert "-L/opt/llvm/lib" in linked
    assert "-Wl,-rpath,/opt/llvm/lib" in linked


def test_openmp_link_flags_omit_libdir_when_not_pinned():
    runtime = OpenMPRuntime(name=LIBOMP.name, soname=LIBOMP.soname, lib_dir="")
    assert not [f for f in runtime.link_flags("clang") if f.startswith(("-L", "-Wl,-rpath"))]


def desc_of_shape(shape):
    sdfg = dace.SDFG("d")
    sdfg.add_array("s", shape, dace.float64)
    return sdfg.arrays["s"]


def test_scalar_elem_indexes_every_dimension():
    # is_scalar is rank-agnostic (total_size == 1), so a keepdims (1,1) buffer landed here too; name[0]
    # selects a shape-(1,) SUB-ARRAY, not the element.
    assert scalar_elem("s", desc_of_shape([1])) == "s[0]"
    assert scalar_elem("s", desc_of_shape([1, 1])) == "s[0, 0]"
    assert scalar_elem("s", desc_of_shape([1, 1, 1])) == "s[0, 0, 0]"


def extern_call(abi_order, inputs):
    manifest = {
        "array_args": list(abi_order),
        "output_args": [],
        "init": {
            "arrays": {
                a: {
                    "dtype": "float64"
                }
                for a in abi_order
            },
            "scalars": {}
        },
    }
    node = ExternalCall("k", inputs=set(inputs), outputs=set(), config=manifest)
    node.symbol, node.abi_order = "k_fp64", list(abi_order)
    sdfg = dace.SDFG("host")
    state = sdfg.add_state()
    state.add_node(node)
    for conn in inputs:
        name = conn[len("_in_"):]
        sdfg.add_array(name, [8], dace.float64)
        state.add_edge(state.add_read(name), None, node, conn, dace.Memlet.from_array(name, sdfg.arrays[name]))
    return node, state


def test_abi_arg_without_a_connector_is_refused():
    # the compiled signature exposes a caller-allocated scratch buffer that never crosses the ExternalCall
    # boundary; emitting the call anyway referenced an undefined identifier in the tasklet.
    with pytest.raises(ValueError, match="no '_in_scratch' connector"):
        proto_and_call(*extern_call(["A", "scratch"], inputs=["_in_A"]))


def test_extern_lib_env_accumulates_every_nest_library():
    # one shared environment class: assigning (not appending) kept only the LAST expanded nest's library, so
    # every earlier nest's extern-C symbol was unresolved at link.
    ExternLibEnv.reset()
    assert ExternLibEnv.cmake_libraries == []
    ExternLibEnv.configure("/tmp/libone_nest.so")
    ExternLibEnv.configure("/tmp/libtwo_nest.so")
    assert ExternLibEnv.cmake_libraries == ["/tmp/libone_nest.so", "/tmp/libtwo_nest.so"]
    ExternLibEnv.configure("/tmp/libone_nest.so")  # deduplicated
    assert len(ExternLibEnv.cmake_libraries) == 2
    ExternLibEnv.reset()
    assert ExternLibEnv.cmake_libraries == [] and ExternLibEnv.cmake_link_flags == []


def test_every_link_search_path_is_paired_with_an_rpath():
    """Repo-wide invariant: any ``-L<dir>`` we emit is accompanied by ``-Wl,-rpath,<dir>``.

    ``-L`` satisfies the LINKER only. Without the matching rpath the built artifact needs
    LD_LIBRARY_PATH to run, which broke the OpenMP lane (the ctypes.CDLL right after the build could not
    find libomp). Pinning it here keeps every future link site honest -- a built .so must just run.
    """
    package = Path(nestforge.__file__).parent
    offenders = []
    for path in sorted(package.rglob("*.py")):
        lines = path.read_text().splitlines()
        for num, line in enumerate(lines):
            if not re.search(r"""["']-L""", line):
                continue
            window = " ".join(lines[num:num + 2])  # the flag list may wrap onto the next line
            if "rpath" not in window:
                offenders.append(f"{path.name}:{num + 1}: {line.strip()}")
    assert not offenders, "a -L without a paired -Wl,-rpath:\n" + "\n".join(offenders)


def test_ccache_is_auto_detected_but_never_used_where_time_is_measured():
    """The native build pipeline picks up a compiler cache when installed, EXCEPT where compile time is the
    reported measurement -- a cache hit returns in ~0s and would make that number meaningless."""
    from nestforge.build.toolchain import ccache_available, ccache_prefix
    auto = ccache_prefix(None)
    assert auto == (["ccache"] if ccache_available() else [])  # AUTO follows availability
    assert ccache_prefix(False) == []  # measurement paths opt out unconditionally


def test_ccache_can_be_disabled_by_env(monkeypatch):
    from nestforge.build import toolchain
    monkeypatch.setenv("NF_NO_CCACHE", "1")
    toolchain.ccache_available.cache_clear()
    assert toolchain.ccache_available() is False
    toolchain.ccache_available.cache_clear()  # leave the probe clean for other tests


def test_compare_link_modes_forces_the_cache_off():
    # the two compile_seconds it returns ARE the comparison, so it must not be served from a cache.
    import inspect
    from nestforge.build import sdfg as build
    src = inspect.getsource(build.compare_link_modes)
    assert src.count("use_ccache=False") == 2  # both the monolithic and external compiles


def test_relative_maxdiff_accepts_a_reassociated_reduction_that_absolute_rejects():
    """A vectorized reduce reassociates: at reduction magnitudes the result lands a few ULP away, which is
    ORDERS above the 1e-14 ABSOLUTE tolerance but correct to ~1e-15 relative. The absolute gate would mark
    every reduction kernel wrong and silently drop it from the sweep."""
    rng = np.random.default_rng(0)
    a = rng.random(32000)
    seq = np.array([sum(a.tolist())])
    vec = np.array([float(a.reshape(-1, 8).sum(axis=0).sum())])
    atol = flags.FP_ATOL["strict-ieee"]

    assert arena.maxdiff({"s": seq}, {"s": vec}) > atol  # the absolute gate rejects a CORRECT answer
    assert arena.relative_maxdiff({"s": seq}, {"s": vec}) <= atol


def test_relative_maxdiff_still_catches_a_real_miscompile():
    """Scaling must not become norm_error: a genuinely wrong element still fails, and small-magnitude
    values keep the strict ABSOLUTE reading because the scale denominator floors at 1.0."""
    good = np.ones(1000)
    bad = good.copy()
    bad[7] = 1.5  # one wrong element among a thousand right ones
    atol = flags.FP_ATOL["strict-ieee"]
    assert arena.relative_maxdiff({"x": good}, {"x": bad}) > atol

    tiny = np.full(4, 1e-9)
    off = tiny + 1e-13  # absolute 1e-13: huge RELATIVE to 1e-9, but the floor keeps it an absolute call
    assert arena.relative_maxdiff({"x": tiny}, {"x": off}) > atol


def test_relative_maxdiff_fails_on_nan_and_inf():
    """NaN/Inf must not sneak through the division."""
    atol = flags.FP_ATOL["strict-ieee"]
    ok = np.array([1.0, 2.0])
    assert not (arena.relative_maxdiff({"x": ok}, {"x": np.array([1.0, np.nan])}) <= atol)
    assert not (arena.relative_maxdiff({"x": ok}, {"x": np.array([1.0, np.inf])}) <= atol)


def test_maxdiff_does_not_swallow_a_nan_in_a_later_output():
    """Builtin max drops a NaN that is not the first item, so a NaN-poisoned second output used to report
    0.0 -- a PERFECT match -- and could be crowned the arena winner."""
    ref = {"x": np.array([1.0]), "y": np.array([1.0])}
    got = {"x": np.array([1.0]), "y": np.array([np.nan])}
    assert arena.maxdiff(ref, got) == float("inf")
    assert arena.maxdiff(ref, ref) == 0.0
