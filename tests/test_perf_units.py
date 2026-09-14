# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Compile-free unit tests for the build/arena plumbing: signature parsing and FP-precision x cost-model
flag composition -- pure logic on synthetic inputs, so no compiler needed.
"""
import ctypes
from pathlib import Path

import numpy as np

from nestforge.build import arena

from nestforge.build.isolation import run_isolated
from nestforge.build import flags
from nestforge.build import harness
from nestforge.build.toolchain import Toolchain


# --- emitted-source signature order (harness.signature_order) ------------------------------------------
def test_signature_order_c_and_fortran_multiline():
    csrc = "void s000_fp64(double* a, double* out, int64_t N) {"
    assert harness.signature_order(csrc, "s000_fp64", "c") == ["a", "out", "N"]
    # a long Fortran arg list wraps with `&` continuations; they must be stripped, not become arg names.
    ftn = "subroutine s1115_fp64(aa, &\n  & bb_slice, cc, &\n  & LEN_2D) bind(c, name='s1115_fp64')\n"
    assert harness.signature_order(ftn, "s1115_fp64", "fortran") == ["aa", "bb_slice", "cc", "LEN_2D"]


def test_abi_order_pointer_star_stripped():
    assert harness.signature_order("void k_fp64(double *a, double* b, int64_t N) {", "k_fp64") == ["a", "b", "N"]


# --- flag composition (flags.*) -----------------------------------------------------------------------
def test_base_flags_native_tuning_per_family():
    assert flags.base_flags("gnu") == ["-O3", "-march=native", "-fPIC", "-shared"]
    assert flags.base_flags("nvidia")[1] == "-tp=native"  # nvc uses -tp=native, not -march=native


def test_fortran_fp_flags_strip_unsupported_and_add_gfortran_guards():
    # gfortran rejects -fexcess-precision=standard and -fno-math-errno; they must be dropped.
    strict_f = flags.fp_flags("gnu", "strict-ieee", "fortran")
    assert "-fexcess-precision=standard" not in strict_f and "-fno-math-errno" not in strict_f
    assert "-fno-frontend-optimize" in strict_f  # gfortran reassociates at -O without this
    assert "-fno-protect-parens" in flags.fp_flags("gnu", "fast-math", "fortran")  # only at the fast rung
    # the C spelling keeps the flags the Fortran frontend rejects.
    assert "-fexcess-precision=standard" in flags.fp_flags("gnu", "strict-ieee", "c")


def test_cost_flags_no_vec_and_cheap_collapse():
    assert flags.cost_flags("gnu", "no-vec") == ["-fno-tree-vectorize"]
    assert flags.cost_flags("llvm", "no-vec") == ["-fno-vectorize", "-fno-slp-vectorize"]
    assert flags.cost_flags("nvidia", "no-vec") == ["-Mnovect"]
    assert flags.cost_flags("gnu", "cheap") == ["-fvect-cost-model=cheap"]
    assert flags.cost_flags("llvm", "cheap") == []  # clang has no cheap knob -> collapses to default
    assert flags.cost_flags("gnu", "default") == []


def test_flag_matrix_atol_covers_every_level():
    # every emitted level has a validation tolerance, and strict is the tightest.
    assert set(flags.FP_ATOL) == set(flags.FP_LEVELS)
    assert flags.FP_ATOL["strict-ieee"] < flags.FP_ATOL["fast-math"]
    for level, model, cflags in flags.flag_matrix("gnu"):
        assert cflags[:1] == ["-O3"] and level in flags.FP_LEVELS and model in flags.COST_MODELS


def test_veclib_flags_compose_and_gate_by_compatibility():
    # 'none'/empty -> no flags; incompatible (svml on gcc) or missing compiler -> rejected with a reason.
    # -L/-rpath is machine-dependent, so assert membership, not exact lists.
    assert flags.veclib_flags("g++", "none") == ([], None)
    assert flags.veclib_flags("clang++", None) == ([], None)
    fl, r = flags.veclib_flags("clang++", "sleef")  # x86: emit via libmvec token, link libsleefgnuabi
    assert r is None and "-fveclib=libmvec" in fl and any("-lsleefgnuabi" in a for a in fl)
    flg, rg = flags.veclib_flags("g++", "libmvec")  # glibc: no compile flag, -lmvec pinned at link
    assert rg is None and any("-lmvec" in a for a in flg) and not any("-fveclib" in a for a in flg)
    bad, reason = flags.veclib_flags("g++", "svml")  # gcc emits _ZGV*, never __svml_* -> unusable
    assert bad is None and "incompatible" in reason
    nocc, reason2 = flags.veclib_flags(None, "sleef")
    assert nocc is None and "without a compiler" in reason2
    assert set(flags.VECLIBS) == {"none", "sleef", "libmvec", "svml"}


def test_lane_flags_threads_veclib_and_rejects_incompatible():
    ok, r = flags.lane_flags("llvm", "default-fp", "default", "c", compiler="clang++", veclib="sleef")
    assert r is None and "-fveclib=libmvec" in ok and any("-lsleefgnuabi" in a for a in ok)
    bad, reason = flags.lane_flags("gnu", "default-fp", "default", "c", compiler="g++", veclib="svml")
    assert bad is None and "incompatible" in reason  # unsupported cell recorded, never silently emitted


def toolchain_labelled(label, cc):
    return Toolchain(name=label, cc=cc, cxx=None, version=(0, 0), source="path")


def test_toolchain_fp_family_maps_labels_to_fp_families():
    assert toolchain_labelled("gcc", "gcc").fp_family == "gnu"
    assert toolchain_labelled("clang", "clang").fp_family == "llvm"
    assert toolchain_labelled("nvhpc", "nvc").fp_family == "nvidia"
    assert toolchain_labelled("intel", "icx").fp_family == "intel"
    assert toolchain_labelled("unknown", "some-cc").fp_family == "gnu"  # safe default


# --- key_seed determinism -----------------------------------------------------------------------------


# --- fault isolation edge cases (run_isolated) --------------------------------------------------------
def test_run_isolated_malformed_result_is_error_not_crash():
    # a non-JSON-able return is caught in the child and comes back as an error sentinel; parent survives.
    res = run_isolated(lambda: {"bad": {1, 2, 3}})  # a set is not JSON-serializable
    assert "error" in res and "TypeError" in res["error"]


def test_run_isolated_passes_through_plain_dict():
    assert run_isolated(lambda: {"ok": True, "n": 7}) == {"ok": True, "n": 7}


# --- call_c output snapshotting (harness.call_c) -------------------------------------------------------
class CountingArray(np.ndarray):
    """An ndarray that counts its own .copy() calls, so a test can assert call_c did not snapshot."""
    copies = 0

    def copy(self, *a, **kw):
        self.copies += 1
        return super().copy(*a, **kw)


class FakeBoundary:
    """Both halves of :class:`nestforge.extract.Boundary` that call_c reads. ``inputs`` is not optional
    padding: their INTERSECTION with ``outputs`` is what call_c restores between timed reps, so a fixture
    carrying only ``outputs`` cannot express an in-place kernel at all."""

    def __init__(self, outputs, inputs=()):
        self.outputs = outputs
        self.inputs = list(inputs)


class FakeKernel:
    """Stands in for the ctypes entry: records calls, and accepts the argtypes/restype the binder sets."""

    def __init__(self):
        self.calls = 0
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        self.calls += 1


def call_c_on_stub(monkeypatch, reps, read_write=False, **kw):
    """Drive harness.call_c against a stubbed .so -- the ABI marshalling is real, only the compiled entry
    is faked, so no compiler/toolchain is needed. ``read_write`` marks ``a`` as an in-place buffer (read AND
    written), the case whose per-rep restore decides what the timing measures."""
    fn = FakeKernel()
    monkeypatch.setattr(harness.ctypes, "CDLL", lambda path: {"k_fp64": fn})
    buf = np.zeros(4, dtype=np.float64).view(CountingArray)
    inputs = {"a": buf}
    argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int64]
    boundary = FakeBoundary(["a"], inputs=["a"] if read_write else [])
    out, us = harness.call_c(Path("stub.so"), "k_fp64", ["a", "LEN_1D"], argtypes, boundary, inputs, {"LEN_1D": 4},
                             reps, **kw)
    return out, us, buf, fn


def test_call_c_skips_the_output_snapshot_when_not_requested(monkeypatch):
    # The timing path discards the outputs; at XL one output is GBs, so the snapshot must not be built.
    out, _, buf, fn = call_c_on_stub(monkeypatch, reps=3, copy_outputs=False)
    assert out is None
    assert buf.copies == 0
    assert fn.calls == 5  # correctness + warm + reps


def test_call_c_snapshots_outputs_by_default(monkeypatch):
    # The validate path still needs the post-correctness-run values, snapshotted before timing mutates them.
    out, _, buf, _ = call_c_on_stub(monkeypatch, reps=1)
    assert set(out) == {"a"} and buf.copies == 1


def test_call_c_restores_an_in_place_buffer_before_every_timed_rep(monkeypatch):
    """An array that is both read and written must start each timed rep from the same values. Without the
    restore an in-place kernel times ``a * b**k`` -- denormal arithmetic by a handful of reps -- and the
    ranking E1 reads off granularity rungs becomes "which candidate decayed slower"."""
    restored = []
    fn = FakeKernel()
    monkeypatch.setattr(harness.ctypes, "CDLL", lambda path: {"k_fp64": fn})
    buf = np.zeros(4, dtype=np.float64)

    class Recorder(np.ndarray):
        """Records what the restore writes back, so the assertion is on VALUES, not on a copy count."""

        def __setitem__(self, key, value):
            restored.append(np.asarray(value).copy())
            super().__setitem__(key, value)

    inputs = {"a": buf.view(Recorder)}
    inputs["a"][...] = 0.25  # the pristine values the restore must reinstate
    restored.clear()
    harness.call_c(Path("stub.so"),
                   "k_fp64", ["a", "LEN_1D"], [ctypes.POINTER(ctypes.c_double), ctypes.c_int64],
                   FakeBoundary(["a"], inputs=["a"]),
                   inputs, {"LEN_1D": 4},
                   reps=3)
    # 4 = one per rep, plus one before the WARM call so the call the CPU trains its caches and predictors
    # on starts from the same state the timed reps do.
    assert len(restored) == 4, f"expected one restore per rep plus the warm call, saw {len(restored)}"
    for values in restored:
        np.testing.assert_array_equal(values, np.full(4, 0.25))


def test_call_c_does_not_restore_a_write_only_buffer(monkeypatch):
    """A fully-overwritten output cannot accumulate, so it must NOT be snapshotted: at the profiling preset
    a blanket copy of every output doubles the forked child's peak RSS for nothing."""
    _, _, buf, _ = call_c_on_stub(monkeypatch, reps=3, copy_outputs=False)
    assert buf.copies == 0


# --- the rewind primitive, shared by every timing loop in the repo ------------------------------------
def test_accumulating_outputs_is_the_read_write_intersection():
    """Read-only and write-only buffers are both excluded: the first is never written, the second holds
    nothing that survives into the next rep. Only the intersection can feed on its own output."""
    buffers = {n: np.zeros(2) for n in ("rw", "wo", "ro")}
    boundary = FakeBoundary(["rw", "wo"], inputs=["rw", "ro"])
    assert arena.accumulating_outputs(boundary, buffers) == ["rw"]
    # A buffer the caller never allocated cannot be rewound; naming it must not KeyError mid-loop.
    assert arena.accumulating_outputs(FakeBoundary(["absent"], inputs=["absent"]), buffers) == []


def test_rewind_restores_values_taken_before_the_kernel_ran():
    a = np.full(4, 0.25)
    snapshot = arena.rewind_snapshot(FakeBoundary(["a"], inputs=["a"]), {"a": a})
    a *= 0.5  # stand in for an in-place kernel consuming its own output
    arena.rewind(snapshot)
    np.testing.assert_array_equal(a, np.full(4, 0.25))


def test_rewind_snapshot_writes_through_to_the_bound_buffer():
    """The pairs must hold the LIVE array, not a copy of it: every timing path binds its ctypes pointers
    once, before the rep loop, so a rewind into a detached array would restore nothing the kernel reads."""
    a = np.full(4, 0.25)
    snapshot = arena.rewind_snapshot(FakeBoundary(["a"], inputs=["a"]), {"a": a})
    assert snapshot[0][0] is a


def test_toolchain_fp_family_only_ever_names_a_real_fp_family():
    """`Toolchain.fp_family` feeds `flags.lane_flags`, which indexes the FP tables by family. A toolchain it
    maps to a family those tables do not have would decline every cell -- or worse, `base_flags` would
    silently fall back to `-march=native` and the cell would be measured under flags nobody chose."""
    for label, cc in (("gcc", "gcc"), ("clang", "clang"), ("nvhpc", "nvc"), ("intel", "icx"), ("future", "fcc")):
        assert toolchain_labelled(label, cc).fp_family in flags._FP, label
        assert toolchain_labelled(label, cc).fp_family in flags._REDUCED_FP, label


def test_the_two_family_vocabularies_stay_apart():
    """toolchain.compiler_family classifies an EXECUTABLE for its OpenMP ABI; Toolchain.fp_family classifies a
    toolchain for the FP tables. They are not interchangeable, and this pins the exact disagreement that makes
    that true, so a future 'simplification' that collapses them fails here instead of in a sweep."""
    from nestforge.build.toolchain import compiler_family

    assert compiler_family("icc") == "intel-classic" and compiler_family("icc") not in flags._FP
    assert compiler_family("icx") == "llvm"  # an Intel compiler classified llvm: the ABI, not the FP family
    assert toolchain_labelled("intel", "icx").fp_family == "intel"
