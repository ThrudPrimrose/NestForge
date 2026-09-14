# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Compile-free unit tests for the perf/arena plumbing: signature parsing and FP-precision x cost-model
flag composition -- pure logic on synthetic inputs, so no compiler needed.
"""
import ctypes
import shutil
from pathlib import Path

import numpy as np
import pytest

from nestforge import arena, tsvc
from nestforge.isolation import run_isolated
from nestforge.perf import flags, harness, tsvc_full


# --- native-baseline signature parsing (tsvc.native_signature) ----------------------------------------
def test_native_signature_strips_qualifiers_and_types():
    cpp = 'extern "C" void s000_d(double* restrict a, const double * b, int64_t LEN_1D, int n1) {'
    sig = tsvc.native_signature(cpp, "s000_d")
    assert sig == [("a", "double", True), ("b", "double", True), ("LEN_1D", "int64_t", False), ("n1", "int", False)]


def test_native_signature_float_and_missing_symbol():
    assert tsvc.native_signature("void k(float* x) {", "k") == [("x", "float", True)]
    with pytest.raises(LookupError):
        tsvc.native_signature("void other(double* x) {", "k")


def test_native_symbol_fallback_to_first_kernel():
    # The convention symbol is used when present; otherwise the first `void <name>(` is taken.
    assert harness.native_symbol("void s000_d(double* a)", "s000_d") == "s000_d"
    assert harness.native_symbol("void renamed_kernel(double* a)", "s000_d") == "renamed_kernel"
    with pytest.raises(LookupError):
        harness.native_symbol("int not_a_kernel;", "s000_d")


# --- emitted-source signature order (harness.signature_order) ------------------------------------------
def test_signature_order_c_and_fortran_multiline():
    csrc = "void s000_fp64(double* a, double* out, int64_t N) {"
    assert harness.signature_order(csrc, "s000_fp64", "c") == ["a", "out", "N"]
    # a long Fortran arg list wraps with `&` continuations; they must be stripped, not become arg names.
    ftn = "subroutine s1115_fp64(aa, &\n  & bb_slice, cc, &\n  & LEN_2D) bind(c, name='s1115_fp64')\n"
    assert harness.signature_order(ftn, "s1115_fp64", "fortran") == ["aa", "bb_slice", "cc", "LEN_2D"]


def test_fortran_unmunge_multiple_and_no_underscore():
    # a leading `_` munges to `x`; a non-underscore name is unchanged; both reverse cleanly.
    order = ["x_a", "xb", "LEN_1D"]
    names = ["__a", "xb", "LEN_1D"]
    assert harness.fortran_unmunge(order, names) == ["__a", "xb", "LEN_1D"]


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
    ok, r = flags.lane_flags("llvm", "default-fp", "default", "sequential", "c", 4, compiler="clang++", veclib="sleef")
    assert r is None and "-fveclib=libmvec" in ok and any("-lsleefgnuabi" in a for a in ok)
    bad, reason = flags.lane_flags("gnu", "default-fp", "default", "sequential", "c", 4, compiler="g++", veclib="svml")
    assert bad is None and "incompatible" in reason  # unsupported cell recorded, never silently emitted


def test_source_has_math_gates_the_veclib_axis():
    from nestforge.perf import tsvc_full
    assert tsvc_full.source_has_math("y[i] = sin(x[i]) + 1.0;")
    assert tsvc_full.source_has_math("z = pow(a, b);")
    assert not tsvc_full.source_has_math("c[i] = a[i] + b[i] * 2.0;")  # arithmetic-only nest -> no veclib


def test_veclibs_for_gates_on_math_and_compatibility():
    from nestforge.perf import tsvc_full
    # veclibs_for takes the PRECOMPUTED per-lang has_math flag (source scanned once at ctx-build time).
    assert tsvc_full.veclibs_for(True, ("none", "libmvec"), "gcc") == ("none", "libmvec")  # math + compatible
    assert tsvc_full.veclibs_for(True, ("none", "sleef"), "g++") == ("none", "sleef")  # gcc DOES sleef (gnuabi)
    assert tsvc_full.veclibs_for(True, ("none", "svml"), "g++") == ("none", )  # svml incompatible w/ gcc
    assert tsvc_full.veclibs_for(False, ("none", "libmvec"), "gcc") == ("none", )  # no math -> none only


def test_resolve_veclibs_spec_and_auto():
    from nestforge.perf import tsvc_full
    assert tsvc_full.resolve_veclibs(["none"]) == ("none", )
    assert tsvc_full.resolve_veclibs(["libmvec"]) == ("none", "libmvec")  # 'none' ensured present
    assert tsvc_full.resolve_veclibs(["sleef", "libmvec"])[0] == "none"
    auto = tsvc_full.resolve_veclibs(["auto"])  # none + characterized winner, or none if nothing installed
    assert auto[0] == "none" and 1 <= len(auto) <= 2


@pytest.mark.skipif(shutil.which("gcc") is None, reason="gcc not on PATH")
def test_enumerate_cells_gates_veclib_cells_by_nest_math(tmp_path):
    """The veclib axis fans lane-3 cells off the PRECOMPUTED per-lang ``has_math`` flag: a math nest gets
    both none and libmvec timing cells, a plain-arithmetic nest gets none only. Dummy paths -- no source I/O."""
    from nestforge.perf import tsvc_full
    from nestforge.toolchain import discover_toolchains
    tcs = discover_toolchains("gcc")
    axes = {
        "opt_mode": "simplify-parallel",
        "parallelism": ["sequential"],
        "cost_models": ["default"],
        "fp_modes": ["default-fp"],
        "gate": False,
        "matrix_preset": "lean",
        "veclibs": ("none", "libmvec")
    }
    pend, _ = tsvc_full.enumerate_cells(
        {
            "lang_src": {
                "c": (Path("m_fp64.c"), ["a", "b"], [None, None])
            },
            "has_math": {
                "c": True
            },
            "symbol": "m_fp64",
            "nest_idx": 0
        }, tcs, {}, axes, 4, flags.CXX_STD, tmp_path)
    assert {p.cell.veclib for p in pend if p.cell.role == "timing"} == {"none", "libmvec"}
    pend2, _ = tsvc_full.enumerate_cells(
        {
            "lang_src": {
                "c": (Path("p_fp64.c"), ["a", "b"], [None, None])
            },
            "has_math": {
                "c": False
            },
            "symbol": "p_fp64",
            "nest_idx": 0
        }, tcs, {}, axes, 4, flags.CXX_STD, tmp_path)
    assert {p.cell.veclib for p in pend2 if p.cell.role == "timing"} == {"none"}


def test_family_of_maps_labels_to_fp_families():
    assert harness.family_of("gcc") == "gnu"
    assert harness.family_of("clang") == "llvm"
    assert harness.family_of("nvhpc") == "nvidia"
    assert harness.family_of("intel") == "intel"
    assert harness.family_of("unknown") == "gnu"  # safe default


# --- key_seed determinism -----------------------------------------------------------------------------
def test_key_seed_is_stable_and_distinct():
    assert tsvc.key_seed("s000") == tsvc.key_seed("s000")  # process-independent (not salted hash)
    assert tsvc.key_seed("s000") != tsvc.key_seed("s112")  # different keys -> different offsets
    assert 0 <= tsvc.key_seed("anything") <= 0xFFFF


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


def test_collect_samples_restores_the_snapshot_before_every_rep():
    """tsvc_full's nest, native and DaCe-cpp lanes all time through collect_samples. Passing no snapshot
    let each of them measure a decaying buffer -- and the lanes are divided by each other, so the bias did
    not cancel."""
    a = np.full(4, 0.25)
    boundary = FakeBoundary(["a"], inputs=["a"])
    snapshot = arena.rewind_snapshot(boundary, {"a": a})
    seen = []

    def fn(*_args):
        seen.append(a.copy())  # what THIS call was handed
        a[...] *= 0.25  # in-place decay, the shape of the bug

    tsvc_full.collect_samples(fn, (), reps=3, snapshot=snapshot)
    assert len(seen) == 4, "warm call plus one per rep"
    for values in seen:
        np.testing.assert_array_equal(values, np.full(4, 0.25))


def test_collect_samples_without_a_snapshot_still_times_the_reps():
    """A nest that writes nothing it reads needs no rewind; the default must not become mandatory
    plumbing for those lanes."""
    calls = []
    tsvc_full.collect_samples(lambda *_: calls.append(1), (), reps=3)
    assert len(calls) == 4 and len(tsvc_full.collect_samples(lambda *_: None, (), reps=3)) == 3


def test_the_native_signature_type_set_matches_what_the_arena_can_bind():
    """`native_signature` produces base-type STRINGS that `harness.C_BASE` turns into ctypes types. A type
    accepted by the parser but absent from that mapping would KeyError mid-bind, and one accepted by the
    mapping but refused by the parser makes a legitimate baseline unmeasurable. Pin them equal."""
    assert set(tsvc.NATIVE_C_BASE) == set(harness.C_BASE)


def test_native_signature_refuses_a_type_it_cannot_bind():
    """The old fallback made every unrecognised declaration an `int`, so a `bool` bound as a 4-byte int:
    a silent ABI mismatch ctypes cannot catch and the timings cannot reveal."""
    with pytest.raises(ValueError, match="cannot bind"):
        tsvc.native_signature('extern "C" void k_d(bool flag, double* a) {', "k_d")


def test_native_signature_ignores_a_doc_comment_naming_the_symbol():
    """29 of the 245 foundation baselines put ``// <symbol> (<note>): ...`` directly above the declaration.
    A ``\\b<symbol>\\s*\\(`` search matches the COMMENT first and returns its parenthetical as the whole
    parameter list, so the parse raised and -- because both drivers catch only LookupError -- the ValueError
    killed the entire kernel, not just its native column. The ``void`` anchor is what rules the comment out."""
    cpp = ('// argmax_value_d (s314): x = a[0]; for i: if a[i] > x: x = a[i]\n'
           'extern "C" void argmax_value_d(const double* __restrict__ a, double* __restrict__ out, int64_t n) {')
    assert tsvc.native_signature(cpp, "argmax_value_d") == [("a", "double", True), ("out", "double", True),
                                                            ("n", "int64_t", False)]


def test_native_signature_accepts_a_namespace_qualified_type():
    """The C++ baselines spell the <cstdint> types both bare and ``std::``-qualified -- same type, same ABI.
    Unhandled, the qualified spelling failed NATIVE_C_BASE and dropped every gather/scatter kernel (whose
    index array is the one declared that way) from the sweep."""
    cpp = 'extern "C" void ext_gather_load_d(double* dst, const std::int64_t* __restrict__ idx, int len) {'
    assert tsvc.native_signature(cpp, "ext_gather_load_d") == [("dst", "double", True), ("idx", "int64_t", True),
                                                               ("len", "int", False)]


def test_every_foundation_baseline_signature_parses():
    """The parser's real input is the shipped corpus, and both defects above were invisible to hand-written
    fixtures. Parse all 245 for real: a kernel whose signature will not parse is one no sweep can measure."""
    from nestforge.perf.harness import native_symbol
    failed = {}
    kernels = tsvc.iter_tsvc_kernels(corpus="foundation")
    assert kernels, "the foundation corpus came back empty; this test would prove nothing"
    for kernel in kernels:
        text = kernel.native_cpp.read_text()
        try:
            tsvc.native_signature(text, native_symbol(text, kernel.native_symbol))
        except (LookupError, ValueError) as e:
            failed[kernel.key] = f"{type(e).__name__}: {e}"
    assert not failed, f"{len(failed)} of {len(kernels)} baselines do not parse: {sorted(failed)[:10]}"


def test_native_signature_does_not_eat_a_name_containing_const():
    """Qualifiers are stripped as whole words. A substring strip turned a parameter named `const_term`
    into `_term`, binding the argument list one name out of step with the compiled signature."""
    sig = tsvc.native_signature('extern "C" void k_d(const double* const_term, int64_t LEN_1D) {', "k_d")
    assert sig == [("const_term", "double", True), ("LEN_1D", "int64_t", False)]


def test_family_of_only_ever_names_a_real_fp_family():
    """`family_of` feeds `flags.fp_flags`/`base_flags`, which index the FP tables by family. A label it maps
    to a family those tables do not have would KeyError mid-sweep -- or worse, `base_flags` would silently
    fall back to `-march=native` and the cell would be measured under flags nobody chose."""
    labels = ["gcc", "clang", "nvhpc", "intel", "some-future-toolchain"]
    for label in labels:
        assert harness.family_of(label) in flags._FP, label
        assert harness.family_of(label) in flags._REDUCED_FP, label


def test_the_two_family_vocabularies_stay_apart():
    """toolchain.compiler_family classifies an EXECUTABLE for its OpenMP ABI; family_of classifies a toolchain
    LABEL for the FP tables. They are not interchangeable, and this pins the exact disagreement that makes
    that true, so a future 'simplification' that collapses them fails here instead of in a sweep."""
    from nestforge.toolchain import compiler_family

    assert compiler_family("icc") == "intel-classic" and compiler_family("icc") not in flags._FP
    assert compiler_family("icx") == "llvm"  # an Intel compiler classified llvm: the ABI, not the FP family
    assert harness.family_of("intel") == "intel"
