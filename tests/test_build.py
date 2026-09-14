# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""nest-forge owns the DaCe build (BUILD.md): generate DaCe's C++, compile+link it ourselves, call it via
ctypes with manual init/program/exit -- not ``dace.compile``.

Tests build real corpus nests through :mod:`nestforge.build.sdfg` and check the owned-built kernel matches the
numpy oracle: source-tree layout, the init/program/exit call sequence, and per-parameter ctype marshaling.
"""

import ctypes.util
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import warnings
from pathlib import Path

import numpy as np
import pytest

import dace

import nestforge.build.sdfg as build_mod
import nestforge.build.toolchain as toolchain_mod
from nestforge.build.toolchain import LIBOMP, runtime_library

assert shutil.which("g++") is not None, "g++ not on PATH (setup_apt.sh installs it)"

from nestforge.corpus.bench import iter_dace_kernels
from nestforge.phases.scopes import parallel_top_level_maps
from nestforge.ir.extract import extract_nest_to_sdfg
from nestforge.corpus.translate import prepare
from nestforge.build.arena import make_inputs, run_oracle
from nestforge.build.sdfg import BuildOptions, build_sdfg, dace_runtime_include
from nestforge.build.toolchain import (
    OpenMPRuntime,
    compiler_family,
    driver_lib_path,
    driver_search_dirs,
    hint_dirs,
    ldconfig_dirs,
    linkable_lib_dir,
    llvm_version,
    parse_params,
    runtime_installed,
)


def kernels():
    return {k.short_name: k for k in iter_dace_kernels()}


def first_nest(short):
    sdfg = kernels()[short].to_sdfg(simplify=True)
    parent, node = parallel_top_level_maps(sdfg)[0]
    return extract_nest_to_sdfg(parent, node, name="nest")


def owned_build_matches_oracle(short, size=48, opts=None):
    boundary = first_nest(short)
    shape_syms = {
        s for s in boundary.symbols if any(s in str(d.shape) for d in boundary.standalone_sdfg.arrays.values())
    }
    sizes = {s: (size if s in shape_syms else 0) for s in boundary.symbols}
    inputs = make_inputs(boundary, sizes, seed=0)
    prep = prepare(boundary, "k", Path(tempfile.mkdtemp()))
    oracle = run_oracle(prep, boundary, inputs, sizes)

    built = build_sdfg(boundary.standalone_sdfg, Path(tempfile.mkdtemp(prefix="nf_build_")), opts)
    buf = {k: v.copy() for k, v in inputs.items()}
    built.run(buf, sizes)  # init -> program -> exit
    for o in oracle:
        np.testing.assert_allclose(buf[o], oracle[o], rtol=1e-9, atol=1e-9, equal_nan=True)
    return built


def test_dace_runtime_include_exists():
    assert (dace_runtime_include() / "dace" / "dace.h").exists()


def test_owned_build_gemm_matches_oracle():
    """gemm: int64_t size symbols + a Scalar (alpha/beta) passed by value through the owned build."""
    owned_build_matches_oracle("scientific_computing/dense_linear_algebra/gemm/gemm")


def test_owned_build_jacobi_matches_oracle():
    """jacobi_1d: an ``int`` (not int64_t) size symbol -- guards the per-parameter ctype marshaling."""
    owned_build_matches_oracle("scientific_computing/structured_grids/jacobi_1d/jacobi_1d")


def without_search_paths(flags):
    """``flags`` minus any discovery path -- ``-L`` AND the matching ``-Wl,-rpath,``: WHICH runtime is
    selected, not WHERE it was found (the latter is host-dependent -- e.g. Ubuntu moves libomp-18-dev under
    /usr/lib/llvm-18/lib, which makes ``link_flags`` emit BOTH). Filter rather than index since the
    position varies by family.
    """
    return [f for f in flags if not f.startswith("-L") and not f.startswith("-Wl,-rpath")]


def test_library_dirs_come_from_the_toolchain_not_from_hardcoded_layouts():
    """Where a runtime lives is ASKED (driver ``-print-search-dirs``, then the loader cache), because a
    hardcoded distro ladder goes stale per distro and per toolchain version. Guessed layouts survive only
    as a last-resort hint, after every query."""
    cc = "g++" if shutil.which("g++") else "gcc"
    dirs = driver_search_dirs(cc)
    assert dirs and all(os.path.isabs(d) for d in dirs), dirs
    assert driver_search_dirs("no-such-compiler-42") == []  # a missing driver is empty, never a crash
    # libc is in the loader cache on every Linux box, so this exercises the parse without pinning a path.
    # Assert NON-EMPTY: `all()` over [] passes, which would green-light a layer that found nothing at all
    # (e.g. ldconfig unreachable because /usr/sbin is off PATH -- the exact failure this must catch).
    libc_dirs = ldconfig_dirs("c")
    assert libc_dirs and all(os.path.isabs(d) for d in libc_dirs), libc_dirs
    assert all(os.path.isabs(d) for d in hint_dirs())
    # A library nothing provides must resolve to None -- no layer may invent a directory for it.
    assert linkable_lib_dir("nosuchlib42", cc) is None


def test_llvm_version_parses_the_number_not_the_string():
    assert llvm_version(Path("/usr/lib/llvm-21/lib")) == (21,)
    assert llvm_version(Path("/usr/lib/llvm-9/lib")) == (9,)
    assert llvm_version(Path("/usr/lib/llvm-18.1/lib")) == (18, 1)  # point release ranks ABOVE bare 18
    assert llvm_version(Path("/usr/lib/llvm-18/lib")) < llvm_version(Path("/usr/lib/llvm-18.1/lib"))
    assert llvm_version(Path("/usr/lib/x86_64-linux-gnu")) == (-1,)  # not an llvm-N dir at all


def test_hint_dirs_rank_by_version_across_all_roots(tmp_path, monkeypatch):
    """The ranking must be GLOBAL, not per root.

    Two traps this pins, both of which shipped: sorting the glob as strings puts llvm-9 above llvm-21, and
    sorting within each root then concatenating puts /usr/lib's llvm-14 above /usr/lib64's llvm-18 -- the
    normal mixed-install layout. Real directories on disk, because the previous version of this test called
    hint_dirs() on the host and passed identically with the buggy sort: the box simply had no single-digit
    LLVM, so the assertion never discriminated.
    """
    lib, lib64 = tmp_path / "lib", tmp_path / "lib64"
    for root, versions in ((lib, ("llvm-9", "llvm-21")), (lib64, ("llvm-14", "llvm-18"))):
        for v in versions:
            (root / v / "lib").mkdir(parents=True)
    monkeypatch.setattr(toolchain_mod, "LIB_DIR_HINT_ROOTS", (str(lib), str(lib64)))
    monkeypatch.setattr(toolchain_mod, "LIB_DIR_HINTS", ())

    hints = hint_dirs()
    assert [Path(d).parent.name for d in hints] == ["llvm-21", "llvm-18", "llvm-14", "llvm-9"]
    assert len(hints) == len(set(hints)), hints  # and no duplicates across roots


def test_hint_dirs_is_a_total_order_so_two_identical_boxes_agree(tmp_path, monkeypatch):
    """Equal versions must still order deterministically. Path.glob returns raw directory order, which
    varies with inode layout, so a tie left to glob order resolves libomp differently on machines built
    from the same image."""
    root = tmp_path / "lib"
    for name in ("llvm-18", "llvm-18.1"):
        (root / name / "lib").mkdir(parents=True)
    (root / "llvm-18" / "lib64").mkdir()  # same version, two dirs -> the tiebreaker has to decide
    monkeypatch.setattr(toolchain_mod, "LIB_DIR_HINT_ROOTS", (str(root),))
    monkeypatch.setattr(toolchain_mod, "LIB_DIR_HINTS", ())

    assert hint_dirs() == hint_dirs()  # stable across calls
    assert hint_dirs()[0].endswith("llvm-18.1/lib")  # newest first, ties broken by path


def test_ldconfig_candidates_are_version_ranked_before_first_match(monkeypatch):
    """The loader cache lists dirs in ITS order, and linkable_lib_dir returns the FIRST hit -- so without
    ranking, the version fix in hint_dirs is unreachable whenever ldconfig knows any llvm dir at all."""
    monkeypatch.setattr(toolchain_mod, "linker_finds", lambda soname, compiler: False)
    monkeypatch.setattr(toolchain_mod, "env_library_dirs", lambda: [])
    monkeypatch.setattr(toolchain_mod, "driver_lib_path", lambda soname, compiler: None)
    monkeypatch.setattr(toolchain_mod, "driver_search_dirs", lambda compiler: [])
    monkeypatch.setattr(toolchain_mod, "llvm_config_libdir", lambda: None)
    # cache order is deliberately oldest-first, the order that used to win
    monkeypatch.setattr(toolchain_mod, "ldconfig_dirs", lambda soname: ["/opt/llvm-14/lib", "/opt/llvm-18/lib"])
    monkeypatch.setattr(toolchain_mod.Path, "exists", lambda self: "llvm-" in str(self))

    linkable_lib_dir.cache_clear()
    assert linkable_lib_dir("omp", "g++") == "/opt/llvm-18/lib"
    linkable_lib_dir.cache_clear()


def test_openmp_runtime_is_a_separate_per_compiler_flag_axis():
    """The OpenMP runtime maps to the right flag PER COMPILER, so mixed-compiler builds share ONE runtime."""
    rt = OpenMPRuntime()  # default libomp
    assert compiler_family("gfortran") == "gnu" and compiler_family("flang") == "llvm"
    assert compiler_family("icx") == "llvm"
    # LLVM family selects the runtime BY NAME (flang -fopenmp=libomp).
    assert rt.compile_flags("flang") == ["-fopenmp=libomp"]
    assert rt.compile_flags("clang++") == ["-fopenmp=libomp"]
    assert rt.compile_flags("icx") == ["-fopenmp=libomp"]
    # gnu emits GOMP calls at compile and links the mandated runtime explicitly (not -fopenmp -> libgomp).
    assert rt.compile_flags("g++") == ["-fopenmp"] and without_search_paths(rt.link_flags("g++")) == ["-lomp"]
    # a lib_dir threads onto the link line as a -L/-rpath PAIR (so the .so is found at run time too),
    # and both are discovery, not selection -- without_search_paths must drop both.
    pinned = OpenMPRuntime(lib_dir="/opt/omp/lib").link_flags("g++")
    assert "-L/opt/omp/lib" in pinned and "-Wl,-rpath,/opt/omp/lib" in pinned
    assert without_search_paths(pinned) == ["-lomp"]


def test_openmp_runtime_registry_covers_the_popular_runtimes():
    """The three popular runtimes are ready knobs: libgomp (GNU), libomp (LLVM), libiomp5 (Intel,
    ABI-compat with libomp)."""
    from nestforge.build.toolchain import LIBGOMP, LIBIOMP5, OPENMP_RUNTIMES

    assert set(OPENMP_RUNTIMES) == {"libomp", "libgomp", "libiomp5"}
    # gcc on Intel's runtime (GOMP-compat); search paths filtered (see without_search_paths).
    assert without_search_paths(LIBIOMP5.link_flags("g++")) == ["-liomp5"]
    assert without_search_paths(LIBGOMP.link_flags("g++")) == ["-lgomp"]


def test_openmp_abi_compatibility_is_enforced():
    """A runtime is usable only if the compiler can actually LINK it, which depends on HOW the family
    selects a runtime, not ABI alone: gcc links any gomp-capable runtime by soname; LLVM name-selects
    only libomp/libiomp5 (kmpc ABI). Mismatches raise."""
    from nestforge.build.toolchain import LIBGOMP, LIBIOMP5, LIBOMP

    # clang name-selects libomp/libiomp5 but NOT libgomp (no __kmpc_*).
    assert LIBOMP.compatible("clang++") and LIBIOMP5.compatible("clang++")
    assert not LIBGOMP.compatible("clang++")
    with pytest.raises(ValueError, match="kmpc"):  # clang + libgomp: wrong ABI
        LIBGOMP.link_flags("clang++")
    # gcc (GOMP) works against every runtime, since libomp/libiomp5 carry a GOMP-compat layer.
    for rt in (LIBOMP, LIBGOMP, LIBIOMP5):
        assert rt.compatible("g++")


def test_gcc_compiled_kernel_links_against_libomp():
    """A g++-compiled kernel (GOMP_* calls under -fopenmp) links + runs against libomp via its GOMP-compat
    ABI -- proof a GCC node library can share the same libomp a clang/flang node library uses."""
    assert ctypes.util.find_library("omp") is not None, "libomp not installed (setup_apt.sh: libomp-dev)"
    boundary = first_nest("scientific_computing/dense_linear_algebra/gemm/gemm")
    shape_syms = {
        s for s in boundary.symbols if any(s in str(d.shape) for d in boundary.standalone_sdfg.arrays.values())
    }
    sizes = {s: (32 if s in shape_syms else 0) for s in boundary.symbols}
    inputs = make_inputs(boundary, sizes, seed=0)
    prep = prepare(boundary, "k", Path(tempfile.mkdtemp()))
    oracle = run_oracle(prep, boundary, inputs, sizes)
    built = build_sdfg(
        boundary.standalone_sdfg,
        Path(tempfile.mkdtemp(prefix="nf_omp_")),
        BuildOptions(compiler="g++", openmp=OpenMPRuntime()),
    )  # gcc object on libomp
    buf = {k: v.copy() for k, v in inputs.items()}
    built.run(buf, sizes)
    for o in oracle:
        np.testing.assert_allclose(buf[o], oracle[o], rtol=1e-9, atol=1e-9, equal_nan=True)


def parallel_axpy_sdfg(name="paxpy"):
    """A minimal SDFG with ONE genuinely parallel map (``CPU_Multicore`` -> ``#pragma omp parallel for``):
    ``Z[i] = X[i] + Y[i]``. Hermetic, so the OpenMP link matrix tests a guaranteed-parallel loop."""
    N = dace.symbol("N", dace.int64)
    sdfg = dace.SDFG(name)
    for a in ("X", "Y", "Z"):
        sdfg.add_array(a, [N], dace.float64)
    st = sdfg.add_state()
    me, mx = st.add_map("m", {"i": "0:N"}, schedule=dace.ScheduleType.CPU_Multicore)
    t = st.add_tasklet("t", {"x", "y"}, {"z"}, "z = x + y")
    st.add_memlet_path(st.add_read("X"), me, t, dst_conn="x", memlet=dace.Memlet("X[i]"))
    st.add_memlet_path(st.add_read("Y"), me, t, dst_conn="y", memlet=dace.Memlet("Y[i]"))
    st.add_memlet_path(t, mx, st.add_write("Z"), src_conn="z", memlet=dace.Memlet("Z[i]"))
    return sdfg


def test_link_flags_pins_a_runtime_that_is_off_the_default_linker_path(tmp_path, monkeypatch):
    """REGRESSION: the linker and the loader don't search the same places, so "installed" doesn't imply
    "-l<soname> resolves" -- e.g. Ubuntu's libomp-dev package moves the lib off the default linker path
    across releases. Pinning the apt package can't fix that; finding the file can.
    """
    (tmp_path / "libfakeomp.so").write_bytes(b"")  # a linkable lib, deliberately off the default path
    # LD_LIBRARY_PATH (not LIBRARY_PATH): the LOADER searches it, the LINKER does not -- exactly where a
    # spack/module runtime lives. LIBRARY_PATH would prove nothing (the linker already searches that).
    monkeypatch.setenv("LD_LIBRARY_PATH", str(tmp_path))
    toolchain_mod.linkable_lib_dir.cache_clear()
    rt = OpenMPRuntime(name="libfakeomp", soname="fakeomp")
    assert not toolchain_mod.linker_finds("fakeomp", "g++"), "premise: the linker cannot find it unaided"
    assert f"-L{tmp_path}" in rt.link_flags("g++")
    assert toolchain_mod.lib_linkable("fakeomp", "g++")  # and the honest probe agrees it can be linked
    toolchain_mod.linkable_lib_dir.cache_clear()


def test_link_flags_add_no_search_path_when_the_linker_already_finds_the_runtime(monkeypatch):
    # Discovery must stay invisible when the lib is already on the default path. Forced rather than read
    # off this box, so the assertion means the same thing wherever it runs.
    monkeypatch.setattr(toolchain_mod, "linker_finds", lambda *a, **kw: True)
    toolchain_mod.linkable_lib_dir.cache_clear()
    assert OpenMPRuntime().link_flags("g++") == ["-lomp"]
    toolchain_mod.linkable_lib_dir.cache_clear()


def test_driver_lib_path_normalises_the_answer_without_following_the_symlink(tmp_path):
    """``libomp.so`` IS a symlink (-> ``libomp.so.5``) and the two can live in different directories, so the
    answer needs normalising WITHOUT following it: ``resolve()`` would follow the symlink to a directory
    with no ``libomp.so``, so lexical normalisation is used instead.
    """
    link_dir, target_dir = tmp_path / "linkdir", tmp_path / "targetdir"
    link_dir.mkdir()
    target_dir.mkdir()
    (target_dir / "libsplit.so.5").write_bytes(b"")
    (link_dir / "libsplit.so").symlink_to(target_dir / "libsplit.so.5")  # the real distro layout

    fake_cc = tmp_path / "fake-cc"  # a driver that answers the way gcc does: full of ".." segments
    fake_cc.write_text(f'#!/bin/sh\necho "{link_dir}/../linkdir/libsplit.so"\n')
    fake_cc.chmod(0o755)

    got = driver_lib_path("split", str(fake_cc))
    assert got == link_dir / "libsplit.so", f"expected the symlink itself, got {got}"
    assert got.parent == link_dir, "the -L must be the symlink's own dir, never its target's"


def test_an_explicitly_pinned_lib_dir_beats_discovery(monkeypatch):
    # A spack/module runtime is pinned by hand and must win; "" means "I know: use a bare -l".
    monkeypatch.setattr(toolchain_mod, "linker_finds", lambda *a, **kw: False)
    toolchain_mod.linkable_lib_dir.cache_clear()
    assert "-L/opt/spack/omp" in OpenMPRuntime(lib_dir="/opt/spack/omp").link_flags("g++")
    assert OpenMPRuntime(lib_dir="").link_flags("g++") == ["-lomp"]
    toolchain_mod.linkable_lib_dir.cache_clear()


def test_parallel_map_emits_omp_pragma():
    """The sanity nest is actually parallel: DaCe lowers ``CPU_Multicore`` to an OpenMP pragma in the
    generated C++ (so the cross-compiler tests below really exercise the runtime link)."""
    from nestforge.build.sdfg import generate_program_folder

    frame, _ = generate_program_folder(parallel_axpy_sdfg(), Path(tempfile.mkdtemp(prefix="nf_omp_src_")))
    assert "#pragma omp parallel for" in frame.read_text()


# Each compiler builds the SAME parallel nest, linking the ONE mandated runtime (libomp) -- the
# mixed-compiler / single-runtime sanity matrix. icpx is a vendor compiler, only ever present in a
# vendor-configured environment (setup_apt.sh --oneapi).
@pytest.mark.parametrize(
    "compiler",
    [
        "g++",
        "clang++",
        pytest.param("icpx", marks=pytest.mark.vendor),  # vendor compiler: absent on the CI runner
    ],
)
def test_parallel_loop_links_openmp_across_compilers(compiler):
    assert shutil.which(compiler) is not None, f"{compiler} not on PATH"
    rt = LIBOMP
    assert runtime_installed(rt), f"{rt.name} not installed here (no OpenMP runtime on PATH/LD_LIBRARY_PATH/ldconfig)"
    assert rt.compatible(compiler), f"{compiler} must be able to link {rt.name}"
    n = 256
    x, y = np.random.default_rng(0).random(n), np.random.default_rng(1).random(n)
    buf = {"X": x.copy(), "Y": y.copy(), "Z": np.zeros(n)}
    built = build_sdfg(
        parallel_axpy_sdfg(),
        Path(tempfile.mkdtemp(prefix="nf_par_")),
        BuildOptions(compiler=compiler, flags=["-O2", "-fPIC", "-shared", "-std=c++20"], openmp=rt),
    )
    built.run(buf, {"N": n})
    np.testing.assert_allclose(buf["Z"], x + y, rtol=1e-12, atol=1e-12)


def test_build_tracks_optimization_and_compile_time():
    """Every owned build records both the codegen (optimization) time and the compile (toolchain) time."""
    built = build_sdfg(parallel_axpy_sdfg(), Path(tempfile.mkdtemp(prefix="nf_time_")))
    assert built.codegen_seconds > 0.0
    assert built.compile_seconds > 0.0


def test_unload_after_close_is_a_noop():
    """The documented lifecycle -- ``run()`` (init -> program -> close) then ``unload()`` once the sweep is
    done with a kernel -- must not raise: by the time ``unload()`` runs, ``close()`` has already dropped the
    handle, so there is nothing left for ``unload`` to reconcile."""
    built = build_sdfg(parallel_axpy_sdfg(), Path(tempfile.mkdtemp(prefix="nf_unload_")))
    n = 8
    buf = {"X": np.zeros(n), "Y": np.zeros(n), "Z": np.zeros(n)}
    built.run(buf, {"N": n})
    assert built.handle is None
    built.unload()
    assert built.lib is None


def test_close_after_unload_raises_when_a_handle_is_still_open():
    """Misuse case: unloading the library while a handle from ``init()`` is still open leaves nothing able
    to run ``__dace_exit`` on that handle. This must fail loudly rather than silently leak the handle or
    crash on a null CDLL lookup."""
    built = build_sdfg(parallel_axpy_sdfg(), Path(tempfile.mkdtemp(prefix="nf_unload2_")))
    built.init({"N": 8})
    built.unload()
    with pytest.raises(RuntimeError, match="unload"):
        built.close()


def test_external_linking_build_is_correct():
    """A nest built as a separate static ``.a`` (link_external) and linked into the ``.so`` runs identically
    to the monolithic build -- external linking is correct, not merely timeable."""
    built = owned_build_matches_oracle(
        "scientific_computing/dense_linear_algebra/gemm/gemm", opts=BuildOptions(link_external=True)
    )
    assert built.compile_seconds > 0.0
    assert (built.so_path.parent / f"lib{built.name}_nest.a").exists()  # the static node lib was produced


def test_parse_params_strips_the_const_qualifier_only_as_a_word():
    """``const`` is a QUALIFIER, not a substring: params literally named ``constant``/``const_term`` must
    keep their name, or the ctypes bind looks them up under a mangled key."""
    params = parse_params("k_state_t *__state, const double * __restrict__ constant, const int const_term")
    assert [p.name for p in params] == ["constant", "const_term"]
    assert params[0].is_pointer and params[0].ctype == ctypes.POINTER(ctypes.c_double)
    assert not params[1].is_pointer and params[1].ctype == ctypes.c_int


def test_parse_params_refuses_an_unmapped_by_value_scalar_type():
    """An unmapped by-value type must fail LOUD: defaulting to int64 puts a float in a GP register (SysV
    ABI), so the callee reads garbage with no ctypes error."""
    with pytest.raises(ValueError, match="uint64_t"):
        parse_params("k_state_t *__state, uint64_t n")


def test_parse_params_refuses_an_unmapped_pointer_base_type():
    """An unmapped pointer base type must fail LOUD too, the same as the scalar branch: silently defaulting
    to ``double*`` marshals a differently-sized element through the ABI with no ctypes error."""
    with pytest.raises(ValueError, match="uint64_t"):
        parse_params("k_state_t *__state, uint64_t *n")


def test_int_and_complex_pointers_bind_by_address():
    """xsbench's entry takes ``int*`` and scattering_self_energies' ``dace::complex128*``. A pointer carries only
    the buffer address, so each binds as a pointer to its element (or complex component) type."""
    params = parse_params("const int * __restrict__ index_grid, dace::complex128 * __restrict__ D, int64_t N")
    assert [(p.name, p.ctype) for p in params] == [
        ("index_grid", ctypes.POINTER(ctypes.c_int)),
        ("D", ctypes.POINTER(ctypes.c_double)),
        ("N", ctypes.c_int64),
    ]


def test_owned_build_reusable_handle_program():
    """After one init, __program can be called repeatedly in place (the timing path) on one handle, and
    every call still computes the right answer (this nest's output does not read its own prior value, so
    repeating the call is idempotent and one oracle run covers every rep)."""
    boundary = first_nest("scientific_computing/dense_linear_algebra/gemm/gemm")
    shape_syms = {
        s for s in boundary.symbols if any(s in str(d.shape) for d in boundary.standalone_sdfg.arrays.values())
    }
    sizes = {s: (32 if s in shape_syms else 0) for s in boundary.symbols}
    inputs = make_inputs(boundary, sizes, seed=1)
    prep = prepare(boundary, "k", Path(tempfile.mkdtemp()))
    oracle = run_oracle(prep, boundary, inputs, sizes)
    built = build_sdfg(boundary.standalone_sdfg, Path(tempfile.mkdtemp(prefix="nf_build_")))
    buf = {k: v.copy() for k, v in inputs.items()}
    built.init(sizes)
    try:
        for _ in range(5):
            built.program(buf, sizes)  # repeated in-place calls on the same state handle
    finally:
        built.close()
    for o in oracle:
        np.testing.assert_allclose(buf[o], oracle[o], rtol=1e-9, atol=1e-9, equal_nan=True)


def test_vectorized_owned_build_matches_oracle():
    """The DaCe multi-dim tile-op vectorizer plugs into the owned build: a VectorizeConfig on BuildOptions
    still matches the numpy oracle (AUTO resolves to the host ISA, so this stays host-agnostic)."""
    from dace.transformation.passes.vectorization.config import VectorizeConfig

    owned_build_matches_oracle(
        "scientific_computing/structured_grids/jacobi_1d/jacobi_1d",
        size=256,
        opts=BuildOptions(vectorize=VectorizeConfig(widths=(8,), target_isa="AUTO")),
    )


def test_toolchain_is_importable_without_dace():
    """The point of splitting it out of `build`: asking whether ldconfig knows about libomp must not drag
    in the DaCe codegen stack. `perf/flags` used to import `build` lazily for exactly this reason -- a
    workaround that only held as long as nobody hoisted the import. Load the module file with nothing else
    in sys.modules and assert dace never arrives."""
    # A SUBPROCESS, not this interpreter: the test module imports dace at line 18, and every other test in
    # the suite has too, so an in-process check could only ever assert `had_dace or ...` -- true before the
    # module is even loaded. The isolation being tested only exists in a fresh interpreter.
    path = Path(__file__).resolve().parents[1] / "nestforge" / "build" / "toolchain.py"
    probe = textwrap.dedent(f"""
        import importlib.util, sys
        spec = importlib.util.spec_from_file_location("nf_toolchain_isolated", {str(path)!r})
        module = importlib.util.module_from_spec(spec)
        sys.modules["nf_toolchain_isolated"] = module
        spec.loader.exec_module(module)
        assert "dace" not in sys.modules, "importing nestforge.build.toolchain pulled in dace"
        assert module.compiler_family("icx") == "llvm"          # a real probe, not just an import
        assert module.lib_findable("m", None) in (True, False)  # reaches ctypes.util, a SUBMODULE import
    """)
    r = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-800:]


# compiler diagnostics on the DaCe-generated C++
def test_resolved_flags_guarantee_the_standard_and_warnings_without_forcing_them():
    """Both are FILLED IN, not appended blindly: nearly every caller passes its own ``flags`` for one axis
    (an -O level, an FP mode) and would otherwise lose them. ``-Werror`` is deliberately absent -- this
    compiles generated C++ we do not own, so a warning is a codegen signal, not a failed measurement."""
    assert BuildOptions().resolved_flags()[-1] == "-Wall"
    assert BuildOptions(flags=["-O2"]).resolved_flags() == ["-O2", "-std=c++20", "-Wall"]
    # an explicit choice wins in both directions: -w silences, -Wextra is not duplicated into -Wall
    assert "-Wall" not in BuildOptions(flags=["-O2", "-w"]).resolved_flags()
    assert BuildOptions(flags=["-O2", "-Wall", "-Wextra"]).resolved_flags().count("-Wall") == 1
    assert not any(f.startswith("-Werror") for f in BuildOptions().resolved_flags())


def test_a_succeeding_compile_surfaces_its_warnings(tmp_path):
    """-Wall is inert unless the diagnostics are read: toolchain.run captured stderr and dropped it on
    success, so every warning from the generated C++ went to /dev/null."""
    src = tmp_path / "warn.cpp"
    src.write_text("int main() { int unused_variable = 1; return 0; }\n")
    with pytest.warns(UserWarning, match="unused_variable"):
        toolchain_mod.run(["g++", "-Wall", "-c", str(src), "-o", str(tmp_path / "warn.o")])


def test_a_clean_compile_warns_about_nothing(tmp_path):
    """The sweep compiles thousands of cells; a run that warned unconditionally would bury the real ones."""
    src = tmp_path / "clean.cpp"
    src.write_text("int main() { return 0; }\n")
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning at all fails this test
        toolchain_mod.run(["g++", "-Wall", "-c", str(src), "-o", str(tmp_path / "clean.o")])


def test_a_cpu_build_links_an_openmp_runtime_by_default():
    """dace emits `#pragma omp parallel for` for every multicore map. A build that passes no OpenMP flag
    DROPS the pragma and runs the schedule serially -- silently, since the program is still correct. So the
    runtime is resolved rather than left to the caller."""
    rt = toolchain_mod.usable_openmp("g++")
    assert rt is not None, "no OpenMP runtime linkable by g++ on this box"
    assert "-fopenmp" in " ".join(rt.compile_flags("g++"))


def test_the_resolved_runtime_is_named_never_a_bare_fopenmp():
    """A bare -fopenmp lets each family link its own default (gcc->libgomp, clang->libomp), so a sweep
    spanning compilers ends up with two thread pools in one process. libomp is preferred because it is
    LLVM-selectable AND GOMP-compatible, so gcc- and clang-built objects share one pool."""
    assert toolchain_mod.usable_openmp("g++") is toolchain_mod.usable_openmp("clang++")
    assert toolchain_mod.usable_openmp("g++").name == "libomp"


def test_an_explicit_runtime_is_not_overridden(tmp_path, monkeypatch):
    """A lane pinning a runtime (the support matrix sweeps them) must keep it THROUGH the compile.

    The previous version asserted `BuildOptions(openmp=X).openmp is X` -- a dataclass readback that holds
    for any implementation. Delete the `opts.openmp or` in build.compile and it still passed, while every
    pinned lane silently linked the resolved default and the runtime sweep measured one runtime under four
    names. This intercepts the flags the compile actually issues."""
    seen = []
    monkeypatch.setattr(build_mod, "run", lambda cmd, **k: seen.append(list(cmd)))
    monkeypatch.setattr(build_mod, "usable_openmp", lambda compiler: toolchain_mod.LIBOMP)
    src = tmp_path / "x.cpp"
    src.write_text("int main() { return 0; }\n")

    build_mod.compile(src, tmp_path, "x", BuildOptions(openmp=toolchain_mod.LIBGOMP))
    issued = " ".join(t for cmd in seen for t in cmd)
    # -lgomp vs -lomp is what actually separates the two on g++ (both compile with a plain -fopenmp), so
    # this is the token that fails if the pin is dropped for the resolved default.
    assert "-lgomp" in issued, f"the pinned libgomp never reached the link: {issued}"
    assert "-lomp" not in issued.replace("-lgomp", ""), "the resolved libomp replaced the pinned runtime"


def test_compiler_warnings_are_reported_but_bounded():
    """-Wall on DaCe-generated C++ fires on nearly every cell, each with its own paths and line numbers.
    `warnings.warn` dedups on exact TEXT, so it deduped nothing: a phase-1 sweep printed a distinct
    multi-KB block per compiled cell and grew __warningregistry__ for the life of the rank. Keyed on the
    warning KIND instead, and counted past a budget -- suppressed, never silently dropped."""
    toolchain_mod.WARNED.clear()
    try:
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            for cell in range(50):  # the same KIND, 50 different files
                toolchain_mod.warn_once("g++", f"/build/cell{cell}/x.cpp:{cell}:9: warning: unused [-Wunused-variable]")
        assert len(seen) == 1, f"one warning kind reported {len(seen)} times"
        summary = toolchain_mod.warning_summary()
        assert any("unused-variable" in line and "49 further" in line for line in summary), summary

        # a genuinely NEW kind is still reported, up to the budget
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            toolchain_mod.warn_once("g++", "x.cpp:1:1: warning: set but not used [-Wunused-but-set-variable]")
        assert len(seen) == 1
    finally:
        toolchain_mod.WARNED.clear()


def write_tool(bin_dir: Path, name: str, answer: str) -> None:
    """A fake driver on PATH that answers every query with ``answer``."""
    tool = bin_dir / name
    tool.write_text(f'#!/bin/sh\necho "{answer}"\n')
    tool.chmod(0o755)


@pytest.fixture
def cold_lookup_caches():
    """Runtime lookups are cached per name; a fake PATH must neither read nor leave a cached answer."""
    caches = (toolchain_mod.driver_lib_path, toolchain_mod.llvm_config_libdir, toolchain_mod.linkable_lib_dir)
    for cache in caches:
        cache.cache_clear()
    yield
    for cache in caches:
        cache.cache_clear()


def isolate_lookup(tmp_path: Path, monkeypatch) -> Path:
    """A PATH holding only the fake drivers a test writes, no environment, loader cache or layout hint."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    monkeypatch.setenv("PATH", str(bin_dir))
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    monkeypatch.delenv("LIBRARY_PATH", raising=False)
    monkeypatch.setattr(toolchain_mod, "ldconfig_dirs", lambda soname: [])
    monkeypatch.setattr(toolchain_mod, "LIB_DIR_HINT_ROOTS", ())
    monkeypatch.setattr(toolchain_mod, "LIB_DIR_HINTS", ())
    for cache in (toolchain_mod.driver_lib_path, toolchain_mod.llvm_config_libdir, toolchain_mod.linkable_lib_dir):
        cache.cache_clear()
    return bin_dir


@pytest.mark.parametrize(
    "gxx_answers, clang_answers, found_by",
    [(True, True, "gxx"), (False, True, "clang"), (False, False, "llvm-config")],
    ids=["the_compiler_first", "then_an_llvm_driver", "then_llvm_config_with_only_a_versioned_so"],
)
def test_libomp_lookup_asks_the_compiler_then_an_llvm_driver_then_llvm_config(
    tmp_path, monkeypatch, cold_lookup_caches, gxx_answers, clang_answers, found_by
):
    """Ubuntu installs libomp under the LLVM prefix, where g++ does not search: the lookup must reach it through
    clang or llvm-config, in that order, and accept libomp.so.5 when the unversioned symlink is missing."""
    bin_dir = isolate_lookup(tmp_path, monkeypatch)
    dirs = {name: tmp_path / name for name in ("gxx", "clang", "llvm-config")}
    for directory in dirs.values():
        directory.mkdir()
    (dirs["gxx"] / "libomp.so").write_bytes(b"")
    (dirs["clang"] / "libomp.so").write_bytes(b"")
    (dirs["llvm-config"] / "libomp.so.5").write_bytes(b"")
    write_tool(bin_dir, "g++", f"{dirs['gxx']}/libomp.so" if gxx_answers else "libomp.so")
    write_tool(bin_dir, "clang", f"{dirs['clang']}/libomp.so" if clang_answers else "libomp.so")
    write_tool(bin_dir, "llvm-config", str(dirs["llvm-config"]))
    expected = {"gxx": dirs["gxx"] / "libomp.so", "clang": dirs["clang"] / "libomp.so"}.get(
        found_by, dirs["llvm-config"] / "libomp.so.5"
    )

    assert runtime_library("omp", "g++") == expected


OMP_KERNEL_CXX = """extern "C" double kern(const double *a, int n) {
  double s = 0.0;
  #pragma omp parallel for reduction(+:s)
  for (int i = 0; i < n; i++) s += a[i];
  return s;
}
"""

LOAD_AND_CALL = """import ctypes, sys
lib = ctypes.CDLL(sys.argv[1])
lib.kern.restype = ctypes.c_double
n = 1000
values = (ctypes.c_double * n)(*range(n))
print(lib.kern(values, n))
"""


@pytest.mark.e2e
def test_a_libomp_only_llvm_config_finds_links_and_loads_from_its_own_directory(
    tmp_path, monkeypatch, cold_lookup_caches
):
    """The CI layout, rebuilt in a temp dir: libomp only as libomp.so.5 under an LLVM prefix g++ does not search.
    The discovered directory serves the link (-L, -l:libomp.so.5) and the load (rpath), with nothing on the
    loader path."""
    real_gxx, real_libomp = shutil.which("g++"), runtime_library("omp", "g++")
    assert real_gxx and real_libomp, "g++ and libomp are installed (setup_apt.sh)"
    real_env = dict(os.environ)  # the build and the load see the real PATH; only the lookup sees the fake one
    llvm_lib = tmp_path / "llvm" / "lib"
    llvm_lib.mkdir(parents=True)
    (llvm_lib / "libomp.so.5").symlink_to(os.path.realpath(real_libomp))
    bin_dir = isolate_lookup(tmp_path, monkeypatch)
    write_tool(bin_dir, "g++", "libomp.so")
    write_tool(bin_dir, "llvm-config", str(llvm_lib))

    flags = LIBOMP.link_flags("g++")

    assert flags == [f"-L{llvm_lib}", f"-Wl,-rpath,{llvm_lib}", "-l:libomp.so.5"]
    source, obj, shared = tmp_path / "kern.cpp", tmp_path / "kern.o", tmp_path / "libkern.so"
    source.write_text(OMP_KERNEL_CXX)
    subprocess.run([real_gxx, "-O2", "-fPIC", "-fopenmp", "-c", str(source), "-o", str(obj)], check=True, env=real_env)
    subprocess.run(
        [real_gxx, "-shared", "-Wl,--as-needed", str(obj), *flags, "-o", str(shared)], check=True, env=real_env
    )
    dynamic = subprocess.run(
        ["readelf", "-d", str(shared)], capture_output=True, text=True, check=True, env=real_env
    ).stdout
    assert "[libomp.so.5]" in dynamic and str(llvm_lib) in dynamic
    loaded = subprocess.run(
        [sys.executable, "-c", LOAD_AND_CALL, str(shared)], capture_output=True, text=True, env=real_env
    )
    assert loaded.returncode == 0, loaded.stderr[-1500:]
    assert float(loaded.stdout) == 999 * 1000 / 2
