# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Static-offload end-to-end: lower a map-nest to ExternalCall, build the arena winner as a static
``lib<name>_nest.a``, link it INTO the parent SDFG's ``.so`` (``--whole-archive``, no rpath), and run.

This is the ``.a`` feature's payoff: one binary, one libomp. An archive carries objects only -- no
linked runtime -- so the parent supplies the single OpenMP runtime instead of a separate nest ``.so``
dragging its own. The libomp-count assertion is what proves that (a second libomp would mean the nest
brought its own, defeating the whole point)."""
import subprocess

import numpy as np
import pytest
import dace

from nestforge.phases.scopes import lower_nests_to_external_call
from nestforge.ir.libnode import ExternLibEnv
from nestforge.corpus.translate import prepare, emit_sources
from nestforge.build.arena import run_arena, build_winner_archive

N = dace.symbol('N')


@dace.program
def vadd_static(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N]:
        C[i] = A[i] + B[i]


def reference_outputs(n):
    A = np.random.default_rng(0).random(n)
    B = np.random.default_rng(1).random(n)
    return A, B, A + B


def omp_runtimes(so_path: str):
    """The OpenMP runtime shared objects the built parent library pulls in, per ``ldd`` (libgomp for
    gcc, libomp/libiomp for clang/intel). Deduped by soname."""
    out = subprocess.check_output(['ldd', so_path], text=True)
    found = set()
    for line in out.splitlines():
        name = line.strip().split(' ', 1)[0]
        if any(tok in name for tok in ('libgomp', 'libomp', 'libiomp')):
            found.add(name)
    return found


@pytest.mark.integration
def test_static_offload_links_a_into_parent_and_runs(tmp_path):
    sdfg = vadd_static.to_sdfg(simplify=True)
    lowered = lower_nests_to_external_call(sdfg, strategy="outer")
    ext, boundary = lowered[0]

    # Arena -> winner -> materialize the winner as a static archive (NOT a .so).
    prep = prepare(boundary, ext.name, tmp_path / "kern")
    c_source = next(p for p in emit_sources(prep, tmp_path / "gen") if p.suffix == ".c")
    res = run_arena(prep, boundary, c_source, tmp_path / "build", sizes={"N": 1 << 14}, reps=25)
    win = res.winners["strict-ieee"]
    assert win.maxdiff == 0.0
    archive = build_winner_archive(win, c_source, ext.name, tmp_path / "archive")
    assert archive.suffix == ".a" and archive.is_file()
    # The archive is objects only -- no linked runtime rides in it.
    assert subprocess.check_output(['ar', 't', str(archive)], text=True).strip()

    # Point the node at the .a: configure() must switch to the static whole-archive link mode.
    ext.implementation = "ExternCall"
    ext.lib_path = str(archive)
    ext.symbol = win.symbol
    ext.abi_order = win.abi_order
    sdfg.expand_library_nodes()
    sdfg.validate()
    # Static link mode: the .a is a link input, no rpath to a separate .so.
    assert str(archive) in ExternLibEnv.cmake_libraries
    assert not any("-rpath" in f for f in ExternLibEnv.cmake_link_flags)  # statically in, not loaded

    # Build the parent ONCE so we can inspect the produced .so, then run it.
    sdfg.build_folder = str(tmp_path / "parent_cache")
    csdfg = sdfg.compile()
    parent_so = str(csdfg._lib._library_filename)

    n = 1 << 14
    A, B, ref = reference_outputs(n)
    C = np.zeros(n)
    csdfg(A=A, B=B, C=C, N=n)
    np.testing.assert_allclose(C, ref)

    # The point of the .a path: at most ONE libomp in the final binary. A statically-linked nest
    # contributes no runtime of its own, so the parent's is the only one (0 if neither uses OpenMP).
    assert len(omp_runtimes(parent_so)) <= 1


if __name__ == "__main__":
    import tempfile
    import pathlib
    test_static_offload_links_a_into_parent_and_runs(pathlib.Path(tempfile.mkdtemp()))
    print("static offload end-to-end OK")
