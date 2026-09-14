# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Session over the kernel dependency graph: one graph per epoch, the kernel listing, the tree's dependency lines,
and the per-epoch JSON snapshot."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

import dace

from nestforge.ir.depends import UnsupportedProgram
from nestforge.session import Session

N = dace.symbol("N")


@dace.program
def chain(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    T = np.empty_like(A)
    for i in dace.map[0:N]:
        T[i] = A[i] + B[i]
    for i in dace.map[0:N]:
        C[i] = T[i] * 2.0


def scoped_session(sdfg: dace.SDFG, work_dir: Path) -> Session:
    session = Session(sdfg, work_dir=str(work_dir))
    session.define_scopes()
    return session


def test_the_kernel_graph_is_computed_once_per_epoch_and_again_after_a_mutation(tmp_path):
    session = Session(chain.to_sdfg(simplify=True), work_dir=str(tmp_path))
    before = session.kernel_graph()
    assert session.kernel_graph() is before

    session.define_scopes()

    after = session.kernel_graph()
    assert after is not before
    assert (before.kernels, after.kernels) == ((), ("extcall_0", "extcall_1"))


def test_list_kernels_names_each_argument_and_what_reaches_it(tmp_path):
    session = scoped_session(chain.to_sdfg(simplify=True), tmp_path)

    kernels = session.list_kernels()

    assert [{key: value for key, value in kernel.items() if key != "id"} for kernel in kernels] == [
        {
            "name": "extcall_0",
            "device": None,
            "inputs": ["A", "B"],
            "outputs": ["T"],
            "symbols": ["N"],
            "depends": {"A": ["program"], "B": ["program"], "N": ["program"]},
            "carried": {},
        },
        {
            "name": "extcall_1",
            "device": None,
            "inputs": ["T"],
            "outputs": ["C"],
            "symbols": ["N"],
            "depends": {"T": ["extcall_0.T"], "N": ["program"]},
            "carried": {},
        },
    ]
    assert [session.resolve(kernel["id"], "kernel").name for kernel in kernels] == ["extcall_0", "extcall_1"]


def test_kernel_ids_stay_kernel_handles_after_the_tree_stamps_the_same_nodes(tmp_path):
    session = scoped_session(chain.to_sdfg(simplify=True), tmp_path)
    session.describe()

    kernels = session.list_kernels()

    assert [session.resolve(kernel["id"], "kernel").name for kernel in kernels] == ["extcall_0", "extcall_1"]
    assert [kernel["id"] for kernel in session.list_kernels()] == [kernel["id"] for kernel in kernels]


def test_list_kernels_reports_the_device_phase_3_placed_each_kernel_on(tmp_path):
    session = scoped_session(chain.to_sdfg(simplify=True), tmp_path)

    session.offload()

    assert [kernel["device"] for kernel in session.list_kernels()] == ["cpu", "cpu"]


def test_describe_with_deps_prints_each_kernels_line_under_its_row(tmp_path):
    session = scoped_session(chain.to_sdfg(simplify=True), tmp_path)

    lines = session.describe(deps=True).splitlines()

    row = next(index for index, line in enumerate(lines) if "extcall_1  LIBNODE" in line)
    assert lines[row + 1].endswith("extcall_1: T <- extcall_0.T, N <- program")
    assert "extcall_1: T <- extcall_0.T" not in session.describe()


def test_define_scopes_completes_without_a_snapshot_when_the_analysis_refuses_the_program(tmp_path):
    sdfg = chain.to_sdfg(simplify=True)
    sdfg.add_reference("R", [N], dace.float64)
    session = Session(sdfg, work_dir=str(tmp_path))

    kernels = session.define_scopes()

    assert [kernel["name"] for kernel in kernels] == ["extcall_0", "extcall_1"]
    assert not (tmp_path / "kernel_deps").exists()
    with pytest.raises(UnsupportedProgram, match="'R' .* Reference"):
        session.kernel_graph()
    with pytest.raises(UnsupportedProgram, match="'R' .* Reference"):
        session.list_kernels()


def test_define_scopes_saves_the_epochs_graph_byte_identically_for_a_copied_program(tmp_path):
    sdfg = chain.to_sdfg(simplify=True)
    session = scoped_session(copy.deepcopy(sdfg), tmp_path / "first")
    twin = scoped_session(copy.deepcopy(sdfg), tmp_path / "second")

    snapshot = tmp_path / "first" / "kernel_deps" / f"e{session.epoch}.json"

    assert json.loads(snapshot.read_text()) == session.kernel_graph().to_json()
    assert snapshot.read_bytes() == (tmp_path / "second" / "kernel_deps" / f"e{twin.epoch}.json").read_bytes()
