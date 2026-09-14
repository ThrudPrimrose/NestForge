# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Phase 3: the default offload runs every kernel on the GPU target and copies data around it."""

import numpy as np
import pytest

import dace
from dace.libraries.standard.helper import GPU_RESIDENT_STORAGES

from nestforge.ir.depends import KernelGraph, kernel_dependencies
from nestforge.phases.normalize import Targets
from nestforge.phases.offload import Transfer, transfers
from nestforge.phases.scopes import lower_nests_to_external_call
from nestforge.session import Session, StaleHandle

N = dace.symbol("N", dtype=dace.int64)


@dace.program
def scaled_sum(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
    for i in dace.map[0:N]:
        c[i] = 2.0 * a[i] + b[i]


@dace.program
def chain(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
    t = np.empty_like(a)
    for i in dace.map[0:N]:
        t[i] = a[i] + b[i]
    for i in dace.map[0:N]:
        c[i] = t[i] * 2.0


def lowered_graph(program) -> KernelGraph:
    sdfg = program.to_sdfg(simplify=True)
    lower_nests_to_external_call(sdfg)
    return kernel_dependencies(sdfg)


def kernel_session(gpu: bool, tmp_path) -> tuple:
    session = Session(scaled_sum.to_sdfg(simplify=True), targets=Targets(gpu=gpu), work_dir=str(tmp_path))
    kernels = session.define_scopes()
    assert len(kernels) == 1
    return session, kernels[0]["id"]


def test_cpu_target_keeps_the_kernel_on_the_host_and_leaves_the_graph_unchanged(tmp_path):
    session, kernel_id = kernel_session(False, tmp_path)
    before = session.sdfg.to_json()

    result = session.offload()

    assert result == {
        "kernels": [{"id": kernel_id, "name": "extcall_0", "device": "cpu"}],
        "copies": [],
        "transfers": [],
    }
    assert session.sdfg.to_json() == before


def test_gpu_target_schedules_the_kernel_on_the_device_over_device_memory(tmp_path):
    session, _ = kernel_session(True, tmp_path)

    (kernel,) = session.offload()["kernels"]

    ext = session.resolve(kernel["id"], "kernel")
    state = next(s for s in session.sdfg.all_states() if ext in s.nodes())
    operands = [e.src.data for e in state.in_edges(ext)] + [e.dst.data for e in state.out_edges(ext)]
    assert kernel["device"] == "gpu"
    assert ext.schedule == dace.ScheduleType.GPU_Device
    assert len(operands) == 3
    assert all(session.sdfg.arrays[name].storage in GPU_RESIDENT_STORAGES for name in operands)
    session.sdfg.validate()


def test_gpu_offload_copies_the_inputs_to_the_device_and_the_output_back(tmp_path):
    session, _ = kernel_session(True, tmp_path)

    copies = session.offload()["copies"]

    arrays = session.sdfg.arrays
    to_device = sorted(src for src, dst in copies if arrays[dst].storage in GPU_RESIDENT_STORAGES)
    to_host = sorted(dst for src, dst in copies if arrays[src].storage in GPU_RESIDENT_STORAGES)
    assert to_device == ["a", "b"]
    assert to_host == ["c"]


def test_gpu_offload_retires_the_kernel_ids_from_before_it(tmp_path):
    session, kernel_id = kernel_session(True, tmp_path)

    session.offload()

    with pytest.raises(StaleHandle):
        session.resolve(kernel_id, "kernel")


def test_define_scopes_after_gpu_offload_finds_no_further_scope(tmp_path):
    session, _ = kernel_session(True, tmp_path)
    session.offload()

    assert session.define_scopes() == []


def test_a_gpu_kernel_receives_its_program_inputs_and_returns_its_output_to_the_host():
    graph = lowered_graph(scaled_sum)

    moves = transfers(graph, {"extcall_0": "gpu"})

    assert moves == [
        Transfer("to_gpu", "a", "program", "extcall_0"),
        Transfer("to_gpu", "b", "program", "extcall_0"),
        Transfer("to_host", "c", "extcall_0.c", "exit"),
    ]


def test_a_host_kernel_feeding_a_gpu_kernel_moves_only_the_value_between_them():
    graph = lowered_graph(chain)
    assert graph.kernels == ("extcall_0", "extcall_1")

    moves = transfers(graph, {"extcall_0": "cpu", "extcall_1": "gpu"})

    assert moves == [
        Transfer("to_gpu", "t", "extcall_0.t", "extcall_1"),
        Transfer("to_host", "c", "extcall_1.c", "exit"),
    ]


@pytest.mark.gpu
def test_gpu_offload_transfers_name_the_containers_its_copies_move(tmp_path):
    session, _ = kernel_session(True, tmp_path)

    result = session.offload()

    arrays = session.sdfg.arrays
    to_device = sorted(src for src, dst in result["copies"] if arrays[dst].storage in GPU_RESIDENT_STORAGES)
    to_host = sorted(dst for src, dst in result["copies"] if arrays[src].storage in GPU_RESIDENT_STORAGES)
    assert sorted(move["arg"] for move in result["transfers"] if move["direction"] == "to_gpu") == to_device
    assert sorted(move["arg"] for move in result["transfers"] if move["direction"] == "to_host") == to_host


@pytest.mark.gpu
def test_offloaded_program_computes_on_the_gpu_what_numpy_computes(tmp_path):
    session, _ = kernel_session(True, tmp_path)
    session.offload()
    rng = np.random.default_rng(0)
    a, b, c = rng.random(256), rng.random(256), np.zeros(256)

    session.sdfg(a=a, b=b, c=c, N=256)

    np.testing.assert_allclose(c, 2.0 * a + b)
