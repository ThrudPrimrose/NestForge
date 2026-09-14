# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Phase 3: the default offload runs every kernel on the GPU target and copies data around it."""
import numpy as np
import pytest

import dace
from dace.libraries.standard.helper import GPU_RESIDENT_STORAGES

from nestforge.phases.normalize import Targets
from nestforge.session import Session, StaleHandle

N = dace.symbol("N", dtype=dace.int64)


@dace.program
def scaled_sum(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
    for i in dace.map[0:N]:
        c[i] = 2.0 * a[i] + b[i]


def kernel_session(gpu: bool, tmp_path) -> tuple:
    session = Session(scaled_sum.to_sdfg(simplify=True), targets=Targets(gpu=gpu), work_dir=str(tmp_path))
    kernels = session.define_scopes()
    assert len(kernels) == 1
    return session, kernels[0]["id"]


def test_cpu_target_keeps_the_kernel_on_the_host_and_leaves_the_graph_unchanged(tmp_path):
    session, kernel_id = kernel_session(False, tmp_path)
    before = session.sdfg.to_json()

    result = session.offload()

    assert result == {"kernels": [{"id": kernel_id, "name": "extcall_0", "device": "cpu"}], "copies": []}
    assert session.sdfg.to_json() == before


def test_gpu_target_schedules_the_kernel_on_the_device_over_device_memory(tmp_path):
    session, _ = kernel_session(True, tmp_path)

    (kernel, ) = session.offload()["kernels"]

    ext, _ = session.resolve(kernel["id"], "kernel")
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


@pytest.mark.gpu
def test_offloaded_program_computes_on_the_gpu_what_numpy_computes(tmp_path):
    session, _ = kernel_session(True, tmp_path)
    session.offload()
    rng = np.random.default_rng(0)
    a, b, c = rng.random(256), rng.random(256), np.zeros(256)

    session.sdfg(a=a, b=b, c=c, N=256)

    np.testing.assert_allclose(c, 2.0 * a + b)
