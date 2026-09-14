# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Session phases 4 and 5: optimize a kernel, sweep its configurations, link the winner."""

import numpy as np
import pytest

import dace

from nestforge.build import flags
from nestforge.phases.normalize import Targets
from nestforge.session import Session

N = dace.symbol("N", dtype=dace.int64)

pytestmark = pytest.mark.e2e


@dace.program
def scaled_sum(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
    for i in dace.map[0:N]:
        c[i] = 2.0 * a[i] + b[i]


def one_kernel_session(tmp_path) -> tuple:
    sdfg = scaled_sum.to_sdfg(simplify=True)
    session = Session(sdfg, work_dir=str(tmp_path))
    kernels = session.define_scopes()
    assert len(kernels) == 1
    return session, kernels[0]["id"]


def test_optimize_kernel_writes_one_c_entry_with_the_boundary_arguments(tmp_path):
    """The optimized kernel exposes one extern C symbol whose arguments are the boundary's names."""
    session, kernel_id = one_kernel_session(tmp_path)
    info = session.optimize_kernel(kernel_id)
    unit = (tmp_path / info["kernel"]).rglob(f"{info['kernel']}.cpp")
    text = next(unit).read_text()
    assert text.count('extern "C"') == 1
    assert f"void {info['symbol']}(" in text
    assert sorted(info["abi_order"]) == sorted(session.kernel_boundary(kernel_id)["boundary_order"])


def test_sweep_links_the_fastest_correct_variant_into_the_program(tmp_path):
    """After the sweep the kernel calls the winning archive, and the program still computes 2a + b."""
    session, kernel_id = one_kernel_session(tmp_path)
    result = session.sweep_configurations(kernel_id, sizes={"N": 256}, reps=2, compilers=["gcc"])
    assert result["winner"] is not None
    assert result["cells"] >= 1
    assert result["compiler"] == "g++"
    assert result["winner"] == f"g++:{result['fp_mode']}:{result['cost_model']}"
    assert "-O3" in result["flags"] and result["time_us"] > 0.0
    ext, _ = session.resolve(kernel_id, "kernel")
    assert ext.implementation == "ExternCall"
    assert ext.lib_path.endswith(".a")

    a, b, c = np.random.rand(256), np.random.rand(256), np.zeros(256)
    session.sdfg(a=a, b=b, c=c, N=256)
    np.testing.assert_allclose(c, 2.0 * a + b)


def test_sweep_without_matching_compilers_reports_no_winner(tmp_path):
    """A compiler filter that matches nothing leaves the kernel on its DaCe reference expansion."""
    session, kernel_id = one_kernel_session(tmp_path)
    result = session.sweep_configurations(kernel_id, sizes={"N": 64}, reps=1, compilers=["no-such-compiler"])
    assert result["winner"] is None
    assert [result[key] for key in ("compiler", "fp_mode", "cost_model", "flags", "time_us")] == [None] * 5
    ext, _ = session.resolve(kernel_id, "kernel")
    assert ext.implementation != "ExternCall"


@pytest.mark.gpu
def test_a_gpu_sweep_reports_an_nvcc_configuration_without_a_cost_model(tmp_path):
    session = Session(scaled_sum.to_sdfg(simplify=True), targets=Targets(gpu=True), work_dir=str(tmp_path))
    session.define_scopes()
    (kernel,) = session.offload()["kernels"]

    result = session.sweep_configurations(kernel["id"], sizes={"N": 256}, reps=2)

    assert result["winner"] is not None
    assert result["compiler"].startswith("nvcc-") and result["cost_model"] == flags.NO_COST_MODEL
    assert result["fp_mode"] in flags.CUDA_FP_LEVELS and "-arch=native" in result["flags"]
    ext, _ = session.resolve(kernel["id"], "kernel")
    assert ext.implementation == "ExternCall" and ext.lib_path.endswith(".a")
