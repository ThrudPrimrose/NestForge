# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Session phases 4 and 5: optimize a kernel, sweep its configurations, link the winner."""

from pathlib import Path

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
    text = Path(info["unit"]).read_text()
    assert Path(info["unit"]).name == f"{info['kernel']}.cpp"
    assert text.count('extern "C"') == 1
    assert f"void {info['symbol']}(" in text
    assert info["entry"] == f"{info['symbol']}({', '.join(info['abi_order'])})"
    assert sorted(info["abi_order"]) == sorted(session.kernel_boundary(kernel_id)["boundary_order"])


def test_optimize_kernel_binds_its_default_library_and_the_program_still_computes(tmp_path):
    """Phase 4 alone links a library: the kernel calls the archive it built, with its runtimes, and 2a + b holds."""
    session, kernel_id = one_kernel_session(tmp_path)

    info = session.optimize_kernel(kernel_id)

    ext = session.resolve(kernel_id, "kernel")
    assert ext.implementation == "ExternCall"
    assert ext.lib_path == info["library"] and Path(info["library"]).name == f"lib{info['kernel']}.a"
    assert ext.runtime_libraries and info["variant"].count(":") == 2
    a, b, c = np.random.rand(256), np.random.rand(256), np.zeros(256)
    session.sdfg(a=a, b=b, c=c, N=256)
    np.testing.assert_allclose(c, 2.0 * a + b)


def test_a_kernel_another_session_lowered_can_be_optimized_by_a_fresh_session(tmp_path):
    sdfg = scaled_sum.to_sdfg(simplify=True)
    Session(sdfg, work_dir=str(tmp_path / "lowering")).define_scopes()
    fresh = Session(sdfg, work_dir=str(tmp_path / "fresh"))
    (kernel,) = fresh.list_kernels()

    info = fresh.optimize_kernel(kernel["id"])

    ext = fresh.resolve(kernel["id"], "kernel")
    assert ext.implementation == "ExternCall" and ext.lib_path == info["library"]
    assert sorted(info["abi_order"]) == sorted(fresh.kernel_boundary(kernel["id"])["boundary_order"])
    a, b, c = np.random.rand(256), np.random.rand(256), np.zeros(256)
    sdfg(a=a, b=b, c=c, N=256)
    np.testing.assert_allclose(c, 2.0 * a + b)


def test_a_kernel_without_its_standalone_sdfg_is_refused_before_it_is_scheduled(tmp_path):
    session, kernel_id = one_kernel_session(tmp_path)
    session.resolve(kernel_id, "kernel").standalone_sdfg = None

    with pytest.raises(ValueError, match="ExternalCall 'extcall_0' has no standalone SDFG"):
        session.optimize_kernel(kernel_id)


def test_sweep_links_the_fastest_correct_variant_into_the_program(tmp_path):
    """After the sweep the kernel calls the winning archive, and the program still computes 2a + b."""
    session, kernel_id = one_kernel_session(tmp_path)
    result = session.sweep_configurations(kernel_id, sizes={"N": 256}, reps=2, compilers=["gcc"])
    assert result["winner"] is not None
    assert result["cells"] >= 1
    config = result["config"]
    assert config["compiler"] == "g++"
    assert result["winner"] == f"g++:{config['fp_mode']}:{config['cost_model']}"
    assert "-O3" in config["flags"] and config["time_us"] > 0.0
    ext = session.resolve(kernel_id, "kernel")
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
    assert result["config"] == dict.fromkeys(("compiler", "fp_mode", "cost_model", "flags", "time_us"))
    ext = session.resolve(kernel_id, "kernel")
    assert ext.implementation != "ExternCall"


@pytest.mark.gpu
def test_a_gpu_sweep_reports_an_nvcc_configuration_without_a_cost_model(tmp_path):
    session = Session(scaled_sum.to_sdfg(simplify=True), targets=Targets(gpu=True), work_dir=str(tmp_path))
    session.define_scopes()
    (kernel,) = session.offload()["kernels"]

    result = session.sweep_configurations(kernel["id"], sizes={"N": 256}, reps=2)

    assert result["winner"] is not None
    config = result["config"]
    assert config["compiler"].startswith("nvcc-") and config["cost_model"] == flags.NO_COST_MODEL
    assert config["fp_mode"] in flags.CUDA_FP_LEVELS and "-arch=native" in config["flags"]
    ext = session.resolve(kernel["id"], "kernel")
    assert ext.implementation == "ExternCall" and ext.lib_path.endswith(".a")
