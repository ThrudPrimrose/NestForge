# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Kernel optimization on four tiny kernels: the default kernel is a standalone CPF unit, its one C entry takes
CPF's argument order with the types ``ExternalCall`` declares, ``lib<kernel>.a`` defines that entry, and every
built kernel matches its NumPy oracle."""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import dace

import nestforge.build.sdfg as build_sdfg

from nestforge.build.isolation import run_isolated
from nestforge.build.sdfg import compile_linked_program
from nestforge.build.toolchain import needed_libraries, raw_signature, split_params
from nestforge.corpus.translate import prepare
from nestforge.ir.libnode import proto_and_call
from nestforge.phases.kernel import (
    build_kernel_library,
    gpu_schedule,
    kernel_runtime_libraries,
    schedule_kernel,
    use_kernel_library,
    validate_kernel,
)
from nestforge.phases.normalize import Targets, normalize
from nestforge.phases.offload import device_copies, offload
from nestforge.phases.schedule import full_fusion
from nestforge.phases.scopes import lower_nests_to_external_call
from nestforge.phases.variants import device_variants

N = dace.symbol("N")
M = dace.symbol("M")


@dace.program
def vadd(b: dace.float64[N], c: dace.float64[N], a: dace.float64[N]):
    for i in dace.map[0:N]:
        a[i] = b[i] + c[i]


@dace.program
def stencil_row_sum(a: dace.float64[M, N], r: dace.float64[M]):
    for i in dace.map[0:M]:
        acc = 0.0
        for j in range(1, N - 1):
            acc = acc + (0.25 * a[i, j - 1] + 0.5 * a[i, j] + 0.25 * a[i, j + 1])
        r[i] = acc


@dace.program
def axpy_in_place(a: dace.float64[N], b: dace.float64[N]):
    for i in dace.map[0:N]:
        a[i] = a[i] * 2.0 + b[i]


@dace.program
def scaled(alpha: dace.float64, x: dace.float64[N], y: dace.float64[N]):
    for i in dace.map[0:N]:
        y[i] = alpha * x[i]


@dace.program
def scaled_by_cell(alpha: dace.float64[1], x: dace.float64[N], y: dace.float64[N]):
    for i in dace.map[0:N]:
        y[i] = alpha[0] * x[i]


KERNELS = [vadd, stencil_row_sum, axpy_in_place, scaled]
KERNEL_IDS = ["vadd", "stencil_row_sum", "axpy_in_place", "scaled"]
#: Extents off any vector width, so a compiler's remainder loop runs too.
KERNEL_SIZES = [{"N": 1037}, {"M": 13, "N": 67}, {"N": 1037}, {"N": 1037}]
#: DT_NEEDED soname stems of the OpenMP runtimes a library can name.
OPENMP_RUNTIME_STEMS = ("libgomp", "libomp", "libiomp5")
ENTRY_DEFINITION = re.compile(r'^extern "C" void (\w+)\s*\([^)]*\)\s*\{', re.M)


def lowered_kernel(program):
    """``program`` through phases 0-2: ``(parent SDFG, its one ExternalCall, that kernel's Boundary)``."""
    sdfg = program.to_sdfg(simplify=True)
    normalize(sdfg, Targets())
    full_fusion(sdfg, Targets())
    lowered = lower_nests_to_external_call(sdfg)
    assert len(lowered) == 1, [ext.name for ext, _ in lowered]
    ext, boundary = lowered[0]
    return sdfg, ext, boundary


def gpu_lowered_kernel(program, on_device=()):
    """``program`` through phases 0-3 for a GPU target; ``on_device`` names inputs the program keeps in GPU memory."""
    sdfg = program.to_sdfg(simplify=True)
    normalize(sdfg, Targets(gpu=True))
    full_fusion(sdfg, Targets(gpu=True))
    for name in on_device:
        sdfg.arrays[name].storage = dace.StorageType.GPU_Global
    ((ext, boundary),) = lower_nests_to_external_call(sdfg)
    offload(sdfg, Targets(gpu=True))
    return sdfg, ext, boundary


def split_decl(decl: str) -> tuple[str, str]:
    """``(type, name)`` of one C parameter declaration, qualifiers other than ``const`` dropped."""
    text = " ".join(re.sub(r"\b__restrict__\b", "", decl).split()).replace(" *", "*")
    name = re.split(r"[\s*]+", text)[-1]
    return text[: text.rfind(name)].strip(), name


def openmp_runtime_stems(shared):
    return [name.split(".so")[0] for name in needed_libraries(shared) if name.split(".so")[0] in OPENMP_RUNTIME_STEMS]


def entry_params(src):
    return [split_decl(p) for p in split_params(raw_signature(src.unit.read_text(), src.symbol))]


@pytest.mark.parametrize("program", KERNELS, ids=KERNEL_IDS)
def test_the_default_kernel_is_a_standalone_cpf_unit(tmp_path, program):
    """The unit builds with a bare compiler: no DaCe header, no DaCe runtime entry, and the boundary SDFG the
    extraction produced stays untouched, since phases 3 and 4 may schedule it again."""
    _, ext, boundary = lowered_kernel(program)
    before = boundary.standalone_sdfg.to_json()

    src = schedule_kernel(ext, boundary, tmp_path)

    text = src.unit.read_text()
    assert not re.search(r'#include\s*[<"]dace/', text)
    assert "__dace_" not in text
    assert "dace::" not in text
    assert boundary.standalone_sdfg.to_json() == before


def test_the_gpu_schedule_puts_every_argument_array_on_the_device_and_copies_nothing():
    """The kernel takes device pointers, so the whole kernel moves to the device and places no copy of its own."""
    _, _, boundary = lowered_kernel(vadd)

    sdfg = gpu_schedule(boundary)

    arguments = [desc for desc in sdfg.arrays.values() if not desc.transient]
    maps = [node for node, _ in sdfg.all_nodes_recursive() if isinstance(node, dace.nodes.MapEntry)]
    assert len(arguments) == 3
    assert all(desc.storage == dace.StorageType.GPU_Global for desc in arguments)
    assert any(entry.map.schedule == dace.ScheduleType.GPU_Device for entry in maps)
    assert device_copies(sdfg) == []


@pytest.mark.parametrize("program", [vadd, scaled], ids=["vadd", "scaled"])
def test_a_gpu_kernel_is_one_cuda_unit_whose_entry_matches_the_external_call_prototype(tmp_path, program):
    """After phase 3 places the kernel on the GPU, phase 4 renders CUDA with the same entry the parent calls."""
    sdfg, ext, boundary = gpu_lowered_kernel(program)
    src = schedule_kernel(ext, boundary, tmp_path)
    use_kernel_library(ext, tmp_path / "unused.a", src.symbol, src.abi_order, [])

    proto, _ = proto_and_call(ext, next(s for s in sdfg.all_states() if ext in s.nodes()))

    text = src.unit.read_text()
    assert (src.device, src.unit.suffix) == ("gpu", ".cu")
    assert ENTRY_DEFINITION.findall(text) == [ext.name] and text.count('extern "C"') == 1
    assert "cudaLaunchKernel(" in text and "__dace_" not in text
    declared = [split_decl(p) for p in split_params(re.search(r"\((.*)\)", proto).group(1))]
    assert declared == entry_params(src)


def test_a_length_one_device_array_input_is_a_device_pointer_in_the_prototype(tmp_path):
    sdfg, ext, boundary = gpu_lowered_kernel(scaled_by_cell, on_device=("alpha",))
    src = schedule_kernel(ext, boundary, tmp_path)
    use_kernel_library(ext, tmp_path / "unused.a", src.symbol, src.abi_order, [])

    proto, _ = proto_and_call(ext, next(s for s in sdfg.all_states() if ext in s.nodes()))

    assert proto == 'extern "C" void extcall_0(const double* alpha, const double* x, double* y, int64_t N);'
    assert [split_decl(p) for p in split_params(re.search(r"\((.*)\)", proto).group(1))] == entry_params(src)


def test_the_unit_defines_one_entry_in_cpf_order_not_manifest_order(tmp_path):
    """The entry takes CPF's order (arrays by name, then scalars by name). For ``vadd`` the output ``a``
    sorts before the inputs, so the manifest's role order differs -- binding by it would swap same-typed
    pointers silently."""
    _, ext, boundary = lowered_kernel(vadd)

    src = schedule_kernel(ext, boundary, tmp_path)

    assert ENTRY_DEFINITION.findall(src.unit.read_text()) == [ext.name]
    assert [name for _, name in entry_params(src)] == src.abi_order == ["a", "b", "c", "N"]
    assert src.abi_order != list(ext.config["input_args"])
    assert src.symbol == ext.name


@pytest.mark.parametrize("program", [vadd, scaled], ids=["vadd", "scaled"])
def test_the_entry_declares_each_parameter_as_the_external_call_prototype_does(tmp_path, program):
    """C linkage matches on the name alone, so a prototype/definition type mismatch links cleanly and
    corrupts the call: an ``int`` symbol must arrive as the ``int64_t`` the parent declares, an array as a
    pointer, and the read-only scalar ``alpha`` by value."""
    sdfg, ext, boundary = lowered_kernel(program)
    src = schedule_kernel(ext, boundary, tmp_path)
    use_kernel_library(ext, tmp_path / "unused.a", src.symbol, src.abi_order, [])

    proto, _ = proto_and_call(ext, next(s for s in sdfg.all_states() if ext in s.nodes()))

    declared = [split_decl(p) for p in split_params(re.search(r"\((.*)\)", proto).group(1))]
    assert declared == entry_params(src)


def test_a_read_only_scalar_input_crosses_the_boundary_by_value(tmp_path):
    """``alpha`` is a Scalar in the program, so the prototype takes its value and the call passes it."""
    sdfg, ext, boundary = lowered_kernel(scaled)
    src = schedule_kernel(ext, boundary, tmp_path)
    use_kernel_library(ext, tmp_path / "unused.a", src.symbol, src.abi_order, [])

    proto, call = proto_and_call(ext, next(s for s in sdfg.all_states() if ext in s.nodes()))

    assert proto == 'extern "C" void extcall_0(const double* x, double* y, int64_t N, double alpha);'
    assert call == "extcall_0(_in_x, _out_y, N, _in_alpha);"


def test_lowering_refuses_a_host_length_one_array_input_and_leaves_the_program_unchanged():
    """A host kernel takes a scalar input by value; a length-1 array standing in for one is refused, not
    converted, and nothing is extracted before the refusal."""
    sdfg = scaled_by_cell.to_sdfg(simplify=True)
    before = sdfg.to_json()

    with pytest.raises(ValueError, match=r"alpha.*length-1 arrays in host memory"):
        lower_nests_to_external_call(sdfg)

    assert sdfg.to_json() == before


@pytest.mark.e2e
def test_the_archive_defines_the_entry_once_and_no_dace_runtime(tmp_path):
    _, ext, boundary = lowered_kernel(vadd)
    src = schedule_kernel(ext, boundary, tmp_path / "gen")

    archive = build_kernel_library(src, "g++", None, tmp_path / "lib")

    assert archive == tmp_path / "lib" / f"lib{ext.name}.a"
    members = subprocess.run(["ar", "t", str(archive)], capture_output=True, text=True, check=True).stdout.split()
    assert members == [f"{ext.name}.o"]
    defined = subprocess.run(["nm", "--defined-only", str(archive)], capture_output=True, text=True, check=True).stdout
    assert re.findall(rf"^\S+ T ({re.escape(ext.name)})$", defined, re.M) == [ext.name]
    assert "__dace_" not in defined and "__program_" not in defined


@pytest.mark.e2e
@pytest.mark.parametrize("program, sizes", list(zip(KERNELS, KERNEL_SIZES)), ids=KERNEL_IDS)
def test_the_built_kernel_matches_its_numpy_oracle_bit_for_bit(tmp_path, program, sizes):
    """The shipped entry on seeded inputs, forked, against the kernel's NumPy reference at the strict rung;
    the in-place kernel is restored before every timed rep."""
    _, ext, boundary = lowered_kernel(program)
    src = schedule_kernel(ext, boundary, tmp_path / "gen")
    archive = build_kernel_library(src, "g++", None, tmp_path / "lib")
    prep = prepare(boundary, ext.name, tmp_path / "ref")

    verdict = validate_kernel(archive, src, prep, sizes, reps=3)

    assert verdict.error == "", verdict.error
    assert verdict.ok and verdict.maxdiff == 0.0, verdict
    assert verdict.time_us > 0.0


GPU_KERNELS = [(vadd, ()), (scaled, ()), (scaled_by_cell, ("alpha",))]


@pytest.mark.gpu
@pytest.mark.parametrize("program, on_device", GPU_KERNELS, ids=["vadd", "scaled_by_value", "alpha_on_device"])
def test_a_built_gpu_kernel_matches_its_numpy_oracle_bit_for_bit(tmp_path, program, on_device):
    """Device buffers for the arrays, a by-value scalar, and a length-1 device array all reach the CUDA kernel,
    built by nvcc at the strict rung."""
    _, ext, boundary = gpu_lowered_kernel(program, on_device)
    src = schedule_kernel(ext, boundary, tmp_path / "gen")
    strict = next(v for v in device_variants("gpu") if v.fp_mode == "strict-ieee")
    archive = build_kernel_library(src, strict.compiler, list(strict.flags), tmp_path / "lib")
    prep = prepare(boundary, ext.name, tmp_path / "ref")

    verdict = validate_kernel(archive, src, prep, {"N": 1037}, reps=3)

    assert verdict.error == "", verdict.error
    assert verdict.ok and verdict.maxdiff == 0.0, verdict
    assert verdict.time_us > 0.0


@pytest.mark.e2e
@pytest.mark.parametrize("compiler", ["g++", "clang++"])
def test_a_cpu_kernel_library_links_libomp_as_its_only_openmp_runtime(tmp_path, compiler):
    """Whichever compiler builds the kernel, its library names libomp, the process's one OpenMP runtime."""
    _, ext, boundary = lowered_kernel(vadd)
    src = schedule_kernel(ext, boundary, tmp_path / "gen")

    archive = build_kernel_library(src, compiler, None, tmp_path / "lib")

    assert openmp_runtime_stems(archive.with_suffix(".so")) == ["libomp"]


@pytest.mark.gpu
def test_a_gpu_kernel_library_links_cudart_and_no_foreign_openmp_runtime(tmp_path):
    _, ext, boundary = gpu_lowered_kernel(vadd)
    src = schedule_kernel(ext, boundary, tmp_path / "gen")
    strict = next(v for v in device_variants("gpu") if v.fp_mode == "strict-ieee")

    archive = build_kernel_library(src, strict.compiler, list(strict.flags), tmp_path / "lib")

    shared = archive.with_suffix(".so")
    assert any(name.startswith("libcudart.so") for name in needed_libraries(shared))
    assert [stem for stem in openmp_runtime_stems(shared) if stem != "libomp"] == []


@pytest.mark.gpu
def test_the_program_hands_a_length_one_device_array_to_the_gpu_kernel_as_a_device_pointer(tmp_path):
    """The program calls the linked CUDA kernel with ``alpha`` still in device memory. A host address in that
    slot reads garbage or faults, so the program's output is compared with NumPy, in a forked child."""
    import cupy  # the optional GPU array library, needed only where a GPU runs this test

    sdfg, ext, boundary = gpu_lowered_kernel(scaled_by_cell, on_device=("alpha",))
    src = schedule_kernel(ext, boundary, tmp_path / "gen")
    strict = next(v for v in device_variants("gpu") if v.fp_mode == "strict-ieee")
    archive = build_kernel_library(src, strict.compiler, list(strict.flags), tmp_path / "lib")
    use_kernel_library(ext, archive, src.symbol, src.abi_order, kernel_runtime_libraries(src, strict.compiler))
    sdfg.expand_library_nodes()
    compiled = compile_linked_program(sdfg, tmp_path / "parent")
    x = np.random.default_rng(0).random(1037)

    def run() -> dict:
        y = np.zeros_like(x)
        compiled(alpha=cupy.asarray([0.75]), x=x, y=y, N=x.size)
        return {"maxdiff": float(np.abs(y - 0.75 * x).max())}

    result = run_isolated(run)

    assert result == {"maxdiff": 0.0}
    program = compiled._lib._library_filename
    assert any(name.startswith("libcudart.so") for name in needed_libraries(program))
    assert [stem for stem in openmp_runtime_stems(program) if stem != "libomp"] == []


def recorded_links(monkeypatch):
    """Every command the build runs from now on, still run for real: a spy at the subprocess edge."""
    commands = []
    real_run = build_sdfg.run

    def record(cmd, *args, **kwargs):
        commands.append(list(cmd))
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(build_sdfg, "run", record)
    return commands


def shared_link(commands, shared):
    """The command that linked ``shared``."""
    (link,) = [cmd for cmd in commands if "-shared" in cmd and cmd[-1] == str(shared)]
    return link


@pytest.mark.e2e
def test_the_kernel_link_passes_as_needed_before_its_libraries(tmp_path, monkeypatch):
    _, ext, boundary = lowered_kernel(vadd)
    src = schedule_kernel(ext, boundary, tmp_path / "gen")
    commands = recorded_links(monkeypatch)

    archive = build_kernel_library(src, "g++", None, tmp_path / "lib")

    link = shared_link(commands, archive.with_suffix(".so"))
    libraries = [i for i, arg in enumerate(link) if arg.startswith("-l")]
    assert libraries and "-Wl,--as-needed" in link
    assert link.index("-Wl,--as-needed") < min(libraries)


@pytest.mark.gpu
def test_the_gpu_kernel_link_passes_as_needed_before_cudart(tmp_path, monkeypatch):
    _, ext, boundary = gpu_lowered_kernel(vadd)
    src = schedule_kernel(ext, boundary, tmp_path / "gen")
    strict = next(v for v in device_variants("gpu") if v.fp_mode == "strict-ieee")
    commands = recorded_links(monkeypatch)

    archive = build_kernel_library(src, strict.compiler, list(strict.flags), tmp_path / "lib")

    link = shared_link(commands, archive.with_suffix(".so"))
    assert "-Wl,--as-needed" in link and "-lcudart" in link
    assert link.index("-Wl,--as-needed") < link.index("-lcudart")


#: Creates this process's CUDA context, then validates a GPU kernel from the same process. A measurement run
#: in a child forked from a CUDA-initialized parent fails with CUDA status 3 (not initialized).
CUDA_INITIALIZED_PARENT = """import ctypes
import json
import os
import sys
from pathlib import Path

import dace

from nestforge.build.toolchain import discover_cuda_toolchains
from nestforge.corpus.translate import prepare
from nestforge.phases.kernel import build_kernel_library, schedule_kernel, validate_kernel
from nestforge.phases.normalize import Targets, normalize
from nestforge.phases.offload import offload
from nestforge.phases.schedule import full_fusion
from nestforge.phases.scopes import lower_nests_to_external_call
from nestforge.phases.variants import device_variants

N = dace.symbol("N")


@dace.program
def vadd(b: dace.float64[N], c: dace.float64[N], a: dace.float64[N]):
    for i in dace.map[0:N]:
        a[i] = b[i] + c[i]


if __name__ == "__main__":
    out = Path(sys.argv[1])
    cudart = ctypes.CDLL(os.path.join(discover_cuda_toolchains()[0].cudart_dir, "libcudart.so"))
    status = cudart.cudaFree(ctypes.c_void_p(0))
    sdfg = vadd.to_sdfg(simplify=True)
    normalize(sdfg, Targets(gpu=True))
    full_fusion(sdfg, Targets(gpu=True))
    ((ext, boundary),) = lower_nests_to_external_call(sdfg)
    offload(sdfg, Targets(gpu=True))
    strict = next(v for v in device_variants("gpu") if v.fp_mode == "strict-ieee")
    src = schedule_kernel(ext, boundary, out / "gen")
    archive = build_kernel_library(src, strict.compiler, list(strict.flags), out / "lib")
    verdict = validate_kernel(archive, src, prepare(boundary, ext.name, out / "ref"), {"N": 1037}, reps=2)
    print(json.dumps({"init_status": status, "error": verdict.error, "ok": verdict.ok, "maxdiff": verdict.maxdiff}))
"""


@pytest.mark.gpu
def test_a_gpu_kernel_measures_correctly_after_its_process_created_a_cuda_context(tmp_path):
    """A CUDA context does not survive fork, so the measurement must not run in a child forked from a process
    that already used the GPU. A fresh interpreter keeps this test's context out of the rest of the suite."""
    script = tmp_path / "cuda_initialized_parent.py"
    script.write_text(CUDA_INITIALIZED_PARENT)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(Path(__file__).resolve().parents[1]), *sys.path])}

    run = subprocess.run(
        [sys.executable, str(script), str(tmp_path)], capture_output=True, text=True, env=env, timeout=900
    )

    assert run.returncode == 0, run.stderr[-3000:]
    report = json.loads(run.stdout.strip().splitlines()[-1])
    assert report == {"init_status": 0, "error": "", "ok": True, "maxdiff": 0.0}
