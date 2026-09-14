# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Run the default optimizer's phases on HPCAgent-Bench's fuse_diamond and save what each phase produced."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List

import dace

from nestforge.build.sdfg import generate_program
from nestforge.corpus.bench import CorpusKernel, iter_dace_kernels, preset_sizes
from nestforge.ir.introspect import describe_graph
from nestforge.phases.kernel import build_kernel_library, kernel_runtime_libraries
from nestforge.phases.normalize import Targets
from nestforge.phases.variants import device_variants
from nestforge.session import Session

KERNEL = "loop_level_reasoning/fuse_diamond/fuse_diamond"
PRESET = "S"
#: What phase 5 decides per nest, as ``sweep_configurations`` reports it.
CONFIG_KEYS = ("compiler", "fp_mode", "cost_model", "flags", "time_us")


def load_kernel() -> CorpusKernel:
    track = KERNEL.split("/", 1)[0]
    return next(kernel for kernel in iter_dace_kernels(track) if kernel.short_name == KERNEL)


def nest_count(session: Session) -> str:
    nests = session.list_nests()
    maps = sum(nest["kind"] == "map" for nest in nests)
    return f"{maps} map + {len(nests) - maps} loop nests"


def save(session: Session, out: Path, label: str) -> None:
    session.sdfg.save(str(out / f"{label}.sdfg"))


def show_tree(session: Session, out: Path, label: str) -> None:
    """Print the program's structure tree and save it as ``<out>/trees/<label>.txt``."""
    tree = describe_graph(session.sdfg)
    trees = out / "trees"
    trees.mkdir(parents=True, exist_ok=True)
    (trees / f"{label}.txt").write_text(tree + "\n")
    print(tree)


def copy_sources(folder: Path, dest: Path) -> None:
    """Every generated source of a DaCe program folder (C++ frame, CUDA), flattened into ``dest``."""
    dest.mkdir(parents=True, exist_ok=True)
    for src in sorted(folder.glob("src/*/*")):
        shutil.copy2(src, dest / src.name)


def run_phases_0_to_3(session: Session, out: Path) -> List[dict]:
    show_tree(session, out, "0-input")
    before = nest_count(session)
    session.normalize()
    save(session, out, "0-normalize")
    print(f"0 normalize         {before} -> {nest_count(session)}")
    show_tree(session, out, "1-cpf")
    before = nest_count(session)
    session.full_fusion()
    save(session, out, "1-shape-kernels")
    print(f"1 shape kernels     {before} -> {nest_count(session)}")
    show_tree(session, out, "2-shaped")
    scopes = session.define_scopes()
    save(session, out, "2-define-scopes")
    described = ", ".join(
        f"{k['name']} (reads {', '.join(k['reads'])}; writes {', '.join(k['writes'])})" for k in scopes
    )
    print(f"2 define scopes     {len(scopes)} kernel(s): {described}; {nest_count(session)} left")
    placement = session.offload()
    if session.targets.gpu:
        save(session, out, "3-offload")
    devices = ", ".join(f"{k['name']} on {k['device']}" for k in placement["kernels"])
    copies = ", ".join(f"{src} -> {dst}" for src, dst in placement["copies"]) or "none"
    print(f"3 offload           {devices}; copies: {copies}")
    return placement["kernels"]


def optimize_kernels(session: Session, kernels: List[dict], out: Path) -> None:
    """Phase 4: each kernel's CPF unit, built with the first configuration phase 5 sweeps for its device and
    bound to its ``ExternalCall`` with the runtimes it needs."""
    for kernel in kernels:
        info = session.optimize_kernel(kernel["id"])
        name = info["kernel"]
        src = session.kernel_sources[kernel["id"]]
        kernel_dir = out / "kernels" / name
        kernel_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src.unit, kernel_dir / src.unit.name)
        variant = device_variants(src.device)[0]
        built = build_kernel_library(src, variant.compiler, list(variant.flags), session.work_dir / name / "phase4")
        library = kernel_dir / built.name
        shutil.copy2(built, library)
        runtime = kernel_runtime_libraries(src, variant.compiler)
        session.set_kernel(kernel["id"], str(library), info["symbol"], info["abi_order"], runtime_libraries=runtime)
        entry = f"{info['symbol']}({', '.join(info['abi_order'])})"
        print(
            f'4 optimize kernels  {name}: CPF unit {src.unit.name}, extern "C" {entry}, {library.name} by {variant.label}'
        )


def sweep_configurations(session: Session, kernels: List[dict], sizes: Dict[str, int]) -> Dict[str, dict]:
    """Phase 5: the fastest configuration per nest that matches its NumPy oracle."""
    configs: Dict[str, dict] = {}
    for kernel in kernels:
        result = session.sweep_configurations(kernel["id"], sizes)
        name = result["kernel"]
        if result["winner"] is None:
            raise SystemExit(f"phase 5 found no configuration of {name} that matches its NumPy oracle")
        configs[name] = {key: result[key] for key in CONFIG_KEYS}
        print(
            f"5 sweep             {name}: {result['cells']} variants, "
            f"winner {result['winner']} at {result['time_us']:.1f} us"
        )
    return configs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--out", type=Path, default=Path("quickstart_out"))
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    work = out / "work"
    dace.Config.set("default_build_folder", value=str(work / "dacecache"))
    kernel = load_kernel()
    sizes = preset_sizes(kernel, PRESET)
    print(f"{kernel.short_name}, preset {PRESET} {sizes}, device {args.device}")
    session = Session(kernel.to_sdfg(), targets=Targets(gpu=args.device == "gpu"), work_dir=str(work))
    kernels = run_phases_0_to_3(session, out)
    optimize_kernels(session, kernels, out)
    save(session, out, "4-optimize-kernels")
    copy_sources(generate_program(session.sdfg, work / "program").folder, out / "program")
    configs = sweep_configurations(session, kernels, sizes)
    (out / "5-sweep-configurations.json").write_text(json.dumps(configs, indent=2) + "\n")
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
