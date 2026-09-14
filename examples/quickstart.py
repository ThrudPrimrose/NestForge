# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Walk HPCAgent-Bench's fuse_diamond through the Session API, one call per phase, and save what each phase made."""

import argparse
import json
import shutil
from pathlib import Path

import dace

from nestforge.build.sdfg import generate_program
from nestforge.corpus.bench import iter_dace_kernels, preset_sizes
from nestforge.ir.introspect import describe_graph
from nestforge.phases.normalize import Targets
from nestforge.session import Session

KERNEL = "loop_level_reasoning/fuse_diamond/fuse_diamond"
PRESET = "S"


def nests(session: Session) -> str:
    kinds = [nest["kind"] for nest in session.list_nests()]
    return f"{kinds.count('map')} map + {kinds.count('loop')} loop nests"


def save(session: Session, out: Path, label: str) -> None:
    session.sdfg.save(str(out / f"{label}.sdfg"))


def show_tree(session: Session, out: Path, label: str) -> None:
    """Print the program's structure tree and save it as ``trees/<label>.txt``."""
    tree = describe_graph(session.sdfg)
    (out / "trees").mkdir(parents=True, exist_ok=True)
    (out / "trees" / f"{label}.txt").write_text(tree + "\n")
    print(tree)


def run_phases_0_to_3(session: Session, out: Path) -> list[dict]:
    show_tree(session, out, "0-input")
    before = nests(session)
    session.normalize()
    save(session, out, "0-normalize")
    print(f"0 normalize         {before} -> {nests(session)}")
    show_tree(session, out, "1-cpf")
    before = nests(session)
    session.full_fusion()
    save(session, out, "1-shape-kernels")
    print(f"1 shape kernels     {before} -> {nests(session)}")
    show_tree(session, out, "2-shaped")
    scopes = session.define_scopes()
    save(session, out, "2-define-scopes")
    kernels = ", ".join(f"{k['name']} (reads {', '.join(k['reads'])}; writes {', '.join(k['writes'])})" for k in scopes)
    print(f"2 define scopes     {len(scopes)} kernel(s): {kernels}; {nests(session)} left")
    placement = session.offload()
    if session.targets.gpu:
        save(session, out, "3-offload")
    devices = ", ".join(f"{k['name']} on {k['device']}" for k in placement["kernels"])
    copies = ", ".join(f"{src} -> {dst}" for src, dst in placement["copies"]) or "none"
    print(f"3 offload           {devices}; copies: {copies}")
    return placement["kernels"]


def optimize_kernels(session: Session, kernels: list[dict], out: Path) -> None:
    """Phase 4: each kernel's default library, bound by the Session; its unit and library are copied out."""
    for kernel in kernels:
        info = session.optimize_kernel(kernel["id"])
        kernel_dir = out / "kernels" / info["kernel"]
        kernel_dir.mkdir(parents=True, exist_ok=True)
        unit, library = (Path(shutil.copy2(info[key], kernel_dir)) for key in ("unit", "library"))
        print(
            f'4 optimize kernels  {info["kernel"]}: CPF unit {unit.name}, extern "C" {info["entry"]}, {library.name} by {info["variant"]}'
        )


def sweep_configurations(session: Session, kernels: list[dict], sizes: dict[str, int]) -> dict[str, dict]:
    """Phase 5: per nest, the fastest configuration that matches its NumPy oracle."""
    configs: dict[str, dict] = {}
    for kernel in kernels:
        result = session.sweep_configurations(kernel["id"], sizes)
        if result["winner"] is None:
            raise SystemExit(f"phase 5 found no configuration of {result['kernel']} that matches its NumPy oracle")
        configs[result["kernel"]] = result["config"]
        print(
            f"5 sweep             {result['kernel']}: {result['cells']} variants, winner {result['winner']} at {result['config']['time_us']:.1f} us"
        )
    return configs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--out", type=Path, default=Path("quickstart_out"))
    args = parser.parse_args()
    out, work = args.out.resolve(), args.out.resolve() / "work"
    dace.Config.set("default_build_folder", value=str(work / "dacecache"))
    kernel = next(k for k in iter_dace_kernels(KERNEL.split("/", 1)[0]) if k.short_name == KERNEL)
    sizes = preset_sizes(kernel, PRESET)
    print(f"{kernel.short_name}, preset {PRESET} {sizes}, device {args.device}")
    session = Session(kernel.to_sdfg(), targets=Targets(gpu=args.device == "gpu"), work_dir=str(work))
    kernels = run_phases_0_to_3(session, out)
    optimize_kernels(session, kernels, out)
    save(session, out, "4-optimize-kernels")
    (out / "program").mkdir(exist_ok=True)
    for source in sorted(generate_program(session.sdfg, work / "program").folder.glob("src/*/*")):
        shutil.copy2(source, out / "program" / source.name)
    configs = sweep_configurations(session, kernels, sizes)
    (out / "5-sweep-configurations.json").write_text(json.dumps(configs, indent=2) + "\n")
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
