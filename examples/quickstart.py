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
from nestforge.phases.normalize import Targets
from nestforge.session import Session

KERNEL = "loop_level_reasoning/fuse_diamond/fuse_diamond"
PRESET = "S"


def load_kernel() -> CorpusKernel:
    track = KERNEL.split("/", 1)[0]
    return next(kernel for kernel in iter_dace_kernels(track) if kernel.short_name == KERNEL)


def nest_count(session: Session) -> str:
    nests = session.list_nests()
    maps = sum(nest["kind"] == "map" for nest in nests)
    return f"{maps} map + {len(nests) - maps} loop nests"


def save(session: Session, out: Path, label: str) -> None:
    session.sdfg.save(str(out / f"{label}.sdfg"))


def copy_sources(folder: Path, dest: Path) -> None:
    """Every generated source of a DaCe program folder (C++ frame, CUDA), flattened into ``dest``."""
    dest.mkdir(parents=True, exist_ok=True)
    for src in sorted(folder.glob("src/*/*")):
        shutil.copy2(src, dest / src.name)


def run_phases_0_to_3(session: Session, out: Path) -> List[dict]:
    before = nest_count(session)
    session.normalize()
    save(session, out, "0-normalize")
    print(f"0 normalize         {before} -> {nest_count(session)}")
    before = nest_count(session)
    session.full_fusion()
    save(session, out, "1-shape-kernels")
    print(f"1 shape kernels     {before} -> {nest_count(session)}")
    scopes = session.define_scopes()
    save(session, out, "2-define-scopes")
    described = ", ".join(f"{k['name']} (reads {', '.join(k['reads'])}; writes {', '.join(k['writes'])})"
                          for k in scopes)
    print(f"2 define scopes     {len(scopes)} kernel(s): {described}; {nest_count(session)} left")
    placement = session.offload()
    if session.targets.gpu:
        save(session, out, "3-offload")
    devices = ", ".join(f"{k['name']} on {k['device']}" for k in placement["kernels"])
    copies = ", ".join(f"{src} -> {dst}" for src, dst in placement["copies"]) or "none"
    print(f"3 offload           {devices}; copies: {copies}")
    return placement["kernels"]


def optimize_and_sweep(session: Session, kernels: List[dict], sizes: Dict[str, int], out: Path) -> Dict[str, dict]:
    configs: Dict[str, dict] = {}
    for kernel in kernels:
        info = session.optimize_kernel(kernel["id"])
        name = info["kernel"]
        kernel_dir = out / "kernels" / name
        copy_sources(session.kernel_sources[kernel["id"]].program.folder, kernel_dir)
        entry = f"{info['symbol']}({', '.join(info['abi_order'])})"
        print(f"4 optimize kernels  {name}: DaCe C++ behind extern \"C\" {entry}")
        result = session.sweep_configurations(kernel["id"], sizes)
        if result["winner"] is None:
            raise SystemExit(f"phase 5 found no variant of {name} that matches its NumPy oracle")
        ext, _ = session.resolve(kernel["id"], "kernel")
        shutil.copy2(ext.lib_path, kernel_dir / Path(ext.lib_path).name)
        configs[name] = {**result, "symbol": ext.symbol, "abi_order": list(ext.abi_order)}
        print(f"5 sweep             {name}: {result['cells']} variants, "
              f"winner {result['winner']} at {result['time_us']:.1f} us")
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
    if session.targets.gpu:
        raise SystemExit("phases 4 and 5 do not build GPU kernels yet")
    configs = optimize_and_sweep(session, kernels, sizes, out)
    save(session, out, "4-optimize-kernels")
    (out / "5-sweep-configurations.json").write_text(json.dumps(configs, indent=2) + "\n")
    copy_sources(generate_program(session.sdfg, work / "program").folder, out / "program")
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
