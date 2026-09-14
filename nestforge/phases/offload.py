# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Phase 3: give every kernel a device and insert the host/device copies that placement implies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import dace
from dace import dtypes
from dace.libraries.standard.helper import GPU_RESIDENT_STORAGES
from dace.sdfg import nodes
from dace.transformation.passes.offloading.offload_to_accelerator import OffloadToAccelerator

from nestforge.ir.depends import ArgEdge, KernelGraph
from nestforge.ir.libnode import ExternalCall
from nestforge.phases.normalize import Targets


@dataclass(frozen=True, slots=True)
class Placement:
    """Device per kernel name, and ``(source, destination)`` containers of every host/device copy."""

    devices: Dict[str, str]
    copies: Tuple[Tuple[str, str], ...]


def external_calls(sdfg: dace.SDFG) -> List[ExternalCall]:
    return [node for node, _ in sdfg.all_nodes_recursive() if isinstance(node, ExternalCall)]


def kernel_device(ext: ExternalCall) -> str:
    return "gpu" if ext.schedule in dtypes.GPU_SCHEDULES else "cpu"


def on_device(state: dace.SDFGState, node: nodes.AccessNode) -> bool:
    return node.desc(state.sdfg).storage in GPU_RESIDENT_STORAGES


def device_copies(sdfg: dace.SDFG) -> List[Tuple[str, str]]:
    """Access-to-access edges whose two ends live in different memory spaces, in state order."""
    copies: List[Tuple[str, str]] = []
    for state in sdfg.all_states():
        for edge in state.edges():
            if not isinstance(edge.src, nodes.AccessNode) or not isinstance(edge.dst, nodes.AccessNode):
                continue
            if on_device(state, edge.src) != on_device(state, edge.dst):
                copies.append((edge.src.data, edge.dst.data))
    return copies


def offload(sdfg: dace.SDFG, targets: Targets) -> Placement:
    """Default optimizer, in place. With a GPU target DaCe's ``OffloadToAccelerator`` schedules every
    ``ExternalCall`` at host level on the device and places the copies; without one nothing changes."""
    if targets.gpu:
        OffloadToAccelerator().apply_pass(sdfg, {})
    devices = {ext.name: kernel_device(ext) for ext in external_calls(sdfg)}
    return Placement(devices, tuple(device_copies(sdfg)))


@dataclass(frozen=True, slots=True)
class Transfer:
    """A value crossing memory spaces on one dependency edge: ``producer`` feeds ``consumer``'s ``arg``."""

    direction: str
    arg: str
    producer: str
    consumer: str


def edge_transfers(edge: ArgEdge, consumer_device: str, devices: Dict[str, str]) -> List[Transfer]:
    moves: List[Transfer] = []
    # a producer reaching both carried and not moves once
    for producer in dict.fromkeys(reach.producer for reach in edge.producers):
        source_device = devices[producer.name] if producer.kind == "kernel" else "cpu"
        if source_device != consumer_device:
            direction = "to_gpu" if consumer_device == "gpu" else "to_host"
            moves.append(Transfer(direction, edge.arg, producer.label(), edge.consumer))
    return moves


def transfers(graph: KernelGraph, devices: Dict[str, str]) -> List[Transfer]:
    """Every kernel input and program exit whose producer sits in the other memory space. ``program`` and ``host``
    producers, and the program exit, are host memory; ``devices`` maps each kernel to ``cpu`` or ``gpu``."""
    moves: List[Transfer] = []
    for edge in graph.edges:
        if edge.role == "input":
            moves += edge_transfers(edge, devices[edge.consumer], devices)
    for edge in graph.exits:
        moves += edge_transfers(edge, "cpu", devices)
    return moves
