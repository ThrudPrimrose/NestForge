# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Container-level dataflow between kernels: for every ``ExternalCall`` argument, the producers whose value can
reach it. Whole containers only: a read reads all of it, a write replaces all of it (``docs/depends.md``)."""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import dace
from dace.properties import CodeBlock
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.sdfg.graph import MultiConnectorEdge
from dace.sdfg.state import (
    BreakBlock,
    ConditionalBlock,
    ContinueBlock,
    ControlFlowBlock,
    ControlFlowRegion,
    LoopRegion,
    ReturnBlock,
    SDFGState,
)
from dace.transformation.passes.analysis import loop_analysis
from dace.transformation.passes.analysis.analysis import names_read_by_text

from nestforge.ir.libnode import ExternalCall, in_conn, out_conn
from nestforge.ir.names import in_order

INPUT_PREFIX = in_conn("")
OUTPUT_PREFIX = out_conn("")
ROLE_ORDER = {"input": 0, "symbol": 1}


class UnsupportedProgram(Exception):
    """The program holds a construct whose producers cannot be named at container level."""


@dataclass(frozen=True, slots=True)
class Producer:
    """Who wrote a value: ``program`` (never written), ``kernel`` (an ``ExternalCall`` output) or ``host``."""

    kind: str
    name: str = ""
    arg: str = ""

    def label(self) -> str:
        """``program``, ``<kernel>.<arg>`` or ``host:<state>``."""
        if self.kind == "kernel":
            return f"{self.name}.{self.arg}"
        if self.kind == "host":
            return f"host:{self.name}"
        return self.kind


@dataclass(frozen=True, slots=True)
class Reach:
    """A producer reaching a use; ``carried_by`` names each loop whose back edge the value crossed."""

    producer: Producer
    carried_by: Tuple[str, ...] = ()

    def text(self) -> str:
        """The producer label, with ``[carried: <loops>]`` when the value crossed a back edge."""
        carried = f" [carried: {' > '.join(self.carried_by)}]" if self.carried_by else ""
        return self.producer.label() + carried


@dataclass(frozen=True, slots=True)
class ArgEdge:
    """The producers reaching one argument of one consumer (a kernel, or ``exit``)."""

    consumer: str
    arg: str
    role: str
    producers: Tuple[Reach, ...]
    via: Tuple[str, ...] = ()

    def text(self) -> str:
        """``arg <- p0 | p1`` plus the interstate assignments the value flowed through."""
        producers = " | ".join(reach.text() for reach in self.producers) or "none"
        via = " via " + "; ".join(f'"{text}"' for text in self.via) if self.via else ""
        return f"{self.arg} <- {producers}{via}"

    def labels(self) -> List[str]:
        """The distinct producer labels, carried or not, in producer order."""
        return list(dict.fromkeys(reach.producer.label() for reach in self.producers))

    def loops(self) -> List[str]:
        """The distinct loops whose back edge a reaching value crossed."""
        return list(dict.fromkeys(loop for reach in self.producers for loop in reach.carried_by))

    def to_json(self) -> dict:
        """A plain, key-ordered dictionary."""
        producers = [
            {
                "kind": reach.producer.kind,
                "name": reach.producer.name,
                "arg": reach.producer.arg,
                "carried_by": list(reach.carried_by),
            }
            for reach in self.producers
        ]
        return {
            "consumer": self.consumer,
            "arg": self.arg,
            "role": self.role,
            "producers": producers,
            "via": list(self.via),
        }


@dataclass(frozen=True, slots=True)
class KernelGraph:
    """Kernels in program order, their argument edges, and the producers of each program output at exit."""

    kernels: Tuple[str, ...]
    edges: Tuple[ArgEdge, ...]
    exits: Tuple[ArgEdge, ...]

    def consumers_of(self, kernel: str) -> Tuple[ArgEdge, ...]:
        """Every kernel argument edge that ``kernel`` can produce."""
        return tuple(
            edge
            for edge in self.edges
            if any(reach.producer.kind == "kernel" and reach.producer.name == kernel for reach in edge.producers)
        )

    def arguments(self, kernel: str) -> Tuple[ArgEdge, ...]:
        """The argument edges of ``kernel``, inputs then symbols."""
        return tuple(edge for edge in self.edges if edge.consumer == kernel)

    def line(self, kernel: str) -> str:
        """``kernel: arg <- producers, ...``."""
        return f"{kernel}: " + ", ".join(edge.text() for edge in self.arguments(kernel))

    def lines(self) -> List[str]:
        """One :meth:`line` per kernel, then one ``exit:`` line."""
        out = [self.line(kernel) for kernel in self.kernels]
        if self.exits:
            out.append("exit: " + ", ".join(edge.text() for edge in self.exits))
        return out

    def to_json(self) -> dict:
        """A plain dictionary in the graph's sorted order."""
        return {
            "kernels": list(self.kernels),
            "edges": [edge.to_json() for edge in self.edges],
            "exits": [edge.to_json() for edge in self.exits],
        }


@dataclass(frozen=True, slots=True)
class Fact:
    reaches: Tuple[Reach, ...] = ()
    via: Tuple[str, ...] = ()


#: What reaches each container or symbol name at one program point; an absent name has no producer.
Env = Dict[str, Fact]

EMPTY = Fact()
PROGRAM = Fact((Reach(Producer("program")),))


@dataclass(frozen=True, slots=True)
class Outcome:
    normal: Optional[Env] = None
    breaks: Optional[Env] = None
    continues: Optional[Env] = None
    returns: Optional[Env] = None


@dataclass(slots=True)
class Tracker:
    kernels: Dict[str, None] = field(default_factory=dict)
    # (consumer, role, arg) -> edge; the last visit is the fixpoint's
    edges: Dict[Tuple[str, str, str], ArgEdge] = field(default_factory=dict)
    written: Dict[str, None] = field(default_factory=dict)


def reach_key(reach: Reach) -> Tuple[str, Tuple[str, ...]]:
    return reach.producer.label(), reach.carried_by


def merge(*facts: Fact) -> Fact:
    reaches = dict.fromkeys(reach for fact in facts for reach in fact.reaches)
    via = dict.fromkeys(text for fact in facts for text in fact.via)
    return Fact(tuple(sorted(reaches, key=reach_key)), tuple(sorted(via)))


def union(first: Env, second: Env) -> Env:
    names = dict.fromkeys([*first, *second])
    return {name: merge(first.get(name, EMPTY), second.get(name, EMPTY)) for name in names}


def join(*envs: Optional[Env]) -> Optional[Env]:
    present = [env for env in envs if env is not None]
    if not present:
        return None
    joined = present[0]
    for env in present[1:]:
        joined = union(joined, env)
    return joined


def join_outcomes(outcomes: List[Outcome], normal: Optional[Env]) -> Outcome:
    return Outcome(
        normal,
        join(*(outcome.breaks for outcome in outcomes)),
        join(*(outcome.continues for outcome in outcomes)),
        join(*(outcome.returns for outcome in outcomes)),
    )


def assign(env: Env, assignments: Dict[str, str]) -> Env:
    if not assignments:
        return env
    env = dict(env)
    # sequential, as codegen emits them: a later assignment reads an earlier one's target
    for name, text in assignments.items():
        read = [env[read_name] for read_name in sorted(names_read_by_text(text)) if read_name in env]
        env[name] = merge(*read, Fact(via=(f"{name} = {text}",)))
    return env


def kernel_fact(node: ExternalCall, connector: str) -> Fact:
    return Fact((Reach(Producer("kernel", node.label, connector.removeprefix(OUTPUT_PREFIX))),))


def source_fact(state: SDFGState, edge: MultiConnectorEdge, facts: Dict[int, Fact]) -> Fact:
    source = edge.src
    if isinstance(source, ExternalCall) and edge.src_conn is not None and edge.src_conn.startswith(OUTPUT_PREFIX):
        return kernel_fact(source, edge.src_conn)
    if isinstance(source, nodes.AccessNode):
        # an AccessNode -> AccessNode copy forwards its source, so offload copies stay invisible
        return facts[id(source)]
    return Fact((Reach(Producer("host", state.label)),))


@dataclass(frozen=True, slots=True)
class Views:
    # ids of the edges binding a view to what it views, and each view node's root array
    bindings: Dict[int, None]
    roots: Dict[int, str]


def state_views(state: SDFGState) -> Views:
    bindings: Dict[int, None] = {}
    roots: Dict[int, str] = {}
    for node in state.data_nodes():
        if not isinstance(node.desc(state.sdfg), dace.data.View):
            continue
        binding = sdutil.get_view_edge(state, node)
        root = sdutil.get_last_view_node(state, node)
        if binding is None or root is None:
            raise UnsupportedProgram(f"view {node.data!r} in state {state.label!r} binds no container")
        bindings[id(binding)] = None
        roots[id(node)] = root.data
    return Views(bindings, roots)


def access_flow(
    state: SDFGState, node: nodes.AccessNode, views: Views, env: Env, facts: Dict[int, Fact], tracker: Tracker
) -> None:
    # a view reads and writes its root array; the binding edge itself moves no data
    container = views.roots.get(id(node), node.data)
    writes = [edge for edge in state.in_edges(node) if not edge.data.is_empty() and id(edge) not in views.bindings]
    if not writes:
        facts[id(node)] = env.get(container, EMPTY)
        return
    fact = merge(*(source_fact(state, edge, facts) for edge in writes))
    facts[id(node)] = fact
    env[container] = fact
    tracker.written[container] = None


def kernel_symbols(node: ExternalCall) -> List[str]:
    # read once: a dace Property; dace.library.node erases the class type, hiding the property from pyright
    manifest = node.config  # pyright: ignore[reportAttributeAccessIssue]
    if manifest is None:
        raise UnsupportedProgram(f"ExternalCall {node.label!r} has no manifest, so its symbol arguments are unknown")
    # the manifest's non-array inputs are the symbols the kernel takes, body-only ones included
    return sorted(arg for arg in manifest["input_args"] if arg not in manifest["array_args"])


def record(tracker: Tracker, consumer: str, arg: str, role: str, fact: Fact) -> None:
    tracker.edges[(consumer, role, arg)] = ArgEdge(consumer, arg, role, fact.reaches, fact.via)


def kernel_flow(state: SDFGState, node: ExternalCall, env: Env, facts: Dict[int, Fact], tracker: Tracker) -> None:
    tracker.kernels[node.label] = None
    for edge in state.in_edges(node):
        connector = edge.dst_conn
        if connector is None or not connector.startswith(INPUT_PREFIX):
            continue
        record(tracker, node.label, connector.removeprefix(INPUT_PREFIX), "input", source_fact(state, edge, facts))
    for name in kernel_symbols(node):
        record(tracker, node.label, name, "symbol", env.get(name, EMPTY))


def state_flow(state: SDFGState, env: Env, tracker: Tracker) -> Env:
    env = dict(env)
    facts: Dict[int, Fact] = {}
    views = state_views(state)
    for node in in_order(state):
        if isinstance(node, nodes.AccessNode):
            access_flow(state, node, views, env, facts, tracker)
        elif isinstance(node, ExternalCall):
            kernel_flow(state, node, env, facts, tracker)
    return env


def block_input(
    region: ControlFlowRegion, block: ControlFlowBlock, entry: Env, edge_envs: Dict[int, Optional[Env]]
) -> Optional[Env]:
    arriving = [edge_envs.get(id(edge)) for edge in region.in_edges(block)]
    return join(entry if block is region.start_block else None, *arriving)


def region_exit(region: ControlFlowRegion, blocks: List[ControlFlowBlock], outcomes: Dict[int, Outcome]) -> Outcome:
    reached = [(block, outcomes[index]) for index, block in enumerate(blocks) if index in outcomes]
    sinks = [outcome.normal for block, outcome in reached if region.out_degree(block) == 0]
    return join_outcomes([outcome for _, outcome in reached], join(*sinks))


def region_flow(region: ControlFlowRegion, entry: Env, tracker: Tracker) -> Outcome:
    blocks = in_order(region)
    if not blocks:
        return Outcome(normal=entry)
    rank = {id(block): index for index, block in enumerate(blocks)}
    start = rank[id(region.start_block)]
    inputs: Dict[int, Env] = {start: entry}
    outcomes: Dict[int, Outcome] = {}
    edge_envs: Dict[int, Optional[Env]] = {}
    queue, queued = [start], {start: None}
    # worklist in block order: a block reruns whenever its input grows, so its last visit sees the fixpoint
    while queue:
        index = heapq.heappop(queue)
        del queued[index]
        block = blocks[index]
        outcome = block_flow(block, inputs[index], tracker)
        outcomes[index] = outcome
        for edge in region.out_edges(block):
            leaving = outcome.normal
            edge_envs[id(edge)] = None if leaving is None else assign(leaving, edge.data.assignments)
            target = rank[id(edge.dst)]
            arriving = block_input(region, edge.dst, entry, edge_envs)
            if arriving is None or (target in inputs and inputs[target] == arriving):
                continue
            inputs[target] = arriving
            if target not in queued:
                queued[target] = None
                heapq.heappush(queue, target)
    return region_exit(region, blocks, outcomes)


def carry(back: Env, incoming: Env, label: str) -> Env:
    carried: Env = {}
    for name, fact in back.items():
        known = incoming.get(name, EMPTY).reaches
        # a reach that also enters from outside the loop is the same value, not a carried one
        fresh = [reach for reach in fact.reaches if reach not in known]
        tagged = tuple(
            reach if label in reach.carried_by else Reach(reach.producer, (*reach.carried_by, label)) for reach in fresh
        )
        carried[name] = Fact(tagged, fact.via)
    return carried


def loop_assign(loop: LoopRegion, statement: Optional[CodeBlock], env: Env) -> Env:
    text = loop_analysis.assignment_text(statement, loop.loop_variable)
    return env if text is None else assign(env, {loop.loop_variable: text})


def loop_flow(loop: LoopRegion, env: Env, tracker: Tracker) -> Outcome:
    incoming = loop_assign(loop, loop.init_statement, env)
    back: Optional[Env] = None
    while True:
        head = incoming if back is None else union(incoming, carry(back, incoming, loop.label))
        body = region_flow(loop, head, tracker)
        ending = join(body.normal, body.continues)
        latch = None if ending is None else loop_assign(loop, loop.update_statement, ending)
        if latch == back:
            break
        back = latch
    leaving = join(latch, body.breaks)
    if not loop_analysis.loop_provably_at_least_one_iteration(loop):
        leaving = join(leaving, incoming)
    return Outcome(normal=leaving, returns=body.returns)


def conditional_flow(block: ConditionalBlock, env: Env, tracker: Tracker) -> Outcome:
    outcomes = [region_flow(branch, env, tracker) for _, branch in block.branches]
    has_else = any(condition is None for condition, _ in block.branches)
    normal = join(*(outcome.normal for outcome in outcomes), None if has_else else env)
    return join_outcomes(outcomes, normal)


def block_flow(block: ControlFlowBlock, env: Env, tracker: Tracker) -> Outcome:
    if isinstance(block, SDFGState):
        return Outcome(normal=state_flow(block, env, tracker))
    if isinstance(block, BreakBlock):
        return Outcome(breaks=env)
    if isinstance(block, ContinueBlock):
        return Outcome(continues=env)
    if isinstance(block, ReturnBlock):
        return Outcome(returns=env)
    if isinstance(block, LoopRegion):
        return loop_flow(block, env, tracker)
    if isinstance(block, ConditionalBlock):
        return conditional_flow(block, env, tracker)
    if isinstance(block, ControlFlowRegion):
        return region_flow(block, env, tracker)
    raise UnsupportedProgram(f"control-flow block {block.label!r} ({type(block).__name__}) is not modeled")


def refuse_unsupported(sdfg: dace.SDFG) -> None:
    for sd in sdfg.all_sdfgs_recursive():
        for name, desc in sd.arrays.items():
            if isinstance(desc, dace.data.Reference):
                raise UnsupportedProgram(
                    f"container {name!r} of SDFG {sd.name!r} is a Reference: its target is bound at run time, "
                    "so no producer can be named for it"
                )
        if sd is sdfg:
            continue
        for state in sd.all_states():
            kernel = next((node for node in state.nodes() if isinstance(node, ExternalCall)), None)
            if kernel is not None:
                raise UnsupportedProgram(
                    f"ExternalCall {kernel.label!r} sits inside nested SDFG {sd.name!r}: kernels are analyzed "
                    "in the top-level SDFG only"
                )


def kernel_dependencies(sdfg: dace.SDFG) -> KernelGraph:
    """The producers reaching every ``ExternalCall`` argument of ``sdfg``, and every program output at exit.

    :param sdfg: The top-level SDFG holding the kernels; it is not modified.
    :returns: The kernel graph, sorted by kernel program order, then role, then argument name.
    :raises UnsupportedProgram: On a ``Reference`` container or an ``ExternalCall`` inside a nested SDFG.
    """
    refuse_unsupported(sdfg)
    tracker = Tracker()
    names = [name for name, desc in sdfg.arrays.items() if not desc.transient] + sorted(sdfg.free_symbols)
    outcome = region_flow(sdfg, dict.fromkeys(names, PROGRAM), tracker)
    final = join(outcome.normal, outcome.returns) or {}
    order = {kernel: index for index, kernel in enumerate(tracker.kernels)}
    edges = sorted(tracker.edges.values(), key=lambda edge: (order[edge.consumer], ROLE_ORDER[edge.role], edge.arg))
    exits = []
    for name in sorted(tracker.written):
        if not sdfg.arrays[name].transient:
            fact = final.get(name, EMPTY)
            exits.append(ArgEdge("exit", name, "input", fact.reaches, fact.via))
    return KernelGraph(tuple(tracker.kernels), tuple(edges), tuple(exits))
