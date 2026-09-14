# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The ``ExternalCall`` library node and its expansions.

Payload (numpy source + manifest config + the chosen compiled lib) rides on the node as
properties. Two expansions:
  * ``ExpandDaceReference`` (default) -- rebuilds the extracted nest as a NestedSDFG (the DaCe
    competitor and the correctness fallback);
  * ``ExpandExternCall`` -- a CPP tasklet calling the extern-C entry of the chosen ``.so``,
    linked via a per-node environment.
"""
from __future__ import annotations

import copy
import os
from typing import List, Optional, Tuple

import numpy as np

import dace
import dace.library
import dace.properties
from dace import dtypes
from dace.sdfg import nodes
from dace.transformation.transformation import ExpandTransformation

_CPP_SCALAR = {"float64": "double", "float32": "float", "int64": "int64_t", "int32": "int32_t"}


def in_conn(name: str) -> str:
    """Connector name for an input array (kept distinct from the array name itself)."""
    return f"_in_{name}"


def out_conn(name: str) -> str:
    """Connector name for an output array."""
    return f"_out_{name}"


def connector_for(arg: str, outputs: set) -> str:
    return out_conn(arg) if arg in outputs else in_conn(arg)


def value_connectors(node: "ExternalCall", state: dace.SDFGState) -> set:
    """Connectors DaCe defines as a VALUE, not a pointer: those whose memlet covers one element.

    A ``double x`` passed where the compiled signature wants ``double*`` is a compile error, and every
    scalar transient crossing the boundary (symbol write-backs, fused index/reduction scalars) is exactly
    this case -- so the call must take their address.
    """
    single = set()
    for edge in state.in_edges(node):
        if edge.dst_conn is not None and bool(edge.data.subset) and edge.data.subset.num_elements() == 1:
            single.add(edge.dst_conn)
    for edge in state.out_edges(node):
        # dace.sdfg.infer_types adds the dynamic clause on the OUT side only; a dynamic non-WCR output
        # stays a pointer, and taking its address would be wrong.
        dynamic_pointer = edge.data.dynamic and edge.data.wcr is None
        if (edge.src_conn is not None and bool(edge.data.subset) and edge.data.subset.num_elements() == 1
                and not dynamic_pointer):
            single.add(edge.src_conn)
    return single


def proto_and_call(node: "ExternalCall", state: dace.SDFGState) -> Tuple[str, str]:
    """Build the ``extern "C"`` prototype and the call expression for the linked kernel.

    Array args are passed by their connector variable (``_in_X`` / ``_out_Y``); size symbols are
    referenced by name (available in the tasklet's symbol scope).

    The parameter order is ``node.abi_order`` -- the order the .so was ACTUALLY compiled with, recorded by
    the arena from the emitted signature. It is NOT the manifest's ``input_args`` (a role order: inputs,
    outputs, symbols): numpyto emits ``param_order()`` (arrays sorted, then scalars) and deliberately
    ignores ``input_args``. Declaring the wrong order here is silent -- C linkage matches on the symbol
    NAME alone, so a self-consistent-but-wrong prototype compiles and links cleanly and simply passes each
    buffer to the wrong parameter.

    Scalar C types come from the manifest, not a blanket ``int64_t``: a float value scalar crosses as a
    ``double``, and declaring it ``int64_t`` both truncates it and breaks the SysV register class (GP vs
    XMM), so the callee reads garbage.
    """
    manifest = node.config
    arrays = set(manifest["array_args"])
    outputs = set(manifest["output_args"])
    dtypes_map = {a: v["dtype"] for a, v in manifest["init"]["arrays"].items()}
    # `init.scalars` carries a bare DEFAULT VALUE, not a descriptor -- emit_yaml writes `{name: 0.0}` and
    # relies on the value's type to say "double" (see its comment on float value scalars). The dict form is
    # still accepted, but requiring it made this mapping unconditionally EMPTY, so every float value scalar
    # fell to the 'int64' default below and was declared int64_t: the truncation + SysV register-class
    # break this function's own docstring warns about, with nothing to signal it.
    scalar_dtypes = {
        n: (v["dtype"] if isinstance(v, dict) else np.dtype(type(v)).name)
        for n, v in (manifest["init"].get("scalars") or {}).items()
    }
    by_value = value_connectors(node, state)
    order = list(node.abi_order or [])
    if not order:
        raise ValueError(f"ExternalCall {node.name!r} has no abi_order: the extern-call expansion must declare the "
                         f"linked symbol in the order it was compiled with (the arena records it on the winning "
                         f"Cell). Falling back to the manifest's role order would silently mis-declare the ABI.")
    params: List[str] = []
    call_args: List[str] = []
    in_conns = node.in_connectors  # cache the leaf: a dace Property, re-resolved every access otherwise
    out_conns = node.out_connectors
    for arg in order:
        if arg in arrays:
            dt = dtypes_map[arg]
            if dt not in _CPP_SCALAR:
                # A dtype with no C spelling here (complex, float16, unsigned, ...) would otherwise KeyError
                # mid-codegen. Refuse with the name + dtype so the caller can keep the DaceReference variant.
                raise ValueError(f"ExternalCall {node.name!r}: array {arg!r} has dtype {dt!r}, which has no "
                                 f"extern-C spelling (known: {sorted(_CPP_SCALAR)}); keep the DaceReference "
                                 "implementation for this nest")
            c = _CPP_SCALAR[dt]
            const = "" if arg in outputs else "const "
            conn = connector_for(arg, outputs)
            if conn not in in_conns and conn not in out_conns:
                # The compiled signature takes an argument this node has no connector for -- typically a
                # caller-allocated SCRATCH transient, which the emitted C exposes as a parameter but which
                # never crosses the ExternalCall boundary. Emitting the call anyway would reference an
                # undefined identifier in the tasklet (and pass no buffer at all). Refuse instead.
                raise ValueError(f"ExternalCall {node.name!r}: abi_order names {arg!r}, but the node has no "
                                 f"{conn!r} connector (a caller-allocated scratch buffer is not passed across "
                                 "the ExternalCall boundary); keep the DaceReference implementation")
            params.append(f"{const}{c}* {arg}")
            call_args.append(f"&{conn}" if conn in by_value else conn)
        else:
            params.append(f"{_CPP_SCALAR.get(scalar_dtypes.get(arg, 'int64'), 'int64_t')} {arg}")
            call_args.append(arg)
    proto = f'extern "C" void {node.symbol}({", ".join(params)});'
    call = f'{node.symbol}({", ".join(call_args)});'
    return proto, call


@dace.library.environment
class ExternLibEnv:
    """Links the chosen compiled ``.so`` into the SDFG program.

    The lib path is per-node, so :meth:`configure` stamps it onto the (module-level, hence
    registry-resolvable) class right before expansion. M0 assumes one extern call per build;
    per-node isolation is an M1 concern.
    """
    __slots__ = ()  # never instantiated: consumed purely as a class (dace resolves environments by class)

    cmake_minimum_version = None
    cmake_packages = []
    cmake_variables = {}
    cmake_includes = []
    cmake_libraries = []
    cmake_compile_flags = []
    cmake_link_flags = []
    cmake_files = []
    headers = []
    state_fields = []
    init_code = ""
    finalize_code = ""
    dependencies = []

    @classmethod
    def reset(cls) -> None:
        """Drop every accumulated library. Call before expanding a fresh SDFG so libraries from a PREVIOUS
        build (whose temp directories may already be gone) are not carried onto this link line."""
        cls.cmake_libraries = []
        cls.cmake_link_flags = []

    @classmethod
    def configure(cls, lib_path: str) -> None:
        """ACCUMULATE one nest's library. Every ``ExternalCall`` shares this single environment class, so
        assigning here (rather than appending) kept only the LAST expanded nest's library: on a multi-nest
        kernel every earlier nest's extern-C symbol was then unresolved at link, or the wrong .so was
        called. Entries are deduplicated, so nests sharing one backend library add it once.

        A ``.a`` is linked into the parent .so directly; the generated tasklet references the nest's entry
        symbol, so ld pulls that member in. NOTE a multi-member archive is NOT reliably pulled under DaCe's
        sorted link flags -- prefer a shared library for a multi-nest swap (see ``arena.link_shared``).
        A ``.so`` additionally needs an rpath so the built parent resolves it at load."""
        lib = os.path.abspath(lib_path)
        if lib not in cls.cmake_libraries:
            cls.cmake_libraries = [*cls.cmake_libraries, lib]
        if not lib.endswith(".a"):
            rpath = f"-Wl,-rpath,{os.path.dirname(lib)}"
            if rpath not in cls.cmake_link_flags:
                cls.cmake_link_flags = [*cls.cmake_link_flags, rpath]


@dace.library.expansion
class ExpandDaceReference(ExpandTransformation):
    """Rebuild the extracted nest as a NestedSDFG (DaCe competitor / correctness fallback)."""
    # no __slots__: dace ExpandTransformation (make_properties, __dict__-based)
    environments = []

    @staticmethod
    def expansion(node: "ExternalCall", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG) -> dace.SDFG:
        if node._standalone_sdfg is None:
            raise ValueError(f"ExternalCall {node.name} has no standalone SDFG to fall back to")
        return copy.deepcopy(node._standalone_sdfg)


@dace.library.expansion
class ExpandExternCall(ExpandTransformation):
    """Call the extern-C entry of the chosen compiled ``.so`` from a CPP tasklet."""
    # no __slots__: dace ExpandTransformation (make_properties, __dict__-based)
    environments = []

    @staticmethod
    def expansion(node: "ExternalCall", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG) -> nodes.Tasklet:
        if not node.lib_path or not node.symbol:
            raise ValueError(f"ExternalCall {node.name} needs lib_path + symbol for ExpandExternCall")
        proto, call = proto_and_call(node, parent_state)
        ExternLibEnv.configure(node.lib_path)
        ExpandExternCall.environments = [ExternLibEnv]
        tasklet = nodes.Tasklet(node.name,
                                node.in_connectors,
                                node.out_connectors,
                                call,
                                language=dtypes.Language.CPP,
                                code_global=proto,
                                side_effects=True)
        return tasklet


@dace.library.node
class ExternalCall(nodes.LibraryNode):
    """A loop-/map-nest lowered to an external, separately-compiled call."""
    # no __slots__: dace LibraryNode (make_properties, __dict__-based); also carries _standalone_sdfg,
    # an in-memory-only attr outside the Property schema.

    implementations = {"DaceReference": ExpandDaceReference, "ExternCall": ExpandExternCall}
    default_implementation = "DaceReference"

    numpy_source = dace.properties.Property(dtype=str, default="", desc="numpy reference of the nest")
    config = dace.properties.DictProperty(key_type=str,
                                          value_type=object,
                                          default=None,
                                          desc="OptArena manifest (symbols, shapes, dtypes)")
    symbol = dace.properties.Property(dtype=str, default="", desc="extern-C symbol to call")
    abi_order = dace.properties.ListProperty(element_type=str,
                                             default=[],
                                             desc="parameter order the linked .so was compiled with "
                                             "(the emitted signature order -- NOT the manifest role order)")
    lib_path = dace.properties.Property(dtype=str, default="", desc="compiled static/shared lib")
    fp_mode = dace.properties.Property(dtype=str, default="", desc="winning FP mode")

    def __init__(self,
                 name: str,
                 inputs: Optional[set] = None,
                 outputs: Optional[set] = None,
                 numpy_source: str = "",
                 config: Optional[dict] = None,
                 standalone_sdfg: Optional[dace.SDFG] = None,
                 **kwargs) -> None:
        super().__init__(name, inputs=inputs or set(), outputs=outputs or set(), **kwargs)
        self.numpy_source = numpy_source
        self.config = config
        self._standalone_sdfg = standalone_sdfg  # in-memory only (not serialized in M0)
