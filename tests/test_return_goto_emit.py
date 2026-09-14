# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Early-return (``ReturnBlock``) emission and the non-externalizable control-flow refusals.

Three contracts:

  * a ``ReturnBlock`` in a **whole** SDFG (:func:`sdfg_to_numpy`) emits a python ``return`` and
    short-circuits -- ``return`` exits the kernel function, which is exactly SDFG-return semantics;
  * a nest carrying a ``ReturnBlock`` is **not externalizable** (:func:`nest_to_numpy`): the return
    would exit only the extracted kernel, not the enclosing SDFG the original return targeted;
  * an **unstructured goto** (a conditional inter-state edge, i.e. a state-machine branch DaCe did not
    lift into a ``ConditionalBlock``) is refused rather than emitted as straight-line code.
"""
import numpy as np
import pytest

pytest.importorskip("dace")

import dace as dc

from dace.sdfg.state import BreakBlock, ConditionalBlock, ControlFlowRegion, LoopRegion, ReturnBlock

from nestforge.ir.emit_numpy import (UnsupportedNest, load_emitted, nest_to_numpy, reject_nonexternalizable, sdfg_to_numpy)
from nestforge.ir.extract import Boundary

N = dc.symbol("N", dtype=dc.int64)


def build_early_return():
    """``out[:] = a`` then ``if sel > 0: return`` else ``out += 1`` -- a structured early return where the
    return lives in a ``ConditionalBlock`` branch (all inter-state edges stay unconditional)."""
    sdfg = dc.SDFG("earlyret")
    sdfg.add_array("a", [N], dc.float64)
    sdfg.add_array("out", [N], dc.float64)
    sdfg.add_symbol("sel", dc.int64)
    s0 = sdfg.add_state("s0", is_start_block=True)
    s0.add_nedge(s0.add_read("a"), s0.add_write("out"), dc.Memlet("a[0:N]"))
    cond = ConditionalBlock("cond", sdfg=sdfg)
    sdfg.add_node(cond)
    branch = ControlFlowRegion("retbranch", sdfg=sdfg)
    branch.add_node(ReturnBlock("ret", sdfg=branch), is_start_block=True)
    cond.add_branch("sel > 0", branch)
    s1 = sdfg.add_state("s1")
    t = s1.add_tasklet("inc", {"i"}, {"o"}, "o = i + 1.0")
    s1.add_edge(s1.add_read("out"), None, t, "i", dc.Memlet("out[0:N]"))
    s1.add_edge(t, "o", s1.add_write("out"), None, dc.Memlet("out[0:N]"))
    sdfg.add_edge(s0, cond, dc.InterstateEdge())
    sdfg.add_edge(cond, s1, dc.InterstateEdge())
    sdfg.validate()
    return sdfg


def test_early_return_whole_sdfg_emits_and_short_circuits():
    src = sdfg_to_numpy(build_early_return(), "earlyret")
    assert "return" in src
    fn = load_emitted(src, "earlyret").earlyret
    a = np.arange(5.0)
    for sel, expected in ((1, a.copy()), (0, a + 1.0)):  # sel>0 returns before the increment
        out = np.zeros(5)
        fn(a=a.copy(), out=out, sel=sel, N=5)
        np.testing.assert_array_equal(out, expected)


def test_return_nest_is_not_externalizable():
    """Externalizing a nest that carries a return changes its target (nest-function vs enclosing SDFG),
    so :func:`nest_to_numpy` refuses it up front."""
    boundary = Boundary(inputs=[],
                        outputs=[],
                        symbols=[],
                        nsdfg_node=None,
                        state=None,
                        standalone_sdfg=build_early_return())
    with pytest.raises(UnsupportedNest, match="early return"):
        nest_to_numpy(boundary, "k")


def build_loop_with_break():
    """``for i in range(N): body; if sel > 0: break`` -- a break whose target loop (``L``) is present, so
    the whole nest IS externalizable."""
    sdfg = dc.SDFG("loopbreak")
    sdfg.add_array("out", [N], dc.float64)
    sdfg.add_symbol("sel", dc.int64)
    loop = LoopRegion("L",
                      condition_expr="i < N",
                      loop_var="i",
                      initialize_expr="i = 0",
                      update_expr="i = i + 1",
                      sdfg=sdfg)
    sdfg.add_node(loop, is_start_block=True)
    body = loop.add_state("body", is_start_block=True)
    body.add_nedge(body.add_read("out"), body.add_write("out"), dc.Memlet("out[0:N]"))
    cond = ConditionalBlock("c", sdfg=sdfg)
    loop.add_node(cond)
    loop.add_edge(body, cond, dc.InterstateEdge())
    branch = ControlFlowRegion("bregion", sdfg=sdfg)
    branch.add_node(BreakBlock("brk", sdfg=branch), is_start_block=True)
    cond.add_branch("sel > 0", branch)
    sdfg.validate()
    return sdfg


def build_orphan_break():
    """A ``break`` inside a top-level ``ConditionalBlock`` with NO enclosing loop -- the illegal cut a bad
    strategy could produce (extracting an inner scope while leaving the break's loop behind)."""
    sdfg = dc.SDFG("orphanbreak")
    sdfg.add_array("out", [N], dc.float64)
    sdfg.add_symbol("sel", dc.int64)
    cond = ConditionalBlock("c", sdfg=sdfg)
    sdfg.add_node(cond, is_start_block=True)
    branch = ControlFlowRegion("bregion", sdfg=sdfg)
    branch.add_node(BreakBlock("brk", sdfg=branch), is_start_block=True)
    cond.add_branch("sel > 0", branch)
    sdfg.validate()
    return sdfg


def test_break_with_enclosing_loop_is_externalizable():
    """A break whose target loop is inside the nest passes the guard (the ``ext_break_find_first`` shape)."""
    reject_nonexternalizable(build_loop_with_break())  # must not raise


def test_orphan_break_is_not_externalizable():
    """A break whose target loop was cut out lands outside any ``while`` -- refuse rather than emit a
    ``break`` at function-body top level (a SyntaxError / wrong target)."""
    boundary = Boundary(inputs=[],
                        outputs=[],
                        symbols=[],
                        nsdfg_node=None,
                        state=None,
                        standalone_sdfg=build_orphan_break())
    with pytest.raises(UnsupportedNest, match="target loop is outside"):
        nest_to_numpy(boundary, "k")


def test_unstructured_goto_refused():
    """A conditional inter-state edge is an unstructured branch the straight-line emission cannot model;
    emitting the successors as if always-taken would be a miscompile, so it is refused."""
    sdfg = dc.SDFG("goto")
    sdfg.add_array("out", [N], dc.float64)
    sdfg.add_symbol("sel", dc.int64)
    a0 = sdfg.add_state("a0", is_start_block=True)
    a1 = sdfg.add_state("a1")
    a2 = sdfg.add_state("a2")
    sdfg.add_edge(a0, a1, dc.InterstateEdge(condition="sel > 0"))
    sdfg.add_edge(a0, a2, dc.InterstateEdge(condition="sel <= 0"))
    with pytest.raises(UnsupportedNest, match="goto"):
        sdfg_to_numpy(sdfg, "goto")
