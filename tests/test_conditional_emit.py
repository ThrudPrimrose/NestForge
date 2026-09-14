# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Emit ``ConditionalBlock`` control flow (if / elif / else) to numpy and check it runs correctly.

A DaCe scalar-level ``if/elif/else`` at control-flow altitude lowers to a ``ConditionalBlock`` whose
branches are separate regions. The emitter turns that block back into a python ``if/elif/else`` with
one branch body per region; these tests build such SDFGs directly and validate both the structure
(the emitted keywords) and the numerics (the taken branch matches numpy).
"""

import inspect

import numpy as np
import pytest

import dace as dc

from dace.sdfg.state import ConditionalBlock, ControlFlowRegion

from nestforge.ir.emit_numpy import load_emitted, sdfg_to_numpy

N = dc.symbol("N", dtype=dc.int64)


@dc.program
def if_else(flag: dc.int64, a: dc.float64[N], out: dc.float64[N]):
    if flag > 0:
        for i in dc.map[0:N]:
            out[i] = a[i] + 1.0
    else:
        for i in dc.map[0:N]:
            out[i] = a[i] - 1.0


def build_three_branch():
    """A 3-branch ``ConditionalBlock`` (``if``/``elif``/``else``) built directly via the SDFG API.

    The python frontend lowers ``if/elif/else`` into *nested* 2-branch blocks, so a genuine
    multi-branch ``ConditionalBlock`` -- the case that exercises the emitter's ``elif`` -- has to be
    assembled by hand. Each branch is ``out[i] = a[i] + delta`` for a distinct delta.
    """
    sdfg = dc.SDFG("switch3")
    sdfg.add_array("a", [N], dc.float64)
    sdfg.add_array("out", [N], dc.float64)
    sdfg.add_symbol("sel", dc.int64)
    cond = ConditionalBlock("cond", sdfg=sdfg)
    sdfg.add_node(cond, is_start_block=True)
    for label, guard, delta in (("b0", "sel == 0", 10.0), ("b1", "sel == 1", 20.0), ("belse", None, 30.0)):
        region = ControlFlowRegion(label, sdfg=sdfg)
        st = region.add_state(label + "_s", is_start_block=True)
        st.add_mapped_tasklet(
            label + "_m",
            dict(i="0:N"),
            dict(inp=dc.Memlet("a[i]")),
            f"o = inp + {delta}",
            dict(o=dc.Memlet("out[i]")),
            input_nodes={"a": st.add_read("a")},
            output_nodes={"out": st.add_write("out")},
            external_edges=True,
        )
        cond.add_branch(guard, region)
    sdfg.validate()
    return sdfg


def build_switch(name, branches):
    """A ``ConditionalBlock`` with the given ``(guard, delta)`` branches in the given order.

    ``guard is None`` marks the unconditional (else) branch; each branch is ``out[i] = a[i] + delta``.
    Lets a test control branch ordering (e.g. the unconditional branch stored first).
    """
    sdfg = dc.SDFG(name)
    sdfg.add_array("a", [N], dc.float64)
    sdfg.add_array("out", [N], dc.float64)
    sdfg.add_symbol("sel", dc.int64)
    cond = ConditionalBlock("cond", sdfg=sdfg)
    sdfg.add_node(cond, is_start_block=True)
    for i, (guard, delta) in enumerate(branches):
        region = ControlFlowRegion(f"b{i}", sdfg=sdfg)
        st = region.add_state(f"b{i}_s", is_start_block=True)
        st.add_mapped_tasklet(
            f"b{i}_m",
            dict(i="0:N"),
            dict(inp=dc.Memlet("a[i]")),
            f"o = inp + {delta}",
            dict(o=dc.Memlet("out[i]")),
            input_nodes={"a": st.add_read("a")},
            output_nodes={"out": st.add_write("out")},
            external_edges=True,
        )
        cond.add_branch(guard, region)
    sdfg.validate()
    return sdfg


def emit(sdfg, fn_name):
    assert any(isinstance(b, ConditionalBlock) for b in sdfg.all_control_flow_blocks())
    src = sdfg_to_numpy(sdfg, fn_name)
    return vars(load_emitted(src, fn_name))[fn_name], src


def test_if_else_both_branches():
    fn, src = emit(if_else.to_sdfg(simplify=True), "if_else")
    assert "if " in src and "else:" in src
    a = np.arange(6, dtype=np.float64)
    for flag in (1, -1):
        out = np.zeros(6)
        fn(flag=flag, a=a.copy(), out=out, N=6)
        np.testing.assert_array_equal(out, a + 1.0 if flag > 0 else a - 1.0)


def test_if_elif_else_selects_right_branch():
    fn, src = emit(build_three_branch(), "switch3")
    assert "elif " in src  # a genuine multi-branch block -> the middle branch is an elif
    a = np.arange(5, dtype=np.float64)
    for sel, delta in ((0, 10.0), (1, 20.0), (2, 30.0)):
        out = np.zeros(5)
        fn(a=a.copy(), out=out, sel=sel, N=5)
        np.testing.assert_array_equal(out, a + delta)


def test_signature_is_c_style_no_return():
    fn, src = emit(if_else.to_sdfg(simplify=True), "if_else")
    assert "return " not in src  # C-style: outputs are in-place buffer params
    assert set(inspect.signature(fn).parameters) == {"a", "out", "flag", "N"}


def test_branch_condition_is_normalized():
    """A guard that uses a bare math intrinsic must be normalized like every other emitted expression,
    else the kernel references an unqualified ``sqrt`` and raises NameError at exec."""
    fn, src = emit(build_switch("cond_norm", [("sqrt(sel) > 1.0", 10.0), (None, 20.0)]), "cond_norm")
    assert "np.sqrt(sel)" in src and "sqrt(sel)" not in src.replace("np.sqrt(sel)", "")  # normalized
    a = np.arange(4, dtype=np.float64)
    for sel, delta in ((4, 10.0), (0, 20.0)):  # sqrt(4)=2>1 -> if; sqrt(0)=0 -> else
        out = np.zeros(4)
        fn(a=a.copy(), out=out, sel=sel, N=4)
        np.testing.assert_array_equal(out, a + delta)


def test_a_non_final_unconditional_branch_is_refused():
    """An unconditional branch stored before a keyed one is REFUSED, because DaCe refuses it too.

    This test used to assert the opposite -- that the emitter hoists the unconditional branch to a
    trailing ``else`` -- and that was wrong. DaCe's own codegen raises ``Missing branch condition for
    non-final conditional branch`` on this SDFG, so it never runs; the reordering therefore compared
    the emitted kernel against semantics no DaCe build has. Worse, it made a keyed branch stored AFTER
    the unconditional one live (DaCe takes the first matching branch, and an unconditional one always
    matches), and two unconditional branches emitted two ``else:`` clauses -- a SyntaxError.
    """
    from nestforge.ir.emit_numpy import UnsupportedNest

    sdfg = build_switch("else_first", [(None, 20.0), ("sel == 0", 10.0)])
    with pytest.raises(UnsupportedNest, match="unconditional branch"):
        sdfg_to_numpy(sdfg, "else_first")


def test_a_final_unconditional_branch_is_the_else():
    """The legal shape: keyed branches first, the unconditional one last."""
    fn, src = emit(build_switch("else_last", [("sel == 0", 10.0), (None, 20.0)]), "else_last")
    assert src.index("if ") < src.index("else:")
    a = np.arange(4, dtype=np.float64)
    for sel, delta in ((0, 10.0), (5, 20.0)):
        out = np.zeros(4)
        fn(a=a.copy(), out=out, sel=sel, N=4)
        np.testing.assert_array_equal(out, a + delta)
