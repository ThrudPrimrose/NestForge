# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The numpy oracle must reproduce a kernel's precondition traps, not ignore them.

Guards are built by the DaCe passes that really produce them, never by a hand-shaped stand-in.
"""

import ast

import dace
import numpy as np
import pytest

from dace.sdfg import nodes
from dace.transformation.passes.canonicalize.assume_symbols_nonnegative import (
    collect_assumptions,
    insert_assumption_guards,
)

from nestforge.ir.emit_numpy import UnsupportedNest, load_emitted, sdfg_to_numpy, trap_guard_lines

N = dace.symbol("N")


@dace.program
def scaled(a: dace.float64[N], b: dace.float64[N]):
    for i in dace.map[0:N]:
        b[i] = a[i] * 2.0


def compiled(src, name):
    """Import the emitted source and hand back the callable, as the differential harness does."""
    return vars(load_emitted(src, name))[name]


def guard_tasklets(sdfg):
    return [
        n
        for s in sdfg.states()
        for n in s.nodes()
        if isinstance(n, nodes.Tasklet) and not n.in_connectors and not n.out_connectors
    ]


def test_canonicalize_assumption_guard_is_emitted_as_a_python_assertion():
    """The real pass's guard survives into the oracle as an equivalent check, and the kernel still runs."""
    sdfg = scaled.to_sdfg(simplify=True)
    assert collect_assumptions(sdfg) == [N >= 0], "test is vacuous unless the pass has an assumption to guard"
    assert insert_assumption_guards(sdfg) == 1
    guards = guard_tasklets(sdfg)
    assert len(guards) == 1 and guards[0].code.language is dace.dtypes.Language.CPP, guards

    src = sdfg_to_numpy(sdfg, "k")
    ast.parse(src)
    assert "raise AssertionError" in src, src

    a = np.arange(6, dtype=np.float64)
    b = np.zeros(6)
    compiled(src, "k")(a, b, 6)
    np.testing.assert_allclose(b, a * 2.0)


def test_the_emitted_guard_actually_trips_on_a_violated_assumption():
    """Not decoration: the oracle refuses the same inputs the compiled kernel would trap on."""
    sdfg = scaled.to_sdfg(simplify=True)
    insert_assumption_guards(sdfg)
    kernel = compiled(sdfg_to_numpy(sdfg, "k"), "k")
    with pytest.raises(AssertionError):
        kernel(np.zeros(4), np.zeros(4), -1)  # N < 0 -- __builtin_trap() in the compiled kernel


def test_a_guard_outside_the_canonicalize_guard_state_is_translated_too():
    """scatter_conflict_guard emits the same tasklet under its own state label. Matching on the
    canonicalize label -- what the emitter used to do -- missed it and failed the nest as 'not Python'."""
    sdfg = dace.SDFG("scatter_guard")
    sdfg.add_array("a", [N], dace.float64)
    sdfg.add_symbol("overlap", dace.int64)
    trap_state = sdfg.add_state("_scatter_guard_trap_a", is_start_block=True)
    trap = trap_state.add_tasklet(
        "check_assumption_a", {}, {}, "if ((overlap > 0)) { __builtin_trap(); }", language=dace.dtypes.Language.CPP
    )
    trap.side_effects = True
    work = sdfg.add_state_after(trap_state, "work")
    tasklet = work.add_tasklet("w", {"i_a"}, {"o_a"}, "o_a = i_a + 1.0")
    work.add_edge(work.add_read("a"), None, tasklet, "i_a", dace.Memlet("a[0]"))
    work.add_edge(tasklet, "o_a", work.add_write("a"), None, dace.Memlet("a[0]"))
    sdfg.validate()

    src = sdfg_to_numpy(sdfg, "k")
    ast.parse(src)
    assert "if (overlap > 0):" in src, src

    a = np.zeros(3)
    compiled(src, "k")(a, overlap=0)
    assert a[0] == 1.0
    with pytest.raises(AssertionError):
        compiled(src, "k")(np.zeros(3), overlap=1)


@pytest.mark.parametrize(
    "c_cond, py_cond",
    [
        ("(N < 0)", "(N < 0)"),
        ("(N < 0) && (M < 0)", "(N < 0) and (M < 0)"),
        ("(N < 0) || (M != 3)", "(N < 0) or (M != 3)"),
        ("!(N == 0)", "not (N == 0)"),
    ],
)
def test_c_operators_become_python_operators(c_cond, py_cond):
    """``!=`` must survive the ``!`` rewrite; ``not =`` would be a SyntaxError."""
    state = dace.SDFG("g").add_state()
    trap = state.add_tasklet(
        "check_assumption_0", {}, {}, f"if ({c_cond}) {{ __builtin_trap(); }}", language=dace.dtypes.Language.CPP
    )
    assert trap_guard_lines(trap)[0] == f"if {py_cond}:"


def test_a_connectorless_tasklet_that_is_not_a_guard_emits_no_statement():
    """No connectors means no data effect -- recorded, not silently dropped."""
    state = dace.SDFG("g").add_state()
    other = state.add_tasklet("bookkeeping", {}, {}, 'printf("hi");', language=dace.dtypes.Language.CPP)
    assert trap_guard_lines(other) is None
    from nestforge.ir.emit_numpy import tasklet_lines

    assert tasklet_lines(state, state.sdfg, other) == ["# no-op tasklet (bookkeeping): no connectors, no data effect"]


def test_an_untranslatable_guard_condition_is_refused():
    """Fail at emission, where the tasklet name is still known."""
    state = dace.SDFG("g").add_state()
    trap = state.add_tasklet(
        "check_assumption_0", {}, {}, "if (a ? b : c) { __builtin_trap(); }", language=dace.dtypes.Language.CPP
    )
    with pytest.raises(UnsupportedNest, match="not translatable to python"):
        trap_guard_lines(trap)
