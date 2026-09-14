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
from dace.transformation.passes.scatter_conflict_guard import insert_scatter_guard

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
        kernel(np.zeros(4), np.zeros(4), -1)  # N < 0 -- std::abort() in the compiled kernel


def test_a_guard_outside_the_canonicalize_guard_state_is_translated_too():
    """scatter_conflict_guard emits its own trap tasklet under its own state label, not the
    canonicalize guard's label. Matching on the label -- what the emitter used to do -- missed it
    and failed the nest as 'not Python'."""
    sdfg = dace.SDFG("scatter_guard")
    sdfg.add_array("idx", [3], dace.int64)
    sdfg.add_array("out", [1], dace.float64)
    work = sdfg.add_state("work", is_start_block=True)
    tasklet = work.add_tasklet("w", {}, {"o"}, "o = 1.0")
    work.add_edge(tasklet, "o", work.add_write("out"), None, dace.Memlet("out[0]"))
    insert_scatter_guard(sdfg, "idx", elide_if_injective=False)
    sdfg.validate()

    src = sdfg_to_numpy(sdfg, "k")
    ast.parse(src)
    assert "raise AssertionError" in src, src

    out = np.zeros(1)
    compiled(src, "k")(np.array([0, 1, 2], dtype=np.int64), out)
    assert out[0] == 1.0
    with pytest.raises(AssertionError):
        compiled(src, "k")(np.array([0, 1, 1], dtype=np.int64), np.zeros(1))


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
        "check_assumption_0", {}, {}, f"if ({c_cond}) {{ std::abort(); }}", language=dace.dtypes.Language.CPP
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
        "check_assumption_0", {}, {}, "if (a ? b : c) { std::abort(); }", language=dace.dtypes.Language.CPP
    )
    with pytest.raises(UnsupportedNest, match="not translatable to python"):
        trap_guard_lines(trap)
