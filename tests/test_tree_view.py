# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The ASCII tree the agent reads (:func:`nestforge.introspect.describe_graph`).

The tree is the agent's whole view of the program, so its FORMAT is an interface: a change to it
changes what every agent prompt sees. The golden test below pins it in full, so a format change has
to be made deliberately rather than drifting out of an unrelated edit.
"""

import re

import numpy as np

import pytest

import dace as dc

from nestforge.ir import introspect
from nestforge.ir.introspect import describe_graph, interstate_definitions, kernel_body, resolve_scalars
from nestforge.ir.names import normalize_for_tree
from nestforge.session import Session


@dc.program
def shaped(A: dc.float64[20], B: dc.float64[20], out: dc.float64[20]):
    """One map nest, a loop around a conditional, and a scalar statement -- one of each thing the tree
    has a line shape for."""
    for i in dc.map[0:20]:
        B[i] = A[i] + 1.0
    s = B[0] * 2.0
    for i in range(1, 20):
        if A[i] > 0.0:
            out[i] = B[i] * s
        else:
            out[i] = -B[i]


GOLDEN = """\
SDFG 'shaped'
|- state0_0
|  |- kernel1_0  [i0=0:20]  reads=['A'] writes=['B']
|  `- kernel1_1  [__nf_wrap=0:1]  reads=['s0'] writes=['s1']
`- for0_0  i=0:19
   |- state1_0
   `- if1_0
      |- block2_0  when A[i + 1] > 0.0
      |  `- state3_0
      |     |- kernel4_0  [__nf_wrap=0:1]  reads=['s1', 's2'] writes=['s3']
      |     `- kernel4_1  [__nf_wrap=0:1]  reads=['s3'] writes=['out']
      `- block2_1  else
         `- state3_1
            |- kernel4_2  [__nf_wrap=0:1]  reads=['s4'] writes=['s5']
            `- kernel4_3  [__nf_wrap=0:1]  reads=['s5'] writes=['out']
"""


def tree_of(program) -> str:
    sdfg = program.to_sdfg(simplify=True)
    sdfg.name = "shaped"
    normalize_for_tree(sdfg)
    return describe_graph(sdfg)


def test_the_tree_format_is_pinned():
    """If this fails, the format changed. Update GOLDEN only when that change is the intent -- every
    agent prompt reads this shape."""
    assert tree_of(shaped) + "\n" == GOLDEN


def test_every_line_is_a_guide_then_one_labelled_thing():
    for line in tree_of(shaped).splitlines()[1:]:
        assert re.match(
            r"^(\|  |   )*(\|- |`- )"
            r"(state|for|while|if|block|continue|break|return|kernel)\d+_\d+\b",
            line,
        ), line


def test_indentation_tracks_the_level_in_the_label():
    """A line's depth in the tree and the level in its canonical name are the same number, so the
    agent can read either one."""
    for line in tree_of(shaped).splitlines()[1:]:
        guide, body = re.match(r"^((?:\|  |   )*(?:\|- |`- ))(.*)$", line).groups()
        level = int(re.match(r"^[a-z]+(\d+)_", body).group(1))
        assert len(guide) == 3 * (level + 1), f"level {level} at guide width {len(guide)}: {line}"


def test_session_stamps_the_ids_that_act_on_each_line():
    """B1: reading the tree and acting on it use ONE vocabulary. A nest line carries the very handle
    can_fuse/fuse resolve -- not a label the agent has to match against a separate list_nests call."""
    sdfg = shaped.to_sdfg(simplify=True)
    normalize_for_tree(sdfg)
    session = Session(sdfg)
    tree = session.describe()
    nest_ids = re.findall(r"\[(e\d+:nest:\d+)\]", tree)
    assert nest_ids, "no nest line carried a minted handle"
    for hid in nest_ids:
        assert session.resolve(hid, "nest") is not None
    # Two nests off the tree are exactly what can_fuse accepts -- no list_nests call in between.
    assert isinstance(session.can_fuse(nest_ids[0], nest_ids[1]), str)


def test_region_lines_carry_the_stable_descriptive_id_not_a_minted_one():
    """Nothing resolves a ``region`` kind, so minting one would grow the registry on a read-only call
    and hand back an id that raises on the kind guard."""
    sdfg = shaped.to_sdfg(simplify=True)
    normalize_for_tree(sdfg)
    session = Session(sdfg)
    tree = session.describe()
    assert re.search(r"\[region:state0_0\]", tree), tree
    assert not re.search(r"\[e\d+:region:", tree)


# --- conditions read as the arrays they test -------------------------------------------------------


def test_a_condition_names_the_array_it_really_reads():
    """The frontend hoists the scalar read to an interstate assignment (`A_index = A[1 + i]`) and the
    branch then tests a name that means nothing on its own."""
    tree = tree_of(shaped)
    assert "when A[i + 1] > 0.0" in tree, tree
    assert "A_index" not in tree


def test_a_name_with_two_definitions_is_left_alone():
    """Which definition reaches a block depends on the path taken, so folding either one in would show
    a condition the program does not always evaluate."""
    sdfg = shaped.to_sdfg(simplify=True)
    normalize_for_tree(sdfg)
    assert "A_index" in interstate_definitions(sdfg), "fixture no longer hoists the scalar read"
    # Give the same name a second, different definition on another edge -- as two branches assigning
    # one variable do.
    edge = next(
        e
        for cfg in sdfg.all_control_flow_regions(recursive=True)
        for e in cfg.edges()
        if "A_index" in e.data.assignments
    )
    other = next(e for cfg in sdfg.all_control_flow_regions(recursive=True) for e in cfg.edges() if e is not edge)
    other.data.assignments["A_index"] = "A[0]"
    assert "A_index" not in interstate_definitions(sdfg)
    assert "A_index" in describe_graph(sdfg), "an ambiguous name must stay unresolved, not vanish"


def test_resolution_terminates_on_a_self_referential_definition():
    """`i = i + 1` on a back edge is ordinary; substituting it forever is not."""
    assert resolve_scalars("i < N", {"i": "i + 1"}) == "i + 1 < N"


def test_resolution_follows_a_chain_to_the_array():
    assert resolve_scalars("c > 0", {"c": "b * 2", "b": "A[k]"}) == "A[k] * 2 > 0"


def test_an_unparsable_condition_is_passed_through():
    assert resolve_scalars("this is not python", {}) == "this is not python"


# --- bodies: what each kernel computes -------------------------------------------------------------


@dc.program
def nested_maps(A: dc.float64[8, 4], B: dc.float64[8, 4]):
    """An outer kernel whose body is another kernel."""
    for i in dc.map[0:8]:
        for j in dc.map[0:4]:
            B[i, j] = A[i, j] * 2.0


def with_bodies(program) -> str:
    sdfg = program.to_sdfg(simplify=True)
    normalize_for_tree(sdfg)
    return describe_graph(sdfg, bodies=True)


def test_bodies_are_off_by_default():
    """An emit per kernel is not free, and the structure alone is what a fusion decision needs."""
    assert not [line for line in tree_of(shaped).splitlines() if introspect.BODY in line]
    assert [line for line in with_bodies(shaped).splitlines() if introspect.BODY in line], "nothing to be off"


def test_a_leaf_kernel_prints_what_it_computes():
    tree = with_bodies(shaped)
    assert f"{introspect.BODY}B[i0] = " in tree, tree
    for line in tree.splitlines():
        if introspect.BODY in line:
            assert line.split(introspect.BODY, 1)[0].strip("| `") == "", "a body line must sit under its kernel"


def test_a_body_does_not_repeat_its_headers():
    """The kernel line already shows the domain those `for` headers iterate."""
    for line in with_bodies(shaped).splitlines():
        if introspect.BODY in line:
            assert not line.split(introspect.BODY, 1)[1].startswith("for "), line


def test_a_kernel_containing_a_kernel_has_no_body_of_its_own():
    """`map_body_lines` recurses into a nested map, and the tree already gives that map its own row,
    so emitting at both levels would print the inner kernel twice."""
    sdfg = nested_maps.to_sdfg(simplify=True)
    normalize_for_tree(sdfg)
    nested = [
        (st, n)
        for st in sdfg.all_states()
        for n in st.nodes()
        if isinstance(n, dc.sdfg.nodes.MapEntry)
        and any(isinstance(c, dc.sdfg.nodes.MapEntry) for c in st.scope_children()[n])
    ]
    assert nested, "fixture no longer nests one kernel inside another"
    for state, entry in nested:
        assert kernel_body(state, sdfg, entry, state.scope_children()) == []
    # and the inner one, which is a leaf, does carry the statement
    inner = [
        (st, n)
        for st in sdfg.all_states()
        for n in st.nodes()
        if isinstance(n, dc.sdfg.nodes.MapEntry) and st.entry_node(n) is not None
    ]
    assert any(kernel_body(st, sdfg, n, st.scope_children()) for st, n in inner), "inner printed nothing"


def test_an_emitter_refusal_is_reported_on_the_line_not_raised(monkeypatch):
    """The tree is a read-only view. A nest the numpy projection cannot express is exactly what the
    agent needs to be told about, so it must not take the whole tree down."""

    def refuse(state, sdfg, entry):
        raise introspect.UnsupportedNest("no emitter for this")

    monkeypatch.setattr(introspect, "map_body_lines", refuse)
    tree = with_bodies(shaped)
    assert "<not emitted: no emitter for this>" in tree


# --- reductions on the kernel line -----------------------------------------------------------------


@dc.program
def matvec(A: dc.float64[8, 4], B: dc.float64[4], C: dc.float64[8]):
    """A WCR over one of two map axes -- the classic tree reduction."""
    for i, j in dc.map[0:8, 0:4]:
        C[i] += A[i, j] * B[j]


def test_a_reduction_is_named_on_the_kernel_line():
    """A WCR on a map IS a tree reduction: the map declares its iterations independent, so the fold
    order is unspecified and a backend may use a register accumulator or an OpenMP clause. That is
    structural, and the agent should not have to read the body to find it."""
    tree = tree_of(matvec)
    assert "reduce=(+ over i1 -> C)" in tree, tree


def test_only_the_collapsed_axis_is_reported():
    """The reduced axes are the map parameters the OUTPUT subset does not mention -- a map over
    (i0, i1) writing C[i0] has collapsed i1 and only i1."""
    tree = tree_of(matvec)
    assert "over i1 ->" in tree and "over i0" not in tree, tree


def test_a_kernel_without_a_reduction_says_nothing():
    assert "reduce=" not in tree_of(shaped)


def test_the_reduction_op_is_read_off_the_wcr():
    sdfg = matvec.to_sdfg(simplify=True)
    normalize_for_tree(sdfg)
    state, entry = next(
        (st, n) for st in sdfg.all_states() for n in st.nodes() if isinstance(n, dc.sdfg.nodes.MapEntry)
    )
    assert introspect.kernel_reductions(state, entry) == ["+ over i1 -> C"]
    # A different op reads as itself, not as "+".
    exit_node = state.exit_node(entry)
    edge = next(e for e in state.in_edges(exit_node) if e.data.wcr is not None)
    edge.data.wcr = "lambda x, y: max(x, y)"
    assert introspect.kernel_reductions(state, entry) == ["max over i1 -> C"]


def test_a_body_is_not_recovered_by_slicing_the_emitted_block():
    """BK2: the body comes from `map_body_lines`, not from dropping len(params) lines off `map_lines`
    and dedenting by 4 * len(params). That arithmetic held only while every header was exactly one
    line and every body line carried the full indent."""
    sdfg = shaped.to_sdfg(simplify=True)
    normalize_for_tree(sdfg)
    state, entry = next(
        (st, n)
        for st in sdfg.all_states()
        for n in st.nodes()
        if isinstance(n, dc.sdfg.nodes.MapEntry) and st.entry_node(n) is None
    )
    body = kernel_body(state, sdfg, entry, state.scope_children())
    assert body, "the fixture kernel emits nothing"
    for line in body:
        assert not line.startswith(" "), f"a body line arrived still indented: {line!r}"
        assert not line.startswith("for "), f"a header leaked into the body: {line!r}"
    # and it agrees with the full block the emitter produces for the same kernel
    full = introspect.map_body_lines(state, sdfg, entry)
    assert body == full


# --- one kernel's body, by handle -------------------------------------------------------------------


def test_session_hands_back_one_kernel_body_by_its_tree_id():
    """The id on the tree line is the handle: read a line, ask what that kernel computes."""
    sdfg = shaped.to_sdfg(simplify=True)
    normalize_for_tree(sdfg)
    session = Session(sdfg)
    nest_id = re.findall(r"\[(e\d+:nest:\d+)\]", session.describe())[0]
    body = session.kernel_body(nest_id)
    assert body and all(isinstance(line, str) for line in body)
    assert not any(line.startswith("for ") for line in body), "headers are on the kernel line already"


def test_a_reduction_body_is_folded():
    """An explicit accumulate is the only POINT rendering of a reduction; `np.sum` is a whole-array
    spelling that belongs to the slice form."""
    sdfg = matvec.to_sdfg(simplify=True)
    normalize_for_tree(sdfg)
    session = Session(sdfg)
    nest_id = re.findall(r"\[(e\d+:nest:\d+)\]", session.describe())[0]
    body = "\n".join(session.kernel_body(nest_id))
    assert "C[i0] = C[i0] +" in body, body
    assert "np.sum" not in body


def test_a_stale_id_does_not_silently_return_someone_elses_body():
    sdfg = shaped.to_sdfg(simplify=True)
    normalize_for_tree(sdfg)
    session = Session(sdfg)
    nest_id = re.findall(r"\[(e\d+:nest:\d+)\]", session.describe())[0]
    session.bump()
    with pytest.raises(KeyError):
        session.kernel_body(nest_id)


# --- a kernel's REPRESENTATION: pure, runnable numpy -------------------------------------------------


def source_of_first_kernel(program):
    sdfg = program.to_sdfg(simplify=True)
    normalize_for_tree(sdfg)
    session = Session(sdfg)
    nest_id = re.findall(r"\[(e\d+:nest:\d+)\]", session.describe())[0]
    return sdfg, session, session.kernel_source(nest_id)


def test_a_kernel_source_is_a_whole_module_not_a_fragment():
    """`kernel_body` is the excerpt the tree prints; its statements reference loop variables that only
    exist inside their headers. The REPRESENTATION has to be something an agent can run."""
    _, _, source = source_of_first_kernel(shaped)
    assert source.startswith("import numpy as np")
    assert "\ndef kernel" in source
    compile(source, "<kernel>", "exec")  # syntactically a module, not a snippet


def test_a_kernel_source_runs_with_NOTHING_injected():
    """No EMITTED_BUILTINS, no `np` handed in -- pure numpy or it does not count."""
    _, _, source = source_of_first_kernel(matvec)
    namespace = {}
    exec(source, namespace)  # a bare dict: only what the source itself defines
    assert "int_floor" in namespace and "np" in namespace


def test_a_kernel_source_computes_what_the_sdfg_computes():
    """The point of it being runnable: emit, execute, compare. A representation nothing can execute
    cannot be shown to be correct."""
    sdfg, session, source = source_of_first_kernel(matvec)
    namespace = {}
    exec(source, namespace)
    kernel = next(v for k, v in namespace.items() if k.startswith("kernel") and callable(v))

    A = np.linspace(0.5, 4.0, 32).reshape(8, 4).copy()
    B = np.linspace(1.0, 2.0, 4).copy()
    from_source = np.zeros(8)
    kernel(A, B, from_source)

    from_sdfg = np.zeros(8)
    sdfg(A=A.copy(), B=B.copy(), C=from_sdfg)
    assert np.array_equal(from_source, from_sdfg), (from_source, from_sdfg)


def session_and_first_nest(program):
    sdfg = program.to_sdfg(simplify=True)
    normalize_for_tree(sdfg)
    session = Session(sdfg)
    nest_id = re.findall(r"\[(e\d+:nest:\d+)\]", session.describe())[0]
    return sdfg, session, nest_id


def test_c_is_the_same_numpy_lowered_by_the_translator():
    """`lang="c"` runs the point-form numpy through numpyto -- a real C function over the same arrays,
    not a second hand-written emitter."""
    _, session, nest_id = session_and_first_nest(matvec)
    src = session.kernel_source(nest_id, lang="c")
    assert "double" in src and "for (" in src
    assert "A" in src and "B" in src and "C" in src  # the boundary arrays survive to the signature


def test_cpp_is_the_cpp_half_of_the_one_c_family_emit():
    """Bare C++ is the `.cpp` of the plain `c` target (one emit produces both), so it carries the C++
    arithmetic header rather than OpenMP the agent never asked for."""
    _, session, nest_id = session_and_first_nest(matvec)
    src = session.kernel_source(nest_id, lang="cpp")
    assert 'extern "C"' in src and "template" in src  # the templated C++ int_floor/int_ceil header
    assert "#pragma omp" not in src  # NOT cpp_omp


def test_fortran_lowers_through_the_fortran_backend():
    _, session, nest_id = session_and_first_nest(matvec)
    src = session.kernel_source(nest_id, lang="fortran")
    assert "subroutine" in src and "bind(C" in src


def test_lowering_a_kernel_does_not_mutate_the_live_sdfg():
    """kernel_source is a projection: extraction outlines a map in place, so it must run on a COPY.
    The tree structure and the state count are identical before and after -- and no epoch bump, since
    a projection is not a mutation. (describe() mints fresh handle ids per call, so compare with those
    stripped.)"""
    strip = lambda tree: re.sub(r"\[e\d+:nest:\d+\]", "[nest]", tree)
    sdfg, session, nest_id = session_and_first_nest(matvec)
    before, epoch_before = strip(session.describe()), session.epoch
    states_before = len(list(sdfg.all_states()))
    session.kernel_source(nest_id, lang="c")
    assert session.epoch == epoch_before
    assert len(list(sdfg.all_states())) == states_before
    assert strip(session.describe()) == before


def test_an_unknown_language_is_refused_by_name():
    _, session, nest_id = session_and_first_nest(shaped)
    with pytest.raises(ValueError, match="expected 'python'"):
        session.kernel_source(nest_id, lang="rust")
