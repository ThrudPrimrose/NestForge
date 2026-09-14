# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression tests for the code-review findings (corpus selection, C-style emission, emit guards)."""
import numpy as np
import pytest
import dace
from dace.sdfg.state import LoopRegion

from nestforge.build.arena import make_inputs, scratch_names
from nestforge.ir.emit_numpy import load_emitted, maxsize_loop_scratch, nest_to_numpy, sdfg_to_numpy
from nestforge.ir.extract import Boundary
from nestforge.phases.scopes import lower_nests_to_external_call

N = dace.symbol('N')


# ----- corpus: pick the kernel's entry @dace.program, not the first helper (Finder B#1) ----------
def test_corpus_program_is_the_entry_not_a_helper():
    from nestforge.corpus.bench import iter_dace_kernels
    ks = {k.short_name: k for k in iter_dace_kernels()}
    # mlp_dace defines relu, softmax, then mlp; resnet has resnet_basicblock + a _gpu variant after it.
    assert ks["machine_learning/mlp/mlp"].program().name.endswith("mlp")
    assert ks["machine_learning/resnet/resnet"].program().name.endswith("resnet_basicblock")


def test_corpus_module_path_independent_of_namespace_path():
    from nestforge.corpus.bench import module_path
    # Derived from the registry key, not hpcagent_bench.benchmarks.__path__ (which can be stale/multi-root).
    assert module_path("scientific_computing/dense_linear_algebra/gemm/gemm") == \
        "hpcagent_bench.benchmarks.hpc.dense_linear_algebra.gemm.gemm_dace"


# ----- C-style emission: pre-allocated buffers, no internal allocation ----------------------------
def run(src, fn, **buffers):
    vars(load_emitted(src, fn))[fn](**buffers)


@dace.program
def dot_scale(x: dace.float64[N], y: dace.float64[N], z: dace.float64[N], out: dace.float64[N]):
    s = np.dot(x, y)
    for i in dace.map[0:N]:
        out[i] = z[i] * s


def test_scalar_transient_consistent_between_libnode_and_tasklet():
    # Dot writes a scalar transient `s`; the map tasklet reads it. Both must name it identically.
    src = sdfg_to_numpy(dot_scale.to_sdfg(simplify=True), "k")
    n = 6
    rng = np.random.default_rng(0)
    x, y, z, out = rng.random(n), rng.random(n), rng.random(n), np.zeros(n)
    run(src, "k", x=x, y=y, z=z, out=out, N=n)
    np.testing.assert_allclose(out, z * (x @ y))


@dace.program
def matvec_return(A: dace.float64[N, N], v: dace.float64[N]):
    return A @ v


def test_return_and_scratch_are_inplace_buffer_params_no_allocation():
    sdfg = matvec_return.to_sdfg(simplify=True)
    src = sdfg_to_numpy(sdfg, "k")
    assert "np.empty" not in src and "np.zeros" not in src, "C-style: caller pre-allocates, kernel must not"
    assert "return " not in src, "C-style: __return is an in-place output buffer param, not a python return"
    # __return is a parameter written in place.
    header = src.splitlines()[0]
    assert "__return" in header
    n = 5
    rng = np.random.default_rng(1)
    A, v = rng.random((n, n)), rng.random(n)
    ret = np.zeros(n)
    run(src, "k", A=A, v=v, __return=ret, N=n)
    np.testing.assert_allclose(ret, A @ v)


# ----- emit: a map nested in a map is emitted as NESTED for-loops (not dropped, not raised) --------
def nested_map_sdfg():
    sdfg = dace.SDFG("nested")
    sdfg.add_array("A", [N, N], dace.float64)
    sdfg.add_array("B", [N, N], dace.float64)
    st = sdfg.add_state()
    me_i, mx_i = st.add_map("outer", dict(i="0:N"))
    me_j, mx_j = st.add_map("inner", dict(j="0:N"))
    t = st.add_tasklet("t", {"a"}, {"b"}, "b = a * 2.0")
    rA, wB = st.add_read("A"), st.add_write("B")
    st.add_memlet_path(rA, me_i, me_j, t, dst_conn="a", memlet=dace.Memlet("A[i, j]"))
    st.add_memlet_path(t, mx_j, mx_i, wB, src_conn="b", memlet=dace.Memlet("B[i, j]"))
    return sdfg


def test_nested_map_in_map_emits_nested_for_loops():
    """A map nested inside a map (the multi-nest kernels s2275 / s152 need this) is emitted as NESTED
    ``for`` loops with the inner body at the deeper indent -- NOT dropped, and no longer refused. Guards the
    ``map_lines`` recursion that un-skipped the nested-map corpus kernels."""
    src = sdfg_to_numpy(nested_map_sdfg(), "k")
    assert "for i in range(0, N, 1):" in src and "for j in range(0, N, 1):" in src
    assert "B[i, j] = (A[i, j] * 2.0)" in src
    # numerically correct: exec the emitted kernel and compare to B = A * 2.
    n = 6
    rng = np.random.default_rng(0)
    A, B = rng.random((n, n)), np.zeros((n, n))
    load_emitted(src, "k").k(A, B, n)
    assert np.allclose(B, A * 2.0)


# ----- emit staging: a map-entry-sourced scalar read (`b_index = b[i]`) must be emitted -----------
@dace.program
def gather(a: dace.float64[N], b: dace.int64[N], out: dace.float64[N]):
    for i in dace.map[0:N]:
        out[i] = a[b[i]]


def test_indirect_gather_stages_map_entry_read():
    """DaCe requires every array access to be symbolic, so an indirect read ``a[b[i]]`` is staged as a
    scalar access node fed by the map entry (``<sym> = b[i]``) before the gather ``a[<sym>]``. The
    ``copy_lines`` map-entry branch emits that load; without it the gather names an undefined symbol and
    the kernel raises ``NameError`` at exec. Guards the map-entry-staging fix (baseline opt mode)."""
    sdfg = gather.to_sdfg(simplify=True)
    src = sdfg_to_numpy(sdfg, "k")
    assert "= b[i]" in src, f"map-entry-sourced staging load not emitted:\n{src}"
    n = 8
    rng = np.random.default_rng(0)
    a, b, out = rng.random(n), rng.integers(0, n, size=n).astype(np.int64), np.zeros(n)
    run(src, "k", a=a, b=b, out=out, N=n)  # NameError here if the staged read is dropped
    np.testing.assert_allclose(out, a[b])


# ----- arena regression: a value-returning kernel no longer crashes run_oracle (Finder C#1) -------
@dace.program
def scaley(A: dace.float64[N]):
    return A * 2.0


def test_returning_kernel_survives_arena_oracle_and_manifest_matches(tmp_path):
    from nestforge.phases.scopes import lower_nests_to_external_call
    from nestforge.corpus.translate import prepare
    from nestforge.build.arena import make_inputs, run_oracle

    sdfg = scaley.to_sdfg(simplify=True)
    ext, boundary = lower_nests_to_external_call(sdfg)[0]
    assert boundary.outputs == ["__return"]
    prep = prepare(boundary, ext.name, tmp_path / "k")
    # __return is an in-place buffer parameter in the numpy signature AND the manifest -- aligned.
    assert "__return" in prep.numpy_source.splitlines()[0]
    assert "return " not in prep.numpy_source
    # emit_yaml.arg_order and emit_numpy.nest_to_numpy build the signature independently, so the manifest is
    # only usable while they agree: arrays in array_args order (inputs, extra outputs, scratch), then symbols.
    header = prep.numpy_source.splitlines()[0]
    signature = [a.strip() for a in header[header.index("(") + 1:header.rindex(")")].split(",")]
    args = list(prep.manifest["input_args"])
    arrays = list(prep.manifest["array_args"])
    assert args == arrays + [s for s in boundary.symbols if s not in arrays]
    assert args == signature, f"manifest input_args {args} != emitted numpy signature {signature}"
    assert "__return" in prep.manifest["input_args"]
    sizes = {"N": 16}
    out = run_oracle(prep, boundary, make_inputs(boundary, sizes), sizes)  # crashed pre-fix
    assert "__return" in out


# ----- an in-place nest's DaceReference expansion resolves BOTH of its connectors ------------------
@dace.program
def scale_inplace(A: dace.float64[N], b: dace.float64[N]):
    for i in dace.map[0:N]:
        A[i] = A[i] * 2.0 + b[i]


def inplace_lowered():
    sdfg = scale_inplace.to_sdfg(simplify=True)
    ext, boundary = lower_nests_to_external_call(sdfg)[0]
    assert "A" in boundary.inputs and "A" in boundary.outputs, "A must be read+written for this to test anything"
    return sdfg, ext


def test_inplace_nest_reference_sdfg_declares_every_connector():
    """An in-place array is in BOTH boundary.inputs and boundary.outputs, so its ExternalCall carries two
    connectors (``_in_A`` and ``_out_A``) for one array. ``reference_sdfg`` renamed the body to ``_in_A``
    first, which left the ``_out_A`` rename a silent no-op (``SDFG._replace_dict_keys`` skips a name that is
    already gone) and the DaceReference expansion with no ``_out_A`` descriptor -- NestedSDFG validation then
    rejects the connector. Guards the ExternalCall/reference connector alignment for read+write boundaries."""
    sdfg, ext = inplace_lowered()
    arrays = ext._standalone_sdfg.arrays
    for conn in set(ext.in_connectors) | set(ext.out_connectors):
        assert conn in arrays, f"connector {conn} has no descriptor in the reference SDFG: {sorted(arrays)}"
    sdfg.expand_library_nodes()
    sdfg.validate()  # raised InvalidSDFGNodeError('Connector "_out_A" ... not a registered data descriptor')


@pytest.mark.integration  # compiles + runs the DaceReference expansion
def test_inplace_nest_reference_expansion_is_value_preserving():
    """The reference expansion must not just validate, it must still compute: the body works on ``_out_A``
    (the one pointer connector_for also hands the extern call for an in-place arg), which the parent aliases
    to the same AccessNode as ``_in_A``, so it carries the input values on entry."""
    n = 16
    rng = np.random.default_rng(0)
    a, b = rng.random(n), rng.random(n)
    expected = a * 2.0 + b

    sdfg, _ = inplace_lowered()
    got = a.copy()
    sdfg(A=got, b=b.copy(), N=n)
    np.testing.assert_allclose(got, expected)


# ----- arena: caller-side scratch sizing must match the emitter's widening ------------------------
def loop_scratch_boundary():
    """A nest with a scratch transient shaped by the LOOP VARIABLE (``tmp[loop_i + 1]``) -- the shape the
    emitter widens to ``N + 1`` so the buffer stays a caller-allocated parameter."""
    # A dedicated symbol name: ``i`` is a common loop variable, and dace's symbol registry rejects a
    # re-declaration with a different dtype, which would couple this test to whatever ran before it.
    loop_i = dace.symbol("loop_i", dace.int64)
    sdfg = dace.SDFG("loop_scratch")
    sdfg.add_array("a", [N], dace.float64)
    sdfg.add_transient("tmp", [loop_i + 1], dace.float64)
    loop = LoopRegion("loop", "loop_i < N", "loop_i", "loop_i = 0", "loop_i = loop_i + 1")
    sdfg.add_node(loop, is_start_block=True)
    body = loop.add_state("body", is_start_block=True)
    body.add_edge(body.add_read("a"), None, body.add_tasklet("t", {"i0"}, {"o0"}, "o0 = i0 + 1.0"), "i0",
                  dace.Memlet("a[0]"))
    return Boundary(inputs=["a"],
                    outputs=["a"],
                    symbols=["N"],
                    nsdfg_node=None,
                    state=None,
                    standalone_sdfg=sdfg,
                    parent_sdfg=None)


def test_make_inputs_sizes_scratch_the_way_the_emitter_widened_it():
    """make_inputs sized scratch from the RAW descriptor while the emitted kernel is written against the
    ``maxsize_loop_scratch``-widened one, so the caller handed the kernel a buffer smaller than it indexes
    -- a write past the end of the allocation across the ABI (heap corruption in the forked child)."""
    boundary = loop_scratch_boundary()
    sizes = {"N": 4}
    widened = maxsize_loop_scratch(boundary.standalone_sdfg, boundary.symbols).arrays["tmp"]
    assert str(widened.shape[0]) == "N + 1"  # the extent the emitted kernel addresses

    got = make_inputs(boundary, sizes, seed=0)["tmp"]
    assert got.shape == (sizes["N"] + 1, ), "scratch allocated from the raw (smaller) shape, not the emitted one"


def test_scratch_names_reports_the_emitted_scratch_buffers():
    # The scratch parameter list must come from the same widened SDFG the signature is rendered from.
    boundary = loop_scratch_boundary()
    assert scratch_names(boundary) == ["tmp"]
    assert nest_to_numpy(boundary, "k").splitlines()[0] == "def k(a, tmp, N):"
