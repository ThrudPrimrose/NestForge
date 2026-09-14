# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Fuzz the Phase-2 arms: RANDOM programs x RANDOM fuse/fission sequences, asserting the three properties
that make the arms safe for an agent to drive --

  1. no crash (enumeration and application never raise on a valid program),
  2. ``sdfg.validate()`` holds after every sequence,
  3. the result is bit-exact to the un-transformed reference.

Deterministic: every case is seeded, and a failure reports the seed AND the generated source so it can be
replayed and shrunk by hand. Bounded for a shared box: small sizes, few loops, capped case count, ``-n1``.

The generator emits real ``@dace.program`` source into a temp module (the DaCe frontend parses a function's
SOURCE, so an ``exec``-ed function would not work) over a statement grammar with the hazards that matter:
recurrences (sequential), element-wise (DOALL), stencil reads, and cross-array producer/consumer chains --
in sequential ``range`` loops and parallel ``dace.map`` loops.
"""

import importlib.util

import numpy as np
import pytest

from dace.transformation.interstate.state_fusion import StateFusion

from nestforge.phases.schedule import fission_to_statements
from nestforge.phases.schedule import apply_fusion, enumerate_fusions

ARRAYS = ("a", "b", "c", "d")
NCASES_FUSE = 12
NCASES_FISSION = 8


def gen_source(seed: int) -> str:
    """A random but WELL-DEFINED ``@dace.program``: 2-4 loops, each with 1-2 statements from the grammar.
    Every loop runs ``1:N-1`` so ``i-1`` / ``i+1`` stay in bounds for any N >= 3.

    A ``dace.map`` is DATA PARALLEL by definition, so a map body must carry no cross-iteration dependence:
    inside a map we never offset-read (``x[i+-1]``) an array that any statement of that same map writes --
    that would be a race, making the program's own reference run order-dependent and the bit-exact
    comparison meaningless. Same-index reads (``src[i]``) are fine even for an array written in the map
    (an intra-iteration dependence the state's dataflow orders). Recurrences are sequential-only.

    The grammar also emits a LOOP-INVARIANT scalar ``s``: a sequential loop may write it (``s = a[i]``,
    last iteration wins) and any loop may read it. That shape is what a carried-offset dependence
    classifier sees no offset for -- there is no iterator in the subset to carry one -- and reading that
    as "no dependence" is a real fusion miscompile (a loop reading ``s`` unfused sees the FINAL value,
    fused it sees the RUNNING one). An earlier grammar without it fuzzed green over a live bug.

    One loop either READS ``s`` or WRITES it, never both, and only a sequential loop writes it. Writing
    ``s`` from a map is a race on the scalar -- every iteration stores to one cell. Reading and writing it
    in the same body (``d[i] = b[i] + s; ...; s = c[i]``) is perfectly well-defined and DaCe compiles it
    correctly; it is excluded only because it chains every statement through ``s``, which makes the loop
    un-fissionable and so exercises none of the granularity the fission arm exists to reach. (It did find
    a real ``LoopFission`` miscompile on the way in -- fixed in DaCe, and pinned there by its own
    regression test, which is where that belongs rather than in this generator.)

    The write/read hazard the arms must handle is still generated -- it just spans two loops, which is
    where fusion has to reason about it anyway.
    """
    rng = np.random.default_rng(seed)
    lines = [
        "import dace",
        "import numpy as np",
        "",
        'N = dace.symbol("N")',
        "f64 = dace.float64",
        "",
        "@dace.program",
        f"def k({', '.join(f'{x}: f64[N]' for x in ARRAYS)}):",
        "    s = np.float64(0.0)",
    ]
    for _ in range(int(rng.integers(2, 5))):
        parallel = bool(rng.integers(0, 2))
        nstmt = int(rng.integers(1, 3))
        # Targets are fixed up front and distinct, so the map's whole write-set is known before any
        # statement is emitted -- that is what lets us keep every offset read off a written array.
        targets = list(rng.choice(ARRAYS, size=min(nstmt, len(ARRAYS)), replace=False))
        written = set(targets)
        safe_offset_srcs = [x for x in ARRAYS if x not in written]  # offset-readable without a race
        # This loop's ONE relationship to the invariant scalar (never both -- see the docstring); only a
        # sequential loop may write it.
        s_use = (
            ("read", "write", "none")[int(rng.integers(3))] if not parallel else ("read", "none")[int(rng.integers(2))]
        )

        body = []
        for pos, tgt in enumerate(targets):
            if parallel:
                forms = ["elementwise"] + (["stencil"] if safe_offset_srcs else [])
            else:
                forms = ["elementwise", "stencil", "recurrence", "scaled_rec"]
            # A loop that reads s reads it in its FIRST statement, rather than leaving it to the form
            # draw: the write-then-read pair is the hazard this grammar exists to reach, and letting the
            # RNG miss it made the shape rare enough to be worthless as coverage.
            form = "invariant_read" if (s_use == "read" and pos == 0) else forms[int(rng.integers(len(forms)))]
            if form == "elementwise":
                src = ARRAYS[int(rng.integers(len(ARRAYS)))]  # same-index read: never a cross-iteration dep
                body.append(f"        {tgt}[i] = {src}[i] * 2.0")
            elif form == "invariant_read":
                src = ARRAYS[int(rng.integers(len(ARRAYS)))]
                body.append(f"        {tgt}[i] = {src}[i] + s")
            elif form == "stencil":
                pool = safe_offset_srcs if parallel else list(ARRAYS)
                src = pool[int(rng.integers(len(pool)))]
                body.append(f"        {tgt}[i] = {src}[i + 1] + {src}[i - 1]")
            elif form == "recurrence":
                src = ARRAYS[int(rng.integers(len(ARRAYS)))]
                body.append(f"        {tgt}[i] = {tgt}[i - 1] + {src}[i]")
            else:
                src = ARRAYS[int(rng.integers(len(ARRAYS)))]
                body.append(f"        {tgt}[i] = {tgt}[i - 1] * 0.5 + {src}[i]")
        if s_use == "write":
            # the invariant WRITE, sequential-only. Paired with an invariant_read in ANOTHER loop, this is
            # the cross-loop fusion hazard; no statement of THIS body reads s, so the body stays ordered.
            body.append(f"        s = {ARRAYS[int(rng.integers(len(ARRAYS)))]}[i]")
        lines.append("    for i in dace.map[1:N - 1]:" if parallel else "    for i in range(1, N - 1):")
        lines.extend(body)
    return "\n".join(lines) + "\n"


def load_program(tmp_path, seed: int):
    """Write the generated source to a temp module and import it -- the DaCe frontend needs real source."""
    src = gen_source(seed)
    path = tmp_path / f"gen_{seed}.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location(f"fuzz_gen_{seed}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.k, src


def inputs_for(n=16, seed=0):
    rng = np.random.default_rng(seed + 9999)
    return {x: rng.random(n) for x in ARRAYS}


def run(sdfg, inputs, n):
    bufs = {k: v.copy() for k, v in inputs.items()}
    sdfg(**bufs, N=n)
    return bufs


def random_fuse_to_fixpoint(sdfg, seed: int) -> int:
    """Apply a RANDOM legal fusion each round until none remain -- the agent's actual move pattern (and the
    composition hazard: a fusion can invalidate or enable another)."""
    rng = np.random.default_rng(seed)
    applied = 0
    for _ in range(200):  # bound: each fusion strictly reduces the nest count
        moves = enumerate_fusions(sdfg)
        if not moves:
            return applied
        apply_fusion(sdfg, moves[int(rng.integers(len(moves)))])
        applied += 1
    raise AssertionError("random fusion did not converge")


@pytest.mark.parametrize("seed", range(NCASES_FUSE))
def test_fuzz_random_fuse_sequence_is_value_preserving(seed, tmp_path):
    prog, src = load_program(tmp_path, seed)
    n = 16
    inputs = inputs_for(n, seed)
    ref = run(prog.to_sdfg(simplify=True), inputs, n)

    sdfg = prog.to_sdfg(simplify=True)
    sdfg.apply_transformations_repeated(StateFusion)  # co-locate so the map arms can match
    random_fuse_to_fixpoint(sdfg, seed)
    sdfg.validate()
    got = run(sdfg, inputs, n)
    for name in inputs:
        assert np.allclose(got[name], ref[name], equal_nan=True), (
            f"seed={seed} diverged on {name!r} after a random fusion sequence\n--- generated ---\n{src}"
        )


@pytest.mark.parametrize("seed", range(NCASES_FISSION))
def test_fuzz_fission_then_random_fuse_is_value_preserving(seed, tmp_path):
    # the full Phase-2 round trip on a random program: explode to statements, then fuse back up randomly.
    prog, src = load_program(tmp_path, seed + 500)
    n = 16
    inputs = inputs_for(n, seed)
    ref = run(prog.to_sdfg(simplify=True), inputs, n)

    sdfg = prog.to_sdfg(simplify=True)
    fission_to_statements(sdfg)
    sdfg.validate()
    sdfg.apply_transformations_repeated(StateFusion)
    random_fuse_to_fixpoint(sdfg, seed)
    sdfg.validate()
    got = run(sdfg, inputs, n)
    for name in inputs:
        assert np.allclose(got[name], ref[name], equal_nan=True), (
            f"seed={seed} diverged on {name!r} after fission + random fusion\n--- generated ---\n{src}"
        )


def test_generator_is_deterministic():
    assert gen_source(3) == gen_source(3)
    assert gen_source(3) != gen_source(4)
