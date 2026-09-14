# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
import dace

from nestforge.phases.scopes import outer
from nestforge.ir.extract import extract_nest_to_sdfg
from nestforge.corpus.translate import prepare, emit_sources
from nestforge.build.arena import run_arena, discover_compilers
from nestforge.build import flags

N = dace.symbol('N')


@dace.program
def vadd(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N]:
        C[i] = A[i] + B[i]


def test_arena_vadd(tmp_path):
    sdfg = vadd.to_sdfg(simplify=True)
    psdfg, node = outer(sdfg)[0]
    b = extract_nest_to_sdfg(psdfg, node, name="vadd")

    prep = prepare(b, "vadd", tmp_path / "kern")
    srcs = emit_sources(prep, tmp_path / "gen", target="c")
    c_source = next(p for p in srcs if p.suffix == ".c")

    sizes = {"N": 1 << 15}
    res = run_arena(prep, b, c_source, tmp_path / "build", sizes=sizes, reps=50)

    assert discover_compilers(), "no compilers on PATH"
    # every FP mode has at least one correct build, and a winner
    for mode in flags.FP_LEVELS:
        assert mode in res.winners, f"no correct build for {mode}: {[(c.compiler,c.fp_mode,c.ok,c.maxdiff) for c in res.cells if c.fp_mode==mode]}"
    # strict-ieee must be bit-exact vs the numpy oracle
    assert res.winners["strict-ieee"].maxdiff == 0.0
    # all vadd cells are correct (a pure add reassociates trivially)
    assert all(c.ok for c in res.cells), [(c.compiler, c.fp_mode, c.maxdiff, c.error) for c in res.cells if not c.ok]
    # total optimization time (the sweep) and per-candidate compile time are tracked
    assert res.optimization_seconds > 0.0
    assert all(c.compile_us > 0.0 for c in res.cells)
    # EVERY cell is still built and reported; dedup removes measurements, never cells
    assert len(res.cells) == len(discover_compilers()) * len(flags.FP_LEVELS)
    # a pure add reaches none of the fp ladder, so the rungs must collapse -- and a collapsed cell must
    # carry its twin's real numbers, not a blank, or the winner tables silently lose those rungs
    assert res.collapsed, "vadd compiles identically at every rung; nothing collapsed means dedup is inert"
    for c in res.cells:
        if c.same_as is None:
            continue
        twin = next(t for t in res.cells if f"{t.compiler}:{t.fp_mode}" == c.same_as)
        assert (c.ok, c.time_us, c.maxdiff) == (twin.ok, twin.time_us, twin.maxdiff), (c, twin)
    assert len([c for c in res.cells if c.same_as is None]) < len(res.cells), "dedup saved no measurement"
    assert len(res.collapsed) == len({c.same_as for c in res.cells if c.same_as is not None}), "one note per group"


if __name__ == "__main__":
    import tempfile
    import pathlib
    test_arena_vadd(pathlib.Path(tempfile.mkdtemp()))
    print("arena OK")
