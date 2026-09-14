# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The two duplicate-variant keys, and the boundary between what each one can see.

The pair exists because neither key subsumes the other, and using the wrong one silently erases a
search axis: compile flags never reach the emitted C++, so a source-level key reports every fp rung as
the same variant.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

from nestforge.build import flags as flags_mod
from nestforge.build.flags import FP_LEVELS
from nestforge.build.dedup import (
    asm_bodies,
    asm_body_key,
    collapse,
    cpp_body_key,
    function_bodies,
    needed_libraries,
    parse_disassembly,
    representatives,
    variant_key,
)
from nestforge.build.sdfg import BuildOptions, build_archive
from nestforge.build.toolchain import compiler_family

SYMBOL = "k_fp64"
#: A transcendental, in a loop the back end will vectorize, for the needed_libraries link-axis test.
SIN_KERNEL = f"""#include <math.h>
extern "C" void {SYMBOL}(double *__restrict__ a, const double *__restrict__ b, int n) {{
  for (int i = 0; i < n; ++i) a[i] = sin(b[i]);
}}
"""
#: A reduction: reassociation is exactly what separates fast-math from the strict rungs.
SUM_KERNEL = f"""extern "C" void {SYMBOL}(double *__restrict__ a, const double *__restrict__ b, int n) {{
  double s = 0.0;
  for (int i = 0; i < n; ++i) s += b[i] * b[i];
  a[0] = s;
}}
"""

assert shutil.which("g++") is not None, "g++ not on PATH (setup_apt.sh installs it)"
assert shutil.which("clang++") is not None, "clang++ not on PATH (setup_apt.sh installs it)"
assert shutil.which("objdump") is not None, "objdump not on PATH (setup_apt.sh: binutils)"


def build(tmp_path: Path, source: str, fp_mode: str, tag: str = "v", compiler: str = "g++") -> Path:
    """Build ``source`` the way a phase-4 variant is built and return the object the keys read."""
    src = tmp_path / f"{tag}.cpp"
    src.write_text(source)
    family = compiler_family(compiler)
    composed = (
        flags_mod.base_flags(family) + flags_mod.fp_flags(family, fp_mode) + flags_mod.cost_flags(family, "default")
    )
    out = tmp_path / tag
    opts = BuildOptions(compiler=compiler, flags=composed)
    build_archive([src], out, out / f"lib{tag}.a", out / f"lib{tag}.so", opts)
    return out / f"{tag}.o"


# ---------------------------------------------------------------- the C++ key and its blind spot


def test_the_cpp_key_cannot_see_compile_flags_and_the_asm_key_can(tmp_path):
    """The whole reason there are two keys, stated as the grouping a sweep would actually do. One source,
    two fp rungs: the source key puts them in ONE group (it never sees the command line), the object key
    in two. A pruner on the source key alone would measure strict-ieee and report it as fast-math too."""
    strict = build(tmp_path, SUM_KERNEL, "strict-ieee", tag="s_cpp")
    fast = build(tmp_path, SUM_KERNEL, "fast-math", tag="f_cpp")
    by_source = collapse({"strict": cpp_body_key(SUM_KERNEL), "fast": cpp_body_key(SUM_KERNEL)})
    by_object = collapse({"strict": asm_body_key(strict, SYMBOL), "fast": asm_body_key(fast, SYMBOL)})
    assert len(by_source) == 1, "the emitted source is identical -- flags are not an input to it"
    assert len(by_object) == 2, "the objects are not, which is the axis the source key would erase"


def test_the_cpp_key_separates_sources_that_differ_in_the_body():
    other = SIN_KERNEL.replace("sin(b[i])", "cos(b[i])")
    assert cpp_body_key(SIN_KERNEL) != cpp_body_key(other)


def test_the_cpp_key_ignores_layout_and_the_includes_that_follow_the_build_dir():
    """clang-format normalizes spelling, and only BODIES are hashed -- an emitted TU's include paths and
    name comments follow the build directory, so hashing the whole file would never match twice."""
    reflowed = SIN_KERNEL.replace("  for", "\n\n      for").replace("*__restrict__ a", "*__restrict__  a")
    relocated = "#include <math.h>\n// generated into /tmp/build-abc123\n" + SIN_KERNEL.split("\n", 1)[1]
    assert cpp_body_key(SIN_KERNEL) == cpp_body_key(reflowed)
    assert cpp_body_key(SIN_KERNEL) == cpp_body_key(relocated)


def test_function_bodies_does_not_desync_on_braces_inside_literals_or_comments():
    """A generated TU carries error strings and comments; a brace in one would shift every body after it,
    which makes the key depend on unrelated text."""
    code = 'void f() { const char *s = "{ not code }"; /* } neither */ int x = 1; }\nvoid g() { int y = 2; }\n'
    bodies = function_bodies(code)
    assert len(bodies) == 2, bodies
    assert "int x = 1;" in bodies[0] and "int y = 2;" in bodies[1]


def test_function_bodies_keeps_an_escaped_quote_inside_a_literal():
    code = r'void f() { const char *s = "he said \" { \" ok"; int x = 1; }' + "\n"
    assert len(function_bodies(code)) == 1


# ---------------------------------------------------------------- the assembly key


def test_the_asm_key_separates_fp_rungs_the_cpp_key_cannot(tmp_path):
    """The complement of the blind spot above, on the same source: identical C++, different objects."""
    strict = build(tmp_path, SUM_KERNEL, "strict-ieee", tag="strict")
    fast = build(tmp_path, SUM_KERNEL, "fast-math", tag="fast")
    assert asm_body_key(strict, SYMBOL) != asm_body_key(fast, SYMBOL), "reassociation must change the code"


#: One objdump body, verbatim shape: an immediate, a rip-relative load with a relocation comment, and a
#: branch whose target objdump already annotates with the symbol.
DISASM = """
0000000000000000 <k_fp64>:
   0:\tendbr64
   4:\tadd    $0x2,%eax
   7:\tmovsd  0x0(%rip),%xmm1        # f <k_fp64+0xf>
   c:\tcall   1090 <sin@plt>
  11:\tret
"""


def test_parsing_strips_the_address_column_the_reloc_comment_and_the_branch_slot():
    """Each of the three is layout, not content: the address column shifts with position, the reloc
    comment repeats it, and a PLT slot moves between links of identical code."""
    body = parse_disassembly(DISASM)["k_fp64"]
    assert body.splitlines() == ["endbr64", "add    $0x2,%eax", "movsd  0x0(%rip),%xmm1", "call   <sin@plt>"] + ["ret"]


def test_parsing_keeps_an_immediate_that_only_differs_in_value():
    """The exact risk :data:`BRANCH_TARGET` creates -- stripping a branch address must not strip an
    IMMEDIATE. Asserted on the normalizer directly: two objects differing by one operand are hard to get
    out of a real compiler, which is what made an end-to-end version of this test pass vacuously."""
    two = parse_disassembly(DISASM)["k_fp64"]
    three = parse_disassembly(DISASM.replace("$0x2", "$0x3"))["k_fp64"]
    assert two != three, "an operand is content; erasing it collapses two genuinely different builds"


def test_parsing_keeps_a_rip_offset_that_only_differs_in_value():
    """Same argument one step out: the reloc COMMENT is dropped, the addressing operand is not."""
    a = parse_disassembly(DISASM)["k_fp64"]
    b = parse_disassembly(DISASM.replace("0x0(%rip)", "0x8(%rip)"))["k_fp64"]
    assert a != b


def test_asm_bodies_strips_addresses_but_keeps_the_instructions(tmp_path):
    """Addresses shift with layout and would defeat the match; the mnemonics are the content."""
    obj = build(tmp_path, SUM_KERNEL, "strict-ieee", tag="body")
    body = asm_bodies(obj)[SYMBOL]
    assert body and "ret" in body
    assert not any(line.strip().startswith(("0x", "00000")) for line in body.splitlines())


def test_naming_an_absent_symbol_is_an_error_not_a_silent_whole_object_key(tmp_path):
    """Falling back to the whole object would key on init/exit boilerplate and quietly answer a different
    question than the caller asked."""
    obj = build(tmp_path, SUM_KERNEL, "strict-ieee", tag="missing")
    with pytest.raises(LookupError, match="nope"):
        asm_body_key(obj, "nope")


def test_an_object_with_no_disassembly_raises_rather_than_hashing_nothing(tmp_path):
    """Two unreadable objects both hashing the empty string would collapse into one measured variant."""
    empty = tmp_path / "not_an_object.o"
    empty.write_text("")
    with pytest.raises(LookupError):
        asm_body_key(empty, SYMBOL)


def test_needed_libraries_reads_the_link_axis_the_object_key_misses(tmp_path):
    """One object, two links: keying a ``.so`` means composing the two -- same code plus a different
    resolver is still a different variant to measure."""
    obj = build(tmp_path, SIN_KERNEL, "fast-math", tag="need")
    bare, withlib = tmp_path / "libbare.so", tmp_path / "libwith.so"
    subprocess.run(["gcc", "-shared", str(obj), "-o", str(bare)], check=True, capture_output=True)
    subprocess.run(
        ["gcc", "-shared", str(obj), "-Wl,--no-as-needed", "-lmvec", "-o", str(withlib)],
        check=True,
        capture_output=True,
    )
    assert asm_body_key(bare, SYMBOL) == asm_body_key(withlib, SYMBOL), "the code really is the same"
    assert needed_libraries(bare) != needed_libraries(withlib), "the link really is not"
    assert any("mvec" in soname for soname in needed_libraries(withlib))


#: No FMA to contract, no math call to relax, nothing to reassociate: the fp ladder cannot reach it.
ADD_KERNEL = f"""extern "C" void {SYMBOL}(double *__restrict__ c, const double *__restrict__ a, const double *__restrict__ b,
    long n) {{
  for (long i = 0; i < n; ++i) c[i] = a[i] + b[i];
}}
"""


def test_the_pruner_collapses_fp_rungs_a_kernel_cannot_tell_apart(tmp_path):
    """The point of the module, on the arena's own axis. Every rung of the ladder compiles this kernel to
    the same object, so a sweep that measures all four is timing one binary four times. Paired with
    :func:`test_the_asm_key_separates_fp_rungs_the_cpp_key_cannot`, which shows a kernel that CAN tell
    them apart is not collapsed -- otherwise "it collapses" would just mean the key is broken."""
    keys = {rung: variant_key(build(tmp_path, ADD_KERNEL, rung, tag=rung), SYMBOL) for rung in FP_LEVELS}
    assert all(k is not None for k in keys.values()), keys
    groups = collapse(keys)
    assert len(groups) < len(FP_LEVELS), f"no rung collapsed, so the pruner saves nothing here: {groups}"
    picks, notes = representatives(keys)
    assert len(picks) == len(groups) and notes


# ---------------------------------------------------------------- grouping


def test_collapse_groups_by_key_and_keeps_the_first_as_representative():
    groups = collapse({"a": "k1", "b": "k2", "c": "k1"})
    assert groups == {"k1": ["a", "c"], "k2": ["b"]}


def test_representatives_reports_what_it_collapsed():
    """A silent collapse reads exactly like a sweep that covered everything."""
    picks, notes = representatives({"a": "k1", "b": "k2", "c": "k1"})
    assert picks == ["a", "b"]
    assert notes == ["a == c"]


def test_representatives_is_quiet_when_nothing_collapsed():
    picks, notes = representatives({"a": "k1", "b": "k2"})
    assert picks == ["a", "b"] and notes == []
