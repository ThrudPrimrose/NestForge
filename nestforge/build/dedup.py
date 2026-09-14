# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Collapse variants that are the same build twice, so the sweep measures each distinct build once.

Two keys, because they see different things and neither subsumes the other:

  * :func:`cpp_body_key` -- clang-formatted function bodies. Sees CODEGEN knobs ONLY. Every compile
    flag (``-O``, ``-fveclib``, cost model, fp rung) is invisible in the emitted source, so a whole
    flag axis shares one key here and collapsing on it alone would erase that axis.
  * :func:`asm_body_key` -- the disassembled function. Sees the flag cells, which is where the
    duplicates actually are: a veclib cell below its enabling fp rung emits byte-identical code to
    ``veclib=none``, and a cost model the back end ignores emits the same object as ``default``.

What this saves is MEASUREMENT, not compile: an object must exist before it can be hashed, and the
compile is ~50 ms against a full-program timed run. Deriving duplicate-ness from the object also
replaces per-flag rules written by hand, which are only ever right about the toolchain they were
measured on.
"""
from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

#: ``objdump -d`` symbol header: ``0000000000000000 <k_fp64>:``.
SYMBOL_LINE = re.compile(r"^[0-9a-f]+ <([^>]+)>:$")
#: ``objdump -d`` instruction line: ``   1e:\tmovsd  0x0(%rip),%xmm1``.
INSN_LINE = re.compile(r"^\s*[0-9a-f]+:\t(.*)$")
#: objdump's trailing relocation comment, which carries an address. Two+ spaces before the ``#`` so an
#: aarch64 immediate (``mov x0, #1``, one space) is never mistaken for one.
INSN_COMMENT = re.compile(r"\s{2,}#.*$")
#: A branch/call target address that objdump already annotates with the symbol it resolves to
#: (``call 1090 <sin@plt>``). The slot moves with PLT layout, the symbol does not -- and a bare hex run
#: only ever appears here, since a real immediate carries its ``0x``.
BRANCH_TARGET = re.compile(r"\b[0-9a-f]+ (?=<)")

TOOL_TIMEOUT_S: float = 120.0


def tool_stdout(cmd: List[str], stdin: Optional[str] = None) -> Optional[str]:
    """stdout of ``cmd``, or ``None`` if it is missing or fails -- a dedup that cannot run must degrade
    into measuring both variants, never into collapsing them."""
    try:
        done = subprocess.run(cmd, input=stdin, capture_output=True, text=True, timeout=TOOL_TIMEOUT_S)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout if done.returncode == 0 else None


def clang_formatted(code: str) -> str:
    """``code`` with clang-format's spelling, so a key means "same code" and not "same layout".

    Falls back to the input unchanged when clang-format is absent: that direction only costs an extra
    measurement, whereas guessing at normalization could collapse two builds that differ.
    """
    out = tool_stdout(["clang-format", "--assume-filename=nest.cpp"], stdin=code)
    return out if out is not None else code


def function_bodies(code: str) -> List[str]:
    """Every top-level ``{...}`` block, brace-matched outside string/char literals and comments.

    Bodies rather than the whole translation unit: an emitted TU carries ``#include`` paths and name
    comments that follow the build directory, so two builds of identical code would not match on it.
    """
    bodies: List[str] = []
    depth, start, i, n = 0, -1, 0, len(code)
    while i < n:
        c = code[i]
        nxt = code[i + 1] if i + 1 < n else ""
        if c == "/" and nxt == "/":
            end = code.find("\n", i)
            i = n if end < 0 else end
        elif c == "/" and nxt == "*":
            end = code.find("*/", i + 2)
            i = n if end < 0 else end + 2
        elif c in "\"'":
            i += 1
            while i < n and code[i] != c:
                i += 2 if code[i] == "\\" else 1  # a backslash escapes the next char, including the quote
            i += 1
        else:
            if c == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif c == "}" and depth > 0:
                depth -= 1
                if depth == 0:
                    bodies.append(code[start:i + 1])
                    start = -1
            i += 1
    return bodies


def cpp_body_key(code: str) -> str:
    """Key over ``code``'s function bodies. Equal keys == the same generated source, so the pair differs
    only in compile flags -- which this key cannot see (see the module docstring)."""
    bodies = "\n".join(function_bodies(clang_formatted(code)))
    # clang-format keeps blank lines (MaxEmptyLinesToKeep), and they are never semantic
    return hashlib.sha256("\n".join(ln for ln in bodies.splitlines() if ln.strip()).encode()).hexdigest()


def parse_disassembly(out: str) -> Dict[str, str]:
    """``objdump -d --no-show-raw-insn`` text -> symbol -> its instruction text, with addresses and
    relocation comments dropped.

    Normalization stays minimal on purpose: this detects IDENTICAL code, and rewriting immediates would
    erase the very difference (a ``$0x2`` stride against a ``$0x3``) the key exists to see. Split from
    :func:`asm_bodies` so the normalization is testable without a compiler in the loop.
    """
    bodies: Dict[str, List[str]] = {}
    current: Optional[str] = None
    for line in out.splitlines():
        header = SYMBOL_LINE.match(line)
        if header:
            current = header.group(1)
            bodies[current] = []
            continue
        insn = INSN_LINE.match(line)
        if insn and current is not None:
            text = BRANCH_TARGET.sub("", INSN_COMMENT.sub("", insn.group(1)))
            bodies[current].append(text.rstrip())
    return {name: "\n".join(lines) for name, lines in bodies.items()}


def asm_bodies(obj: Path) -> Dict[str, str]:
    """symbol -> instruction text for ``obj``; empty when objdump is missing or the file has no code."""
    out = tool_stdout(["objdump", "-d", "--no-show-raw-insn", str(obj)])
    return parse_disassembly(out) if out is not None else {}


def asm_text(bodies: Mapping[str, str], obj: Path, symbol: Optional[str]) -> str:
    """The instruction text to key: one ``symbol``, or every symbol when ``None``."""
    if symbol is None:
        return "\n".join(f"{name}\n{bodies[name]}" for name in sorted(bodies))
    if symbol not in bodies:
        raise LookupError(f"symbol {symbol!r} not in {obj} (has: {sorted(bodies)})")
    return bodies[symbol]


def asm_body_key(obj: Path, symbol: Optional[str] = None) -> str:
    """Key over ``obj``'s disassembly -- one ``symbol``, or every symbol when ``None``.

    Name a symbol only when it is the code that RUNS. A DaCe ``__program_<name>`` is a trampoline into
    ``__program_<name>_internal`` and disassembles the same whatever the body does, so keying on it
    collapses every variant into one. The whole-object key also covers init/exit, which is the right
    direction anyway: init allocates the persistent storage, so a build that allocates differently is a
    different build. Over-separating costs an extra measurement, over-collapsing deletes a search axis.
    """
    bodies = asm_bodies(obj)
    if not bodies:
        raise LookupError(f"no disassembly for {obj} (objdump missing, or the object has no code)")
    return hashlib.sha256(asm_text(bodies, obj, symbol).encode()).hexdigest()


#: ``objdump -p`` dependency line: ``  NEEDED               libmvec.so.1``.
NEEDED_LINE = re.compile(r"^\s*NEEDED\s+(\S+)$")


def needed_libraries(path: Path) -> Tuple[str, ...]:
    """``DT_NEEDED`` of a linked artifact, sorted.

    The object key cannot close a LINK-time axis: on gcc a veclib contributes no compile flag at all
    (``-ffast-math`` alone emits ``_ZGV*``), so ``libmvec`` and ``sleef`` produce the same object and
    differ only in who resolves the call. Compose this with :func:`asm_body_key` to key a ``.so``.
    """
    out = tool_stdout(["objdump", "-p", str(path)])
    if out is None:
        return ()
    return tuple(sorted(m.group(1) for m in (NEEDED_LINE.match(ln) for ln in out.splitlines()) if m))


def variant_key(artifact: Path, symbol: Optional[str] = None) -> Optional[str]:
    """One key for a BUILT artifact: its code and its link together, or ``None`` when it cannot be read.

    Both halves are load-bearing -- code alone collapses a gnu veclib cell (link-only there), link alone
    collapses every fp rung. ``None`` rather than a raise so a caller can fall back to measuring the
    variant; a failure to inspect must never read as "same as the last one".
    """
    bodies = asm_bodies(artifact)  # one objdump: going through asm_body_key would disassemble twice
    if not bodies:
        return None
    code = hashlib.sha256(asm_text(bodies, artifact, symbol).encode()).hexdigest()
    return hashlib.sha256("\n".join((code, *needed_libraries(artifact))).encode()).hexdigest()


def collapse(keys: Mapping[str, str]) -> Dict[str, List[str]]:
    """``key -> variants sharing it``, insertion-ordered. Each group's FIRST member is the one to
    measure; the rest are that same build under another name."""
    groups: Dict[str, List[str]] = {}
    for name, key in keys.items():
        groups.setdefault(key, []).append(name)
    return groups


def representatives(keys: Mapping[str, str]) -> Tuple[List[str], List[str]]:
    """``(measure_these, collapsed_notes)``. The notes exist because a silent collapse reads exactly like
    a sweep that covered everything; a caller that drops variants must be able to say which and why."""
    groups = collapse(keys)
    picks = [members[0] for members in groups.values()]
    notes = [f"{members[0]} == {', '.join(members[1:])}" for members in groups.values() if len(members) > 1]
    return picks, notes
