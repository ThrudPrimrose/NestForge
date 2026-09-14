# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Collapse variants that are the same build twice, so the sweep measures each distinct build once:
``cpp_body_key`` sees codegen only, ``asm_body_key`` sees the disassembly where flag duplicates show up."""

from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path
from collections.abc import Mapping

SYMBOL_LINE = re.compile(r"^[0-9a-f]+ <([^>]+)>:$")  # objdump -d symbol header
INSN_LINE = re.compile(r"^\s*[0-9a-f]+:\t(.*)$")  # objdump -d instruction line
#: two+ spaces before ``#`` so an aarch64 one-space immediate (``mov x0, #1``) is never mistaken for one.
INSN_COMMENT = re.compile(r"\s{2,}#.*$")
#: objdump's own branch-target address, dropped since the PLT slot moves but the annotated symbol does not.
BRANCH_TARGET = re.compile(r"\b[0-9a-f]+ (?=<)")

TOOL_TIMEOUT_S: float = 120.0


def tool_stdout(cmd: list[str], stdin: str | None = None) -> str | None:
    """stdout of ``cmd``, or ``None`` on failure -- callers must degrade to measuring, never collapsing."""
    try:
        done = subprocess.run(cmd, input=stdin, capture_output=True, text=True, timeout=TOOL_TIMEOUT_S)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout if done.returncode == 0 else None


def clang_formatted(code: str) -> str:
    """``code`` with clang-format's spelling, so a key means "same code", not "same layout"; falls
    back to ``code`` unchanged when clang-format is absent."""
    out = tool_stdout(["clang-format", "--assume-filename=nest.cpp"], stdin=code)
    return out if out is not None else code


def function_bodies(code: str) -> list[str]:
    """Every top-level ``{...}`` block, brace-matched outside string/char literals and comments."""
    bodies: list[str] = []
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
                    bodies.append(code[start : i + 1])
                    start = -1
            i += 1
    return bodies


def cpp_body_key(code: str) -> str:
    """Key over ``code``'s function bodies: equal keys mean the same source, flags aside."""
    bodies = "\n".join(function_bodies(clang_formatted(code)))
    # clang-format keeps blank lines (MaxEmptyLinesToKeep), and they are never semantic
    return hashlib.sha256("\n".join(ln for ln in bodies.splitlines() if ln.strip()).encode()).hexdigest()


def parse_disassembly(out: str) -> dict[str, str]:
    """``objdump -d --no-show-raw-insn`` text -> symbol -> its instruction text, with addresses and
    relocation comments dropped, but immediates left alone since they are what the key must see."""
    bodies: dict[str, list[str]] = {}
    current: str | None = None
    for line in out.splitlines():
        header = SYMBOL_LINE.match(line)
        if header:
            symbol = header.group(1)
            if not isinstance(symbol, str):
                continue  # the pattern's one group is mandatory; unreachable on a real match, but narrows the type
            current = symbol
            bodies[current] = []
            continue
        insn = INSN_LINE.match(line)
        if insn and current is not None:
            text = BRANCH_TARGET.sub("", INSN_COMMENT.sub("", insn.group(1)))
            bodies[current].append(text.rstrip())
    return {name: "\n".join(lines) for name, lines in bodies.items()}


def asm_bodies(obj: Path) -> dict[str, str]:
    """symbol -> instruction text for ``obj``; empty when objdump is missing or the file has no code."""
    out = tool_stdout(["objdump", "-d", "--no-show-raw-insn", str(obj)])
    return parse_disassembly(out) if out is not None else {}


def asm_text(bodies: Mapping[str, str], obj: Path, symbol: str | None) -> str:
    """The instruction text to key: one ``symbol``, or every symbol when ``None``."""
    if symbol is None:
        return "\n".join(f"{name}\n{bodies[name]}" for name in sorted(bodies))
    if symbol not in bodies:
        raise LookupError(f"symbol {symbol!r} not in {obj} (has: {sorted(bodies)})")
    return bodies[symbol]


def asm_body_key(obj: Path, symbol: str | None = None) -> str:
    """Key over ``obj``'s disassembly -- one ``symbol`` or every symbol when ``None``; name a symbol
    only when it is the code that RUNS (a DaCe trampoline disassembles the same regardless of body)."""
    bodies = asm_bodies(obj)
    if not bodies:
        raise LookupError(f"no disassembly for {obj} (objdump missing, or the object has no code)")
    return hashlib.sha256(asm_text(bodies, obj, symbol).encode()).hexdigest()


NEEDED_LINE = re.compile(r"^\s*NEEDED\s+(\S+)$")  # objdump -p dependency line


def needed_libraries(path: Path) -> tuple[str, ...]:
    """``DT_NEEDED`` of a linked artifact, sorted (a link-only axis the object key alone cannot see)."""
    out = tool_stdout(["objdump", "-p", str(path)])
    if out is None:
        return ()
    return tuple(sorted(m.group(1) for m in (NEEDED_LINE.match(ln) for ln in out.splitlines()) if m))


def variant_key(artifact: Path, symbol: str | None = None) -> str | None:
    """One key for a BUILT artifact: its code and its link together, or ``None`` when it cannot be
    read (a caller falls back to measuring; a failure to inspect must never read as "same as before")."""
    bodies = asm_bodies(artifact)  # one objdump: going through asm_body_key would disassemble twice
    if not bodies:
        return None
    code = hashlib.sha256(asm_text(bodies, artifact, symbol).encode()).hexdigest()
    return hashlib.sha256("\n".join((code, *needed_libraries(artifact))).encode()).hexdigest()


def collapse(keys: Mapping[str, str]) -> dict[str, list[str]]:
    """``key -> variants sharing it``; each group's first member is the one to measure."""
    groups: dict[str, list[str]] = {}
    for name, key in keys.items():
        groups.setdefault(key, []).append(name)
    return groups


def representatives(keys: Mapping[str, str]) -> tuple[list[str], list[str]]:
    """``(measure_these, collapsed_notes)``: notes record which variants were dropped, and why."""
    groups = collapse(keys)
    picks = [members[0] for members in groups.values()]
    notes = [f"{members[0]} == {', '.join(members[1:])}" for members in groups.values() if len(members) > 1]
    return picks, notes
