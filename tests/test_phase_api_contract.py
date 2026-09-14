# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Contract tests for skills/*/SKILL.md -- the domain knowledge the agents read.

A skill file is a prompt: an agent reads it and copies the snippet verbatim. So a stale import in a
code block or a leaked local path is not a documentation nit, it is a runtime failure or a privacy
leak in the one consumer that cannot debug it.
"""
import ast
import importlib
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SKILLS = sorted((REPO / "skills").glob("*/SKILL.md"))

#: /home/... and /tmp/... are always machine-specific; a skill must read the same on any box.
ABSOLUTE_PATH_RE = re.compile(r"/home/\S*|/tmp/\S*")
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def skill_imports(path: Path):
    """``(module, name)`` for every ``from X import a, b`` inside a ```python block of one skill."""
    for block in re.findall(r"```python\n(.*?)```", path.read_text(), re.S):
        try:
            tree = ast.parse(block)
        except SyntaxError:
            continue  # an illustrative fragment, not runnable code
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                for alias in node.names:
                    yield node.module, alias.name


def test_a_skill_exists_for_dace_and_python():
    """The three domain skills an agent needs are present."""
    names = {p.parent.name for p in SKILLS}
    assert {"dace", "python-quality", "python-to-numpy"} <= names


@pytest.mark.parametrize("skill", SKILLS, ids=lambda p: p.parent.name)
def test_skill_has_frontmatter_with_name_and_description(skill):
    """An agent selects a skill by its description, so both fields must be present and non-trivial."""
    text = skill.read_text()
    assert text.startswith("---\n"), f"{skill} has no YAML frontmatter"
    front = text.split("---\n", 2)[1]
    assert re.search(r"^name: \S+", front, re.M), f"{skill} frontmatter has no name"
    described = re.search(r"^description: (.+)$", front, re.M | re.S)
    assert described and len(described.group(1)) > 60, f"{skill} needs a description that says WHEN to use it"


@pytest.mark.parametrize("skill", SKILLS, ids=lambda p: p.parent.name)
def test_every_symbol_a_skill_imports_exists(skill):
    """Copy-paste is the point: an import in a skill snippet must resolve, or the agent gets ImportError."""
    for module, name in skill_imports(skill):
        mod = importlib.import_module(module)
        assert name in vars(mod), f"{skill}: `from {module} import {name}` -- {module} has no {name!r}"


@pytest.mark.parametrize("skill", SKILLS, ids=lambda p: p.parent.name)
def test_skill_has_no_absolute_path(skill):
    """A skill teaches domain knowledge that must read the same on any machine, so no local path leaks in."""
    hits = ABSOLUTE_PATH_RE.findall(skill.read_text())
    assert not hits, f"{skill} contains an absolute path: {hits}"


@pytest.mark.parametrize("skill", SKILLS, ids=lambda p: p.parent.name)
def test_skill_has_no_email_address(skill):
    """A skill is committed to the repo, so it must carry no personal contact information."""
    hits = EMAIL_RE.findall(skill.read_text())
    assert not hits, f"{skill} contains an email address: {hits}"
