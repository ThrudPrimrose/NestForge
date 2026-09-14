# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""CI contract tests for the AGENT-FACING phase API -- the surface `skills/*/SKILL.md` teaches.

A skill file is a prompt: an agent reads it and copies the snippet verbatim. So a stale name there is
not a documentation nit, it is a runtime failure in the one consumer that cannot debug it. Two such
drifts already shipped -- `strategy_names()` lost half its entries after `offload.py` registered the
per-unit strategies into the same registry, and `nestforge.optimize` was shadowed by the re-exported
`optimize` FUNCTION, so `nestforge.optimize.optimization_choices` raised AttributeError on a function.

These tests scan the skill files rather than pinning one known name, because the bug class recurs.
"""
import ast
import importlib
import re
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SKILLS = sorted((REPO / "skills").glob("*/SKILL.md"))
#: Just the four phase skills. `agent-graph` is the overview an agent reads first, not a phase.
PHASE_SKILLS = [p for p in SKILLS if p.parent.name.startswith("phase")]

#: The four phases, in cycle order. Each MUST stay reachable as a module: the skills say
#: ``from nestforge.<phase> import ...``, and an agent that instead reaches ``nestforge.<phase>``
#: must get the module, not a same-named re-export.
PHASE_MODULES = ("fusion", "offload", "optimize", "feedback")


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


def test_skills_exist_for_every_phase():
    names = {p.parent.name for p in SKILLS}
    assert {f"phase{i}-{n}" for i, n in enumerate(("fusion", "offload", "optimize", "feedback"), 1)} <= names
    assert "agent-graph" in names, "the entry skill an agent reads first"


@pytest.mark.parametrize("skill", SKILLS, ids=lambda p: p.parent.name)
def test_skill_has_frontmatter_with_name_and_description(skill):
    """An agent selects a skill by its description, so both fields must be present and non-trivial."""
    text = skill.read_text()
    assert text.startswith("---\n"), f"{skill} has no YAML frontmatter"
    front = text.split("---\n", 2)[1]
    assert re.search(r"^name: \S+", front, re.M), f"{skill} frontmatter has no name"
    described = re.search(r"^description: (.+)$", front, re.M)
    assert described and len(described.group(1)) > 60, f"{skill} needs a description that says WHEN to use it"


@pytest.mark.parametrize("skill", SKILLS, ids=lambda p: p.parent.name)
def test_every_symbol_a_skill_imports_exists(skill):
    """Copy-paste is the point: an import in a skill snippet must resolve, or the agent gets ImportError."""
    for module, name in skill_imports(skill):
        mod = importlib.import_module(module)
        assert name in vars(mod), f"{skill}: `from {module} import {name}` -- {module} has no {name!r}"


@pytest.mark.parametrize("phase", PHASE_MODULES)
def test_phase_is_reachable_as_a_module(phase):
    """``nestforge.<phase>`` must be the MODULE.

    Re-exporting a phase's commit function at package top level binds its name over the submodule, so
    ``nestforge.optimize.optimization_choices`` fails on a function object. Keep the four uniform.
    """
    import nestforge
    bound = vars(nestforge)[phase]
    assert isinstance(bound, types.ModuleType), (
        f"nestforge.{phase} is a {type(bound).__name__}, not the submodule -- a top-level re-export is "
        f"shadowing it")


def test_skills_quote_the_real_strategy_names():
    """A skill that PRINTS a registry's contents must print all of it.

    `strategy_names()` silently grew the three offload units; the phase-2 skill kept showing three.
    """
    from nestforge.phases.schedule import fusion_strategy_names
    from nestforge.phases.scopes import strategy_names

    quoted = {
        "phase1-fusion": (fusion_strategy_names(), r"fusion_strategy_names\(\)\s*#\s*(\[[^\]]*\])"),
        "phase2-offload": (strategy_names(), r"strategy_names\(\)\s*#\s*(\[[^\]]*\])"),
    }
    for skill_dir, (actual, pattern) in quoted.items():
        text = (REPO / "skills" / skill_dir / "SKILL.md").read_text()
        shown = re.search(pattern, text)
        assert shown, f"{skill_dir} no longer shows the registry contents"
        assert ast.literal_eval(
            shown.group(1)) == list(actual), (f"{skill_dir} shows {shown.group(1)}, registry holds {list(actual)}")


@pytest.mark.parametrize("skill", PHASE_SKILLS, ids=lambda p: p.parent.name)
def test_phase_skills_state_preconditions_and_guardrails(skill):
    """Prompt hygiene: a phase skill must say what must already hold and what must never be done.

    Without both, an agent applies a phase out of order (Phase 1 cannot see inside an ExternalCall) or
    around the legality gate, and gets a wrong program rather than a refusal.
    """
    text = skill.read_text()
    assert "## Preconditions" in text, f"{skill} must state what must already hold"
    assert "## Guardrails" in text, f"{skill} must state what must never be done"


@pytest.mark.parametrize("skill", SKILLS, ids=lambda p: p.parent.name)
def test_skill_prose_never_names_a_session_tool_that_is_gone(skill):
    """A tool named in BACKTICKS is a tool an agent will call, and prose is where the roster is listed.

    ``skill_imports`` only parses ```python blocks, so three tools deleted from ``Session`` survived in the
    agent-graph roster for several commits: an agent copying that list gets ``AttributeError`` (``Session``
    has no ``__getattr__``). Only names that read as tool calls are checked -- a backticked word that is
    also a Session attribute is enough to be worth keeping true.
    """
    from nestforge.session import Session
    live = {n for n in vars(Session) if not n.startswith("__")}
    # A skill also names FREE functions (`fission_to_statements`) and DaCe methods (`can_be_applied_to`),
    # which are not Session tools and must not be flagged. So the roster is Session's surface plus every
    # public name of the modules the skill's own code blocks import from.
    for module, _ in skill_imports(skill):
        if module.startswith("nestforge") or module.startswith("dace"):
            live |= {n for n in vars(importlib.import_module(module)) if not n.startswith("_")}
    # the vocabulary this guard polices: verbs the phase API actually uses, so an unrelated backticked
    # word (a field name, a python builtin) is never asserted against the surface
    # `can_fuse`, not `can_`: `can_be_applied_to` is a DaCe transformation METHOD, never a module name
    verbs = ("list_", "fuse", "fission", "emit_", "set_", "sweep", "feedback", "externalize", "region_tree",
             "nest_boundary", "can_fuse")
    named = {m for m in re.findall(r"`([a-z_][a-z0-9_]*)`", skill.read_text()) if m.startswith(verbs)}
    gone = sorted(named - live)
    assert not gone, f"{skill} names tools that no longer exist: {gone}"
