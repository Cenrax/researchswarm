#!/usr/bin/env python3
"""Tests to verify that each agent has exactly the skills assigned to it.

Ensures:
  - Each agent only has its designated skills (no leakage)
  - Each skill is assigned to exactly the agent(s) it belongs to
  - All skills have corresponding SKILL.md files on disk
  - Agent tools include "Skill" when skills are assigned
  - No duplicate skills across unrelated agents

Run:
    python -m pytest tests/test_agent_skills.py -v
    # or directly:
    python tests/test_agent_skills.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.arxiv_reader import make_local_reader, make_arxiv_reader
from agents.planner import make_planner
from agents.coder import make_coder
from agents.reviewer import make_reviewer

# ── Expected skill assignments ───────────────────────────────────────────────

EXPECTED_SKILLS = {
    "paper-reader":  ["summarize-paper", "extract-equations"],
    "arxiv-reader":  ["summarize-paper", "extract-equations"],
    "planner":       ["architecture-design", "dependency-analysis"],
    "coder":         ["sandbox-execute", "write-tests"],
    "reviewer":      ["paper-alignment-check", "security-audit"],
}

ALL_SKILLS = {
    "pipeline-status",
    "iterate-feedback",
    "summarize-paper",
    "extract-equations",
    "architecture-design",
    "dependency-analysis",
    "sandbox-execute",
    "write-tests",
    "paper-alignment-check",
    "security-audit",
}

# Skills that should NOT be shared across agent boundaries
# (readers share skills, but coder should never have reviewer skills, etc.)
EXCLUSIVE_GROUPS = [
    ({"paper-reader", "arxiv-reader"}, {"summarize-paper", "extract-equations"}),
    ({"planner"},                       {"architecture-design", "dependency-analysis"}),
    ({"coder"},                         {"sandbox-execute", "write-tests"}),
    ({"reviewer"},                      {"paper-alignment-check", "security-audit"}),
]

# ── Helpers ──────────────────────────────────────────────────────────────────

SKILLS_DIR = Path(__file__).resolve().parent.parent / ".claude" / "skills"


def _all_agents() -> dict:
    """Build all agent definitions and return as {name: definition}."""
    agents = {}
    for factory in (make_local_reader, make_arxiv_reader, make_planner, make_coder, make_reviewer):
        name, defn = factory()
        agents[name] = defn
    return agents


# ── Tests ────────────────────────────────────────────────────────────────────

def test_each_agent_has_correct_skills():
    """Each agent's skills list matches the expected assignment."""
    agents = _all_agents()
    for name, defn in agents.items():
        expected = sorted(EXPECTED_SKILLS[name])
        actual = sorted(defn.skills or [])
        assert actual == expected, (
            f"Agent '{name}' has skills {actual}, expected {expected}"
        )
        print(f"  OK: {name} -> {actual}")


def test_no_skill_leakage():
    """No agent has skills from another agent's exclusive group."""
    agents = _all_agents()
    for agent_name, defn in agents.items():
        agent_skills = set(defn.skills or [])
        for allowed_agents, group_skills in EXCLUSIVE_GROUPS:
            if agent_name not in allowed_agents:
                leaked = agent_skills & group_skills
                assert not leaked, (
                    f"Agent '{agent_name}' has leaked skills {leaked} "
                    f"which belong to {allowed_agents}"
                )
    print("  OK: No skill leakage detected")


def test_all_agents_have_skill_tool():
    """Every agent with skills must have 'Skill' in its tools list."""
    agents = _all_agents()
    for name, defn in agents.items():
        if defn.skills:
            assert "Skill" in (defn.tools or []), (
                f"Agent '{name}' has skills {defn.skills} but 'Skill' "
                f"is not in its tools: {defn.tools}"
            )
    print("  OK: All skilled agents have 'Skill' tool")


def test_skill_files_exist():
    """Every skill has a SKILL.md file on disk."""
    for skill_name in ALL_SKILLS:
        skill_file = SKILLS_DIR / skill_name / "SKILL.md"
        assert skill_file.exists(), (
            f"Skill '{skill_name}' is missing SKILL.md at {skill_file}"
        )
    print(f"  OK: All {len(ALL_SKILLS)} skill files exist")


def test_no_orphan_skill_files():
    """Every SKILL.md on disk is assigned to at least one agent or the director."""
    agents = _all_agents()
    all_assigned = set()
    for defn in agents.values():
        all_assigned.update(defn.skills or [])
    # Director skills are defined in director.py, not in an AgentDefinition
    all_assigned.update(["pipeline-status", "iterate-feedback"])

    on_disk = {
        d.name for d in SKILLS_DIR.iterdir()
        if d.is_dir() and (d / "SKILL.md").exists()
    }

    orphans = on_disk - all_assigned
    assert not orphans, (
        f"Orphan skills on disk not assigned to any agent: {orphans}"
    )
    print(f"  OK: No orphan skill files")


def test_skill_files_have_description():
    """Every SKILL.md has a description in its frontmatter."""
    for skill_name in ALL_SKILLS:
        skill_file = SKILLS_DIR / skill_name / "SKILL.md"
        content = skill_file.read_text()
        assert "---" in content, (
            f"Skill '{skill_name}' SKILL.md has no YAML frontmatter"
        )
        assert "description:" in content, (
            f"Skill '{skill_name}' SKILL.md is missing 'description' field"
        )
    print(f"  OK: All skills have descriptions")


def test_readers_share_same_skills():
    """paper-reader and arxiv-reader must have identical skills."""
    agents = _all_agents()
    local_skills = sorted(agents["paper-reader"].skills or [])
    arxiv_skills = sorted(agents["arxiv-reader"].skills or [])
    assert local_skills == arxiv_skills, (
        f"Readers have different skills: local={local_skills}, arxiv={arxiv_skills}"
    )
    print(f"  OK: Both readers share skills: {local_skills}")


def test_coder_has_no_review_skills():
    """Coder must never have reviewer skills (separation of concerns)."""
    agents = _all_agents()
    coder_skills = set(agents["coder"].skills or [])
    reviewer_skills = set(agents["reviewer"].skills or [])
    overlap = coder_skills & reviewer_skills
    assert not overlap, (
        f"Coder and Reviewer share skills (should not): {overlap}"
    )
    print("  OK: Coder and Reviewer skills are disjoint")


def test_reviewer_has_no_execute_skills():
    """Reviewer must never have execution skills (read-only review)."""
    agents = _all_agents()
    reviewer_skills = set(agents["reviewer"].skills or [])
    execute_skills = {"sandbox-execute", "write-tests"}
    overlap = reviewer_skills & execute_skills
    assert not overlap, (
        f"Reviewer has execution skills (should not): {overlap}"
    )
    print("  OK: Reviewer has no execution skills")


def test_planner_has_no_execute_skills():
    """Planner must never have execution skills (planning only)."""
    agents = _all_agents()
    planner_skills = set(agents["planner"].skills or [])
    execute_skills = {"sandbox-execute", "write-tests"}
    overlap = planner_skills & execute_skills
    assert not overlap, (
        f"Planner has execution skills (should not): {overlap}"
    )
    print("  OK: Planner has no execution skills")


# ── Runner ───────────────────────────────────────────────────────────────────

def run_all():
    """Run all tests manually."""
    tests = [
        test_each_agent_has_correct_skills,
        test_no_skill_leakage,
        test_all_agents_have_skill_tool,
        test_skill_files_exist,
        test_no_orphan_skill_files,
        test_skill_files_have_description,
        test_readers_share_same_skills,
        test_coder_has_no_review_skills,
        test_reviewer_has_no_execute_skills,
        test_planner_has_no_execute_skills,
    ]

    passed = 0
    failed = 0

    for test in tests:
        name = test.__name__
        print(f"\n{'─'*60}")
        print(f"  {name}")
        print(f"{'─'*60}")
        try:
            test()
            passed += 1
            print(f"  -> PASSED")
        except AssertionError as e:
            failed += 1
            print(f"  -> FAILED: {e}")
        except Exception as e:
            failed += 1
            print(f"  -> ERROR: {type(e).__name__}: {e}")

    print(f"\n{'='*60}")
    print(f"  Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(run_all())
