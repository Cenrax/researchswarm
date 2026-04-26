#!/usr/bin/env python3
"""Tests for pipeline hardening features.

Verifies:
  - agent/ directory creation in project output
  - Write-boundary constraints in every agent prompt
  - Write-permission registry completeness
  - TraceWriter JSONL schema and append-only behaviour
  - Session.json lifecycle (start → end)
  - Write guard hook enforcement

Run:
    python -m pytest tests/test_hardening.py -v
    # or directly:
    python tests/test_hardening.py
"""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.arxiv_reader import make_local_reader, make_arxiv_reader
from agents.planner import make_planner
from agents.coder import make_coder
from agents.reviewer import make_reviewer
from config import AGENT_WRITE_PERMISSIONS

AGENT_FACTORIES = {
    "paper-reader": make_local_reader,
    "arxiv-reader": make_arxiv_reader,
    "planner": make_planner,
    "coder": make_coder,
    "reviewer": make_reviewer,
}


# ── Feature 1: agent/ directory ─────────────────────────────────────────────

def test_create_project_output_creates_agent_dir():
    """create_project_output() must create the agent/ directory."""
    from config import create_project_output

    paths = create_project_output("test-agent-dir-creation")
    try:
        agent_dir = paths.get("agent")
        assert agent_dir is not None, "create_project_output() missing 'agent' key"
        assert agent_dir.exists(), f"agent/ directory not created at {agent_dir}"
        assert agent_dir.is_dir(), f"{agent_dir} is not a directory"
        assert agent_dir.name == "agent", f"Expected 'agent', got '{agent_dir.name}'"
    finally:
        shutil.rmtree(paths["project_dir"], ignore_errors=True)
    print("  OK: agent/ directory created")


# ── Feature 2: write-boundary prompts ───────────────────────────────────────

def test_write_boundary_in_agent_prompts():
    """Each agent's prompt must contain a Write Boundary section."""
    for agent_name, factory in AGENT_FACTORIES.items():
        name, defn = factory()
        assert "Write Boundary" in defn.prompt, (
            f"Agent '{name}' prompt missing 'Write Boundary' section"
        )

        # Verify the prompt mentions the correct allowed directory
        allowed = AGENT_WRITE_PERMISSIONS[name]
        for d in allowed:
            assert f"output/{d}/" in defn.prompt, (
                f"Agent '{name}' prompt missing allowed dir 'output/{d}/'"
            )

        assert "FORBIDDEN" in defn.prompt, (
            f"Agent '{name}' prompt missing 'FORBIDDEN' keyword"
        )
    print("  OK: All agent prompts contain write boundary rules")


def test_write_permissions_registry_complete():
    """Every agent with Write or Edit in its tools must be in AGENT_WRITE_PERMISSIONS."""
    for factory in AGENT_FACTORIES.values():
        name, defn = factory()
        has_write = "Write" in (defn.tools or []) or "Edit" in (defn.tools or [])
        if has_write:
            assert name in AGENT_WRITE_PERMISSIONS, (
                f"Agent '{name}' has Write/Edit but no entry in AGENT_WRITE_PERMISSIONS"
            )
            assert len(AGENT_WRITE_PERMISSIONS[name]) > 0, (
                f"Agent '{name}' has empty write permissions list"
            )
    print("  OK: Write permissions registry covers all writing agents")


def test_no_agent_can_write_to_agent_dir():
    """No sub-agent's write permissions include the agent/ directory."""
    for agent_name, allowed in AGENT_WRITE_PERMISSIONS.items():
        assert "agent" not in allowed, (
            f"Agent '{agent_name}' should not have write access to agent/"
        )
    print("  OK: No sub-agent can write to agent/ directory")


# ── Feature 1 continued: TraceWriter ────────────────────────────────────────

def test_trace_jsonl_schema():
    """TraceWriter events must have the required schema."""
    from agents.trace import TraceWriter

    with tempfile.TemporaryDirectory() as tmpdir:
        agent_dir = Path(tmpdir)
        writer = TraceWriter(agent_dir)

        writer.write_event("tool_call", agent_name="director", tool_name="Read", summary="test")
        writer.write_event("agent_spawn", agent_name="director", summary="spawning coder")
        writer.write_event("tool_error", summary="file not found")
        writer.close()

        trace_path = agent_dir / "trace.jsonl"
        assert trace_path.exists(), "trace.jsonl not created"

        lines = trace_path.read_text().strip().split("\n")
        assert len(lines) == 3, f"Expected 3 lines, got {len(lines)}"

        required_fields = {"ts", "event_type"}
        for line in lines:
            entry = json.loads(line)
            missing = required_fields - set(entry.keys())
            assert not missing, f"Missing required fields: {missing} in {entry}"
            assert "T" in entry["ts"], f"Timestamp not ISO format: {entry['ts']}"

        first = json.loads(lines[0])
        assert first["event_type"] == "tool_call"
        assert first["agent_name"] == "director"
        assert first["tool_name"] == "Read"
    print("  OK: trace.jsonl entries have correct schema")


def test_session_json_lifecycle():
    """session.json must be written at start and updated at end."""
    from agents.trace import TraceWriter

    with tempfile.TemporaryDirectory() as tmpdir:
        agent_dir = Path(tmpdir)
        writer = TraceWriter(agent_dir)

        writer.write_session_start("build transformer", "arxiv", ["1706.03762"])
        session_path = agent_dir / "session.json"
        assert session_path.exists(), "session.json not created at start"

        data = json.loads(session_path.read_text())
        assert data["objective"] == "build transformer"
        assert data["mode"] == "arxiv"
        assert data["status"] == "running"
        assert data["end_time"] is None

        writer.write_session_end(
            total_cost_usd=0.1234,
            num_turns=10,
            duration_s=45.6,
            is_error=False,
        )

        data = json.loads(session_path.read_text())
        assert data["status"] == "completed"
        assert data["total_cost_usd"] == 0.1234
        assert data["num_turns"] == 10
        assert data["duration_s"] == 45.6
        assert data["end_time"] is not None
        # Original fields preserved
        assert data["objective"] == "build transformer"
        assert data["papers"] == ["1706.03762"]

        writer.close()
    print("  OK: session.json lifecycle correct")


def test_trace_file_is_append_only():
    """Multiple TraceWriter instances on the same file must append, not truncate."""
    from agents.trace import TraceWriter

    with tempfile.TemporaryDirectory() as tmpdir:
        agent_dir = Path(tmpdir)

        w1 = TraceWriter(agent_dir)
        w1.write_event("first_event", summary="first")
        w1.close()

        w2 = TraceWriter(agent_dir)
        w2.write_event("second_event", summary="second")
        w2.close()

        lines = (agent_dir / "trace.jsonl").read_text().strip().split("\n")
        assert len(lines) == 2, f"Expected 2 lines (append), got {len(lines)}"
        assert json.loads(lines[0])["event_type"] == "first_event"
        assert json.loads(lines[1])["event_type"] == "second_event"
    print("  OK: Trace file is append-only")


# ── Feature 2 continued: write guard hook ───────────────────────────────────

def test_write_guard_blocks_cross_boundary():
    """The write guard hook must block cross-boundary writes."""
    from agents.director import _write_guard_hook

    async def _run():
        # Coder writing to code/ — should be allowed
        result = await _write_guard_hook(
            {
                "hook_event_name": "PreToolUse",
                "session_id": "",
                "transcript_path": "",
                "cwd": "",
                "tool_name": "Write",
                "tool_input": {"file_path": "/project/output/20260101_000000_test/code/main.py"},
                "agent_type": "coder",
                "agent_id": "abc123",
            },
            "test-1",
            {"signal": None},
        )
        specific = result.get("hookSpecificOutput", {})
        assert specific.get("permissionDecision") != "deny", (
            "Coder should be allowed to write to code/"
        )

        # Coder writing to reviews/ — should be BLOCKED
        result = await _write_guard_hook(
            {
                "hook_event_name": "PreToolUse",
                "session_id": "",
                "transcript_path": "",
                "cwd": "",
                "tool_name": "Write",
                "tool_input": {"file_path": "/project/output/20260101_000000_test/reviews/review.md"},
                "agent_type": "coder",
                "agent_id": "abc123",
            },
            "test-2",
            {"signal": None},
        )
        specific = result.get("hookSpecificOutput", {})
        assert specific.get("permissionDecision") == "deny", (
            "Coder should be BLOCKED from writing to reviews/"
        )

        # Reviewer writing to reviews/ — should be allowed
        result = await _write_guard_hook(
            {
                "hook_event_name": "PreToolUse",
                "session_id": "",
                "transcript_path": "",
                "cwd": "",
                "tool_name": "Write",
                "tool_input": {"file_path": "/project/output/20260101_000000_test/reviews/review.md"},
                "agent_type": "reviewer",
                "agent_id": "def456",
            },
            "test-3",
            {"signal": None},
        )
        specific = result.get("hookSpecificOutput", {})
        assert specific.get("permissionDecision") != "deny", (
            "Reviewer should be allowed to write to reviews/"
        )

        # Coder writing to agent/ — should be BLOCKED
        result = await _write_guard_hook(
            {
                "hook_event_name": "PreToolUse",
                "session_id": "",
                "transcript_path": "",
                "cwd": "",
                "tool_name": "Write",
                "tool_input": {"file_path": "/project/output/20260101_000000_test/agent/trace.jsonl"},
                "agent_type": "coder",
                "agent_id": "abc123",
            },
            "test-4",
            {"signal": None},
        )
        specific = result.get("hookSpecificOutput", {})
        assert specific.get("permissionDecision") == "deny", (
            "Coder should be BLOCKED from writing to agent/"
        )

        # Director (no agent_type) — should pass through
        result = await _write_guard_hook(
            {
                "hook_event_name": "PreToolUse",
                "session_id": "",
                "transcript_path": "",
                "cwd": "",
                "tool_name": "Write",
                "tool_input": {"file_path": "/anywhere/file.txt"},
            },
            "test-5",
            {"signal": None},
        )
        specific = result.get("hookSpecificOutput", {})
        assert specific.get("permissionDecision") != "deny", (
            "Director should not be blocked by write guard"
        )

    asyncio.run(_run())
    print("  OK: Write guard hook blocks cross-boundary writes")


# ── Manual runner ───────────────────────────────────────────────────────────

def run_all():
    """Run all tests and report results."""
    tests = [
        test_create_project_output_creates_agent_dir,
        test_write_boundary_in_agent_prompts,
        test_write_permissions_registry_complete,
        test_no_agent_can_write_to_agent_dir,
        test_trace_jsonl_schema,
        test_session_json_lifecycle,
        test_trace_file_is_append_only,
        test_write_guard_blocks_cross_boundary,
    ]
    print(f"\nRunning {len(tests)} hardening tests...\n")
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"  FAIL: {test.__name__}: {e}")
    print(f"\n{'='*40}")
    print(f"  {passed} passed, {failed} failed")
    print(f"{'='*40}")
    return failed == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
