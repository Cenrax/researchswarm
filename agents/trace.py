"""Append-only execution trace and session metadata.

Provides durable logging for the research-to-code pipeline. Two files are
written to the ``agent/`` directory of each project:

- ``trace.jsonl`` — one JSON line per event (tool calls, results, agent
  spawns, errors, task lifecycle). Opened in append mode; never truncated.
- ``session.json`` — written once at pipeline start, updated at pipeline end
  with cost, duration, and final status.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class TraceWriter:
    """Append-only JSONL trace writer.

    Opens ``trace.jsonl`` in append mode so that multiple writer instances
    (or crash-and-restart scenarios) never lose prior entries.
    """

    def __init__(self, agent_dir: Path) -> None:
        self._trace_path = agent_dir / "trace.jsonl"
        self._session_path = agent_dir / "session.json"
        self._file = open(self._trace_path, "a", encoding="utf-8")  # noqa: SIM115
        self.start_time: str | None = None

    # ── Trace events ────────────────────────────────────────────────────────

    def write_event(
        self,
        event_type: str,
        *,
        agent_name: str | None = None,
        tool_name: str | None = None,
        summary: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Append one JSON line to ``trace.jsonl``."""
        entry: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
        }
        if agent_name is not None:
            entry["agent_name"] = agent_name
        if tool_name is not None:
            entry["tool_name"] = tool_name
        if summary:
            entry["summary"] = summary[:500]
        if metadata:
            entry["metadata"] = metadata
        self._file.write(json.dumps(entry, default=str) + "\n")
        self._file.flush()

    # ── Session metadata ────────────────────────────────────────────────────

    def write_session_start(
        self,
        objective: str,
        mode: str,
        papers: list[str],
    ) -> None:
        """Write initial ``session.json`` with status ``running``."""
        self.start_time = datetime.now(timezone.utc).isoformat()
        session = {
            "objective": objective,
            "mode": mode,
            "papers": papers,
            "start_time": self.start_time,
            "end_time": None,
            "duration_s": None,
            "total_cost_usd": None,
            "num_turns": None,
            "status": "running",
        }
        self._session_path.write_text(
            json.dumps(session, indent=2), encoding="utf-8",
        )

    def write_session_end(
        self,
        *,
        total_cost_usd: float | None,
        num_turns: int,
        duration_s: float,
        is_error: bool,
    ) -> None:
        """Update ``session.json`` at pipeline end, preserving original fields."""
        existing = json.loads(self._session_path.read_text(encoding="utf-8"))
        existing.update({
            "end_time": datetime.now(timezone.utc).isoformat(),
            "duration_s": round(duration_s, 1),
            "total_cost_usd": total_cost_usd,
            "num_turns": num_turns,
            "status": "error" if is_error else "completed",
        })
        self._session_path.write_text(
            json.dumps(existing, indent=2), encoding="utf-8",
        )

    # ── Lifecycle ───────────────────────────────────────────────────────────

    def close(self) -> None:
        """Flush and close the trace file."""
        if self._file and not self._file.closed:
            self._file.flush()
            self._file.close()
