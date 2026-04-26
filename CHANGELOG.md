# Changelog

All notable changes to this project will be documented in this file.

## [0.3.0] - 2026-04-26

### Added

- **Append-only execution trace** (`agent/` workspace region)
  - Every pipeline run now writes `trace.jsonl` and `session.json` to `output/<project>/agent/`.
  - `trace.jsonl` is append-only (opened in `"a"` mode, flushed after every write). Contains one JSON line per event: tool calls, agent spawns, task lifecycle, errors, user questions, and the final pipeline result.
  - `session.json` is written at pipeline start (`status: "running"`) and updated at pipeline end with cost, duration, turns, and final status. If a run crashes, the file retains `"running"` and the trace contains all events up to the crash.
  - New module: `agents/trace.py` with the `TraceWriter` class.

- **Structural write guards** (per-agent directory enforcement)
  - Sub-agents are now restricted to their own output subdirectory via a `PreToolUse` SDK hook that fires on every `Write`/`Edit` call. The hook reads `agent_type` from the hook input and checks it against `AGENT_WRITE_PERMISSIONS` in `config.py`. Unauthorized writes are denied with a reason message.
  - Each agent's system prompt now includes a `## Write Boundary` section as a complementary soft guard.
  - The `agent/` directory is protected from all sub-agents — only the Director's `TraceWriter` writes there via direct file I/O.

- **Write permission registry** (`config.py`)
  - New `AGENT_WRITE_PERMISSIONS` dict maps each agent name to its allowed output subdirectories. Single source of truth for both the hook and the tests.

- **8 new tests** (`tests/test_hardening.py`)
  - `agent/` directory creation, prompt write-boundary validation, registry completeness, agent/ protection, JSONL schema, session lifecycle, append-only semantics, and cross-boundary hook enforcement.

### Changed

- `config.py`: `create_project_output()` now creates a 5th subdirectory (`agent/`) and includes it in the returned paths dict.
- `agents/director.py`: `_log_message()` accepts an optional `TraceWriter` and writes every message type to the trace alongside existing terminal output. `run_director()` creates the writer, initializes the session, registers the write guard hook, and closes the writer in a `try/finally`.
- Agent prompts in `coder.py`, `reviewer.py`, `planner.py`, `arxiv_reader.py` now contain `## Write Boundary` sections.

## [0.2.0] - 2026-04-12

### Added

- CAID framework implementation with installation guide and configuration reference.
- Examples directory with research swarm description.
- Director workflow improvements and JEPA implementation example.
- User interaction improvements and paper download tests.

## [0.1.0] - 2026-03-22

### Added

- Initial release of the Research Swarm pipeline.
- 5-stage pipeline: Read, Plan, Code, Review, Deliver.
- Director agent with colored terminal logging and human-in-the-loop approval.
- Local paper reader and ArXiv reader (via MCP) sub-agents.
- Planner, Coder, and Reviewer sub-agents with role-specific skills.
- 10 skills with tested isolation across agent boundaries.
- Slash commands for Claude Code integration.
