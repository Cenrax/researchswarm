"""Director agent — central orchestrator for the research swarm.

The Director manages the full pipeline:
  1. Load paper IDs from the latest week folder.
  2. Dispatch the ArXiv reader agent.
  3. Get user approval on summaries.
  4. Dispatch the planner agent.
  5. Get user approval on the plan.
  6. Dispatch the coder agent.
  7. Get user approval on the code.
  8. Dispatch the reviewer agent.
  9. Present final review to the user.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path

from claude_agent_sdk import (
    AgentDefinition,
    ClaudeAgentOptions,
    PermissionResultAllow,
    PermissionResultDeny,
    query,
)
from claude_agent_sdk.types import (
    AssistantMessage,
    ResultMessage,
    SystemMessage,
    TaskStartedMessage,
    TaskProgressMessage,
    TaskNotificationMessage,
    TextBlock,
    ToolUseBlock,
    ToolResultBlock,
    ThinkingBlock,
    UserMessage,
)

from config import (
    ARXIV_MCP_CONFIG,
    DIRECTOR_MODEL,
    create_project_output,
    get_latest_week_dir,
)

from agents.arxiv_reader import make_arxiv_reader, make_local_reader
from agents.planner import make_planner
from agents.coder import make_coder
from agents.reviewer import make_reviewer


# ── User approval callback ──────────────────────────────────────────────────

async def can_use_tool(tool_name: str, input_data: dict, context) -> object:
    """Human-in-the-loop approval for sensitive operations.

    Auto-approves read-only tools; prompts the user for writes, bash commands,
    and agent spawns.
    """
    # Always allow read-only tools
    read_only = {"Read", "Glob", "Grep", "WebSearch", "WebFetch"}
    if tool_name in read_only:
        return PermissionResultAllow(updated_input=input_data)

    # Always allow writes to our output directories
    if tool_name in ("Write", "Edit"):
        file_path = input_data.get("file_path", "")
        if "/output/" in file_path:
            return PermissionResultAllow(updated_input=input_data)

    # Always allow MCP arxiv tools
    if tool_name.startswith("mcp__arxiv"):
        return PermissionResultAllow(updated_input=input_data)

    # For AskUserQuestion, intercept and show to user in the terminal.
    # The user's answer is returned as modified tool input so the agent
    # receives it as the tool result.
    if tool_name == "AskUserQuestion":
        # The tool input can have two formats:
        # 1. Simple: {"question": "...", "options": [...]}
        # 2. Multi:  {"questions": [{"question": "...", "header": "...", "options": [...]}]}
        questions = input_data.get("questions", [])
        if not questions:
            # Simple format — wrap in list
            questions = [{
                "question": input_data.get("question", ""),
                "header": "",
                "options": input_data.get("options", []),
            }]

        answers = {}
        for q in questions:
            q_text = q.get("question", "")
            q_header = q.get("header", "")
            q_options = q.get("options", [])

            print(f"\n{'='*60}")
            if q_header:
                print(f"  {_CYAN}{_BOLD}{q_header}{_RESET}")
            print(f"  {_BOLD}{q_text}{_RESET}")
            if q_options:
                print()
                for i, opt in enumerate(q_options):
                    if isinstance(opt, dict):
                        label = opt.get("label", opt.get("value", str(opt)))
                        desc = opt.get("description", "")
                        print(f"    {i+1}. {label}" + (f" — {desc}" if desc else ""))
                    else:
                        print(f"    {i+1}. {opt}")
            print(f"{'='*60}")

            user_answer = input("  Your answer: ").strip()
            answers[q_text] = user_answer

        # Return answers back so the agent receives them
        updated = {**input_data, "answers": answers}
        if len(answers) == 1:
            updated["answer"] = list(answers.values())[0]
        return PermissionResultAllow(updated_input=updated)

    # For Agent tool, always allow (director needs to spawn sub-agents)
    if tool_name == "Agent":
        return PermissionResultAllow(updated_input=input_data)

    # For Bash commands, auto-allow if running inside the output directory;
    # otherwise prompt the user.
    if tool_name == "Bash":
        cmd = input_data.get("command", "")
        if "/output/" in cmd:
            return PermissionResultAllow(updated_input=input_data)
        print(f"\n{'='*60}")
        print(f"  Agent wants to run: {cmd}")
        print(f"{'='*60}")
        response = input("  Allow? (y/n): ").strip().lower()
        if response == "y":
            return PermissionResultAllow(updated_input=input_data)
        return PermissionResultDeny(message="User denied this command.")

    # Default: allow
    return PermissionResultAllow(updated_input=input_data)


# ── Director system prompt ──────────────────────────────────────────────────

DIRECTOR_SYSTEM_PROMPT = """\
You are the **Director** of a research-to-code pipeline. You orchestrate a
team of specialized sub-agents to turn research papers into working code.

## Your Sub-Agents

- **paper-reader**: Reads LOCAL paper files (PDF/MD/TXT) from the input folder
  and writes summaries to output/summaries/.
- **arxiv-reader**: Reads papers via arXiv MCP server using paper IDs and
  writes summaries to output/summaries/.
- **planner**: Creates implementation plans in output/plans/.
- **coder**: Writes and tests code in output/code/.
- **reviewer**: Reviews code and writes reports to output/reviews/.

## Your Workflow

Follow these steps IN ORDER. After each agent completes, pause and get user
approval before proceeding to the next step.

### Step 1: Read Papers
- Check the **mode** field in the prompt:
  - If mode is "local": use the **paper-reader** agent. Pass the file paths.
  - If mode is "arxiv": use the **arxiv-reader** agent. Pass the paper IDs.
- Include the objective in your prompt to the reader agent.
- After it finishes, read output/summaries/_overview.md and present a brief
  summary to the user.
- Use AskUserQuestion to ask: "Which papers/techniques should we proceed with?"

### Step 2: Generate Plan
- Tell the planner agent to create an implementation plan.
- Include the user's objective and any preferences from step 1.
- After it finishes, read output/plans/plan.md and present the key points.
- Use AskUserQuestion to ask: "Does this plan look good? Any changes?"

### Step 3: Write Code
- Tell the coder agent to implement the approved plan.
- After it finishes, list the files created in output/code/.
- Use AskUserQuestion to ask: "Code is ready. Want to proceed to review, or make changes?"

### Step 4: Review Code
- Tell the reviewer agent to review the generated code.
- After it finishes, read output/reviews/review.md and present findings.
- If the review says NEEDS_CHANGES or FAIL, use AskUserQuestion to ask the
  user if they want to iterate (send back to coder with review feedback).

### Step 5: Deliver
- Present the final summary: papers read, plan, code location, review result.
- Use AskUserQuestion to ask if the user wants to iterate on anything.

## Your Skills
You have two skills available:
- **pipeline-status**: Use this to check which pipeline stages are complete
  before deciding what to do next.
- **iterate-feedback**: Use this when the reviewer reports NEEDS_CHANGES or
  FAIL and the user approves a rework cycle. It handles sending review
  feedback back to the coder and re-running the reviewer.

## CRITICAL Rules
- You MUST use the **AskUserQuestion** tool for ALL questions to the user.
  NEVER ask questions in plain text. Plain text questions will crash the
  pipeline because the CLI subprocess exits when there are no tool calls.
  Every user-facing question MUST go through the AskUserQuestion tool.
- NEVER output a text-only response as your final message. If you need to
  present information to the user, ALWAYS follow it with an AskUserQuestion
  tool call in the same turn.
- NEVER skip a stage or proceed without user confirmation.
- If a sub-agent fails, report the error and use AskUserQuestion to ask the
  user how to proceed.
- Keep your own messages concise — the sub-agents do the heavy lifting.
- Use the **pipeline-status** skill when resuming or when unsure about state.
- Use the **iterate-feedback** skill for rework loops instead of manual dispatch.
"""

DIRECTOR_SKILLS = ["pipeline-status", "iterate-feedback"]


# ── Logging ──────────────────────────────────────────────────────────────────

# ANSI colors for terminal output
_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_CYAN = "\033[36m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_MAGENTA = "\033[35m"
_BLUE = "\033[34m"


def _ts() -> str:
    """Short timestamp for log lines."""
    return datetime.now().strftime("%H:%M:%S")


def _log_message(message: object) -> None:
    """Pretty-print every SDK message to the terminal."""

    # ── Assistant message (text, tool calls, thinking) ───────────────────
    if isinstance(message, AssistantMessage):
        for block in message.content:
            if isinstance(block, TextBlock):
                print(f"\n{_CYAN}{_BOLD}[{_ts()}] Director:{_RESET}")
                print(f"  {block.text}")

            elif isinstance(block, ToolUseBlock):
                # Detect sub-agent spawns
                if block.name == "Agent":
                    agent_type = block.input.get("subagent_type", "general-purpose")
                    desc = block.input.get("description", "")
                    print(
                        f"\n{_MAGENTA}{_BOLD}[{_ts()}] Spawning sub-agent: "
                        f"{agent_type}{_RESET}"
                    )
                    if desc:
                        print(f"  {_DIM}Description: {desc}{_RESET}")
                    prompt_preview = block.input.get("prompt", "")[:150]
                    if prompt_preview:
                        print(f"  {_DIM}Prompt: {prompt_preview}...{_RESET}")
                elif block.name == "AskUserQuestion":
                    # Don't log here — the can_use_tool callback
                    # will show the full question and collect the answer.
                    print(
                        f"\n{_YELLOW}{_BOLD}[{_ts()}] Asking user for input...{_RESET}"
                    )
                else:
                    print(
                        f"\n{_YELLOW}[{_ts()}] Tool call: {_BOLD}{block.name}{_RESET}"
                    )
                    # Show relevant input (truncated for non-question tools)
                    for key, val in block.input.items():
                        val_str = str(val)
                        if len(val_str) > 200:
                            val_str = val_str[:200] + "..."
                        print(f"  {_DIM}{key}: {val_str}{_RESET}")

            elif isinstance(block, ThinkingBlock):
                preview = block.thinking[:100].replace("\n", " ")
                print(f"  {_DIM}[{_ts()}] Thinking: {preview}...{_RESET}")

    # ── Tool results ─────────────────────────────────────────────────────
    elif isinstance(message, ToolResultBlock):
        content = str(message.content or "")
        if message.is_error:
            print(f"  {_RED}[{_ts()}] Tool error: {content[:200]}{_RESET}")
        else:
            preview = content[:200].replace("\n", " ")
            print(f"  {_GREEN}[{_ts()}] Tool result: {preview}{_RESET}")

    # ── Task lifecycle (sub-agent progress) ──────────────────────────────
    elif isinstance(message, TaskStartedMessage):
        print(
            f"\n{_BLUE}{_BOLD}[{_ts()}] Task started: "
            f"{message.description}{_RESET}"
        )

    elif isinstance(message, TaskProgressMessage):
        tool_info = ""
        if message.last_tool_name:
            tool_info = f" (last tool: {message.last_tool_name})"
        tokens = message.usage.get("total_tokens", 0) if message.usage else 0
        print(
            f"  {_BLUE}[{_ts()}] Task progress: {message.description}"
            f"{tool_info} [{tokens:,} tokens]{_RESET}"
        )

    elif isinstance(message, TaskNotificationMessage):
        status_color = _GREEN if message.status == "completed" else _RED
        print(
            f"\n{status_color}{_BOLD}[{_ts()}] Task {message.status}: "
            f"{message.summary[:200]}{_RESET}"
        )

    # ── System messages (init, compact, etc.) ────────────────────────────
    elif isinstance(message, SystemMessage):
        if message.subtype == "init":
            mcp_servers = message.data.get("mcp_servers", [])
            if mcp_servers:
                for s in mcp_servers:
                    status = s.get("status", "unknown")
                    name = s.get("name", "?")
                    color = _GREEN if status == "connected" else _RED
                    print(
                        f"  {color}[{_ts()}] MCP '{name}': {status}{_RESET}"
                    )
            print(f"  {_DIM}[{_ts()}] Session initialized{_RESET}")
        else:
            print(
                f"  {_DIM}[{_ts()}] System ({message.subtype}){_RESET}"
            )

    # ── Final result ─────────────────────────────────────────────────────
    elif isinstance(message, ResultMessage):
        cost = message.total_cost_usd or 0
        turns = message.num_turns
        duration_s = message.duration_ms / 1000
        print(f"\n{'='*60}")
        print(f"  {_BOLD}Pipeline complete{_RESET}")
        print(f"  Duration: {duration_s:.1f}s | Turns: {turns} | Cost: ${cost:.4f}")
        if message.is_error:
            print(f"  {_RED}Stopped with error: {message.result}{_RESET}")
        elif message.result:
            print(f"\n{message.result}")
        print(f"{'='*60}")

    # ── User messages (echoed input) ─────────────────────────────────────
    elif isinstance(message, UserMessage):
        pass  # Don't echo back user's own input

    # ── Catch-all for unknown message types ──────────────────────────────
    else:
        print(f"  {_DIM}[{_ts()}] {type(message).__name__}: {str(message)[:150]}{_RESET}")


def build_agent_definitions() -> dict[str, AgentDefinition]:
    """Collect all sub-agent definitions into a dict."""
    agents = {}
    for factory in (make_local_reader, make_arxiv_reader, make_planner, make_coder, make_reviewer):
        name, definition = factory()
        agents[name] = definition
    return agents


async def run_director(
    objective: str,
    papers: list[str],
    mode: str = "arxiv",
) -> None:
    """Launch the Director agent.

    Args:
        objective: What the user wants to build.
        papers: List of file paths (local mode) or arXiv IDs (arxiv mode).
        mode: "local" for files in the input folder, "arxiv" for MCP.
    """
    paths = create_project_output(objective)
    project_dir = paths["project_dir"]

    week_dir = get_latest_week_dir()

    if mode == "local":
        papers_section = (
            f"## Mode\nlocal (papers are files on disk)\n\n"
            f"## Paper Files\n"
            + "\n".join(f"- {p}" for p in papers)
        )
        begin_msg = (
            "Begin with Step 1: dispatch the **paper-reader** agent to read "
            "and summarize the local paper files listed above."
        )
    else:
        papers_section = (
            f"## Mode\narxiv (use arxiv MCP server)\n\n"
            f"## Paper IDs\n{json.dumps(papers)}"
        )
        begin_msg = (
            "Begin with Step 1: dispatch the **arxiv-reader** agent to read "
            "and summarize the papers listed above."
        )

    prompt = (
        f"## Objective\n{objective}\n\n"
        f"{papers_section}\n\n"
        f"## Week Folder\n{week_dir}\n\n"
        f"## Project Output Directory\n{project_dir}\n\n"
        f"## Output Directories\n"
        f"- Summaries: {paths['summaries']}\n"
        f"- Plans: {paths['plans']}\n"
        f"- Code: {paths['code']}\n"
        f"- Reviews: {paths['reviews']}\n\n"
        f"{begin_msg}"
    )

    def on_stderr(line: str) -> None:
        print(f"[STDERR] {line}", flush=True)

    # Only attach arXiv MCP server when using arxiv mode
    mcp_servers = {}
    if mode == "arxiv":
        mcp_servers["arxiv-mcp-server"] = ARXIV_MCP_CONFIG

    mode_label = "LOCAL files" if mode == "local" else "arXiv MCP"
    print(f"\n{'='*60}")
    print("  Research Swarm — Director Starting")
    print(f"  Objective: {objective}")
    print(f"  Mode: {mode_label}")
    print(f"  Papers: {len(papers)} paper(s)")
    for p in papers:
        print(f"    - {p}")
    print(f"  Week: {week_dir.name}")
    print(f"  Output: {project_dir}")
    print(f"{'='*60}\n")

    # The SDK requires an AsyncIterable prompt when can_use_tool is set.
    # The stream must stay open so the agent can ask questions and receive
    # follow-up user input. We use an asyncio.Queue to feed messages.
    user_queue: asyncio.Queue[dict | None] = asyncio.Queue()

    # Seed with the initial prompt
    await user_queue.put({
        "type": "user",
        "session_id": "",
        "message": {"role": "user", "content": prompt},
        "parent_tool_use_id": None,
    })

    # Wrap can_use_tool so AskUserQuestion answers feed back into the queue.
    # The tool result alone may not keep the CLI alive; we also push a user
    # message so prompt_stream() unblocks and the agent gets a new turn.
    async def _can_use_tool_with_queue(tool_name: str, input_data: dict, context) -> object:
        result = await can_use_tool(tool_name, input_data, context)
        if tool_name == "AskUserQuestion" and isinstance(result, PermissionResultAllow):
            answer = result.updated_input.get("answer", "")
            if not answer:
                answers = result.updated_input.get("answers", {})
                answer = " | ".join(f"{k}: {v}" for k, v in answers.items()) if answers else "continue"
            await user_queue.put({
                "type": "user",
                "session_id": "",
                "message": {"role": "user", "content": answer},
                "parent_tool_use_id": None,
            })
        return result

    async def prompt_stream():
        while True:
            msg = await user_queue.get()
            if msg is None:
                break
            yield msg

    options = ClaudeAgentOptions(
        model=DIRECTOR_MODEL,
        system_prompt=DIRECTOR_SYSTEM_PROMPT,
        allowed_tools=[
            "Read",
            "Write",
            "Glob",
            "Grep",
            "Agent",
            "AskUserQuestion",
            "Skill",
        ],
        mcp_servers=mcp_servers,
        agents=build_agent_definitions(),
        can_use_tool=_can_use_tool_with_queue,
        max_turns=50,
        cwd=str(Path(__file__).resolve().parent.parent),
        stderr=on_stderr,
        setting_sources=["project"],
    )

    async for message in query(prompt=prompt_stream(), options=options):
        _log_message(message)
