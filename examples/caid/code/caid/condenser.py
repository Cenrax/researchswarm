"""LLMSummarizingCondenser for manager context compression.

Prevents context window overflow during long CAID runs by
periodically compressing the manager's conversation history
while preserving critical state (dependency graph, completed
tasks, unresolved errors).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from caid.graph import DependencyGraph
from caid.llm import LLMClient

logger = logging.getLogger(__name__)

CONDENSER_SYSTEM_PROMPT = """You are a context condenser for a multi-agent software engineering system.
Given a conversation history, produce a concise summary that preserves:
1. The current dependency graph state (which tasks are completed, in-progress, and remaining)
2. Any unresolved errors or merge conflicts
3. Key decisions made and their rationale
4. The current implementation round number

Do NOT preserve: verbose code listings, redundant status updates, or resolved issues.
Output a structured summary as plain text."""


class LLMSummarizingCondenser:
    """Compresses manager conversation history using LLM summarization.

    Maintains critical state information while reducing token count
    to prevent context window overflow during long CAID runs.
    """

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm = llm_client

    async def condense(
        self,
        messages: list[dict[str, str]],
        graph: DependencyGraph,
        preserve_keys: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, str]]:
        """Compress message history while preserving critical state.

        Args:
            messages: Current conversation history.
            graph: Current dependency graph for state extraction.
            preserve_keys: Additional key-value pairs to preserve.

        Returns:
            Condensed message list (system prompt + state summary).
        """
        if len(messages) <= 4:
            return messages  # Too short to compress

        state_header = self._build_state_header(graph, preserve_keys)
        formatted_history = self._format_messages(messages)

        condense_messages = [
            {"role": "system", "content": CONDENSER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Current state:\n{state_header}\n\n"
                    f"History to condense:\n{formatted_history}"
                ),
            },
        ]

        try:
            summary = await self.llm.call(condense_messages)
        except Exception as e:
            logger.warning("Condenser LLM call failed: %s. Keeping original.", e)
            return messages

        # Return condensed: original system prompt + condensed state
        condensed = [
            messages[0],  # Keep original system prompt
            {
                "role": "assistant",
                "content": (
                    f"[Condensed History]\n{state_header}\n\n{summary}"
                ),
            },
        ]

        logger.info(
            "Condensed %d messages to %d (saved ~%d chars)",
            len(messages),
            len(condensed),
            sum(len(m.get("content", "")) for m in messages)
            - sum(len(m.get("content", "")) for m in condensed),
        )
        return condensed

    def condense_sync(
        self,
        messages: list[dict[str, str]],
        graph: DependencyGraph,
        preserve_keys: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, str]]:
        """Synchronous fallback condenser without LLM.

        Truncates history to system prompt + state summary, discarding
        old messages. Used when LLM calls are unavailable.
        """
        if len(messages) <= 4:
            return messages

        state_header = self._build_state_header(graph, preserve_keys)
        return [
            messages[0],
            {
                "role": "assistant",
                "content": f"[Condensed State]\n{state_header}",
            },
        ]

    def _build_state_header(
        self,
        graph: DependencyGraph,
        extras: Optional[dict[str, Any]] = None,
    ) -> str:
        """Build a structured state summary from the current graph."""
        completed = graph.completed
        assignable = graph.get_assignable_tasks()
        in_progress = graph.in_progress
        remaining = graph.all_tasks - completed - in_progress

        lines = [
            f"Completed tasks ({len(completed)}): {sorted(completed)}",
            f"In-progress tasks ({len(in_progress)}): {sorted(in_progress)}",
            f"Assignable tasks ({len(assignable)}): {assignable}",
            f"Remaining blocked tasks ({len(remaining)}): {sorted(remaining)}",
            f"Total progress: {len(completed)}/{len(graph.all_tasks)}",
        ]
        if extras:
            for k, v in extras.items():
                lines.append(f"{k}: {v}")
        return "\n".join(lines)

    @staticmethod
    def _format_messages(messages: list[dict[str, str]]) -> str:
        """Format messages for LLM consumption, truncating long content."""
        parts: list[str] = []
        for m in messages:
            content = m.get("content", "")
            # Truncate individual messages to avoid overwhelming the condenser
            if len(content) > 500:
                content = content[:500] + "... [truncated]"
            parts.append(f"[{m.get('role', 'unknown')}]: {content}")
        return "\n---\n".join(parts)


if __name__ == "__main__":
    from caid.graph import TaskNode

    # Build a test graph
    g = DependencyGraph()
    for tid in ["A", "B", "C", "D"]:
        g.add_task(TaskNode(task_id=tid))
    g.add_dependency("A", "B")
    g.add_dependency("A", "C")
    g.add_dependency("B", "D")
    g.add_dependency("C", "D")
    g.mark_completed("A")
    g.mark_in_progress("B")

    # Test sync condenser
    from caid.config import LLMConfig

    config = LLMConfig(model="test")
    client = LLMClient(config)
    condenser = LLMSummarizingCondenser(client)

    messages = [
        {"role": "system", "content": "You are the CAID manager."},
        {"role": "user", "content": "Explore the repository."},
        {"role": "assistant", "content": "Found 4 tasks: A, B, C, D."},
        {"role": "user", "content": "Build the dependency graph."},
        {"role": "assistant", "content": "Graph built with edges A->B, A->C, B->D, C->D."},
        {"role": "user", "content": "Delegate tasks."},
    ]

    condensed = condenser.condense_sync(messages, g)
    print(f"Original messages: {len(messages)}")
    print(f"Condensed messages: {len(condensed)}")
    assert len(condensed) < len(messages)
    print(f"Condensed content:\n{condensed[1]['content']}")

    # Verify state is preserved
    content = condensed[1]["content"]
    assert "A" in content  # completed
    assert "B" in content  # in progress
    assert "D" in content  # remaining
    print("State preservation verified")

    # Too-short messages should pass through
    short = messages[:3]
    result = condenser.condense_sync(short, g)
    assert result == short
    print("Short message passthrough OK")

    print("Condenser module OK")
