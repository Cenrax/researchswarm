"""Tests for caid.condenser module."""

from __future__ import annotations

from caid.condenser import LLMSummarizingCondenser
from caid.config import LLMConfig
from caid.graph import DependencyGraph, TaskNode
from caid.llm import LLMClient


def _make_condenser() -> LLMSummarizingCondenser:
    client = LLMClient(LLMConfig(model="test"))
    return LLMSummarizingCondenser(client)


def _make_graph() -> DependencyGraph:
    g = DependencyGraph()
    for tid in ["A", "B", "C", "D"]:
        g.add_task(TaskNode(task_id=tid))
    g.add_dependency("A", "B")
    g.add_dependency("A", "C")
    g.add_dependency("B", "D")
    g.add_dependency("C", "D")
    g.mark_completed("A")
    g.mark_in_progress("B")
    return g


class TestSyncCondenser:
    def test_short_messages_pass_through(self) -> None:
        condenser = _make_condenser()
        graph = _make_graph()
        messages = [
            {"role": "system", "content": "You are the manager."},
            {"role": "user", "content": "Start."},
        ]
        result = condenser.condense_sync(messages, graph)
        assert result == messages

    def test_long_messages_get_condensed(self) -> None:
        condenser = _make_condenser()
        graph = _make_graph()
        messages = [
            {"role": "system", "content": "System prompt."},
            {"role": "user", "content": "Message 1"},
            {"role": "assistant", "content": "Response 1"},
            {"role": "user", "content": "Message 2"},
            {"role": "assistant", "content": "Response 2"},
            {"role": "user", "content": "Message 3"},
        ]
        result = condenser.condense_sync(messages, graph)
        assert len(result) < len(messages)
        assert len(result) == 2

    def test_system_prompt_preserved(self) -> None:
        condenser = _make_condenser()
        graph = _make_graph()
        messages = [
            {"role": "system", "content": "Important system prompt."},
            {"role": "user", "content": "M1"},
            {"role": "assistant", "content": "R1"},
            {"role": "user", "content": "M2"},
            {"role": "assistant", "content": "R2"},
        ]
        result = condenser.condense_sync(messages, graph)
        assert result[0]["content"] == "Important system prompt."

    def test_state_preserved_in_condensed(self) -> None:
        condenser = _make_condenser()
        graph = _make_graph()
        messages = [
            {"role": "system", "content": "Sys"},
            {"role": "user", "content": "M1"},
            {"role": "assistant", "content": "R1"},
            {"role": "user", "content": "M2"},
            {"role": "assistant", "content": "R2"},
        ]
        result = condenser.condense_sync(messages, graph)
        content = result[1]["content"]
        # Verify completed task A is mentioned
        assert "A" in content
        # Verify in-progress task B is mentioned
        assert "B" in content
        # Verify progress count
        assert "1/4" in content

    def test_with_preserve_keys(self) -> None:
        condenser = _make_condenser()
        graph = _make_graph()
        messages = [
            {"role": "system", "content": "Sys"},
            {"role": "user", "content": "M1"},
            {"role": "assistant", "content": "R1"},
            {"role": "user", "content": "M2"},
            {"role": "assistant", "content": "R2"},
        ]
        result = condenser.condense_sync(
            messages, graph, preserve_keys={"round": 2}
        )
        content = result[1]["content"]
        assert "round" in content


class TestBuildStateHeader:
    def test_includes_all_categories(self) -> None:
        condenser = _make_condenser()
        graph = _make_graph()
        header = condenser._build_state_header(graph)
        assert "Completed" in header
        assert "In-progress" in header
        assert "Assignable" in header
        assert "Remaining" in header
        assert "Total progress" in header
