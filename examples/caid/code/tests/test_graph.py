"""Tests for caid.graph module (Equations 1-5)."""

from __future__ import annotations

import pytest

from caid.graph import DependencyGraph, TaskNode


class TestTaskNode:
    def test_defaults(self) -> None:
        node = TaskNode(task_id="test")
        assert node.file_path == ""
        assert node.functions == []
        assert node.complexity == "medium"
        assert node.metadata == {}

    def test_custom_values(self) -> None:
        node = TaskNode(
            task_id="t1",
            file_path="src/a.py",
            functions=["func_a", "func_b"],
            complexity="complex",
            metadata={"key": "value"},
        )
        assert node.task_id == "t1"
        assert len(node.functions) == 2


class TestDependencyGraphConstruction:
    def test_add_task(self) -> None:
        g = DependencyGraph()
        g.add_task(TaskNode(task_id="A"))
        assert "A" in g.all_tasks
        assert g.node_count == 1

    def test_add_dependency(self) -> None:
        g = DependencyGraph()
        g.add_task(TaskNode(task_id="A"))
        g.add_task(TaskNode(task_id="B"))
        g.add_dependency("A", "B")
        assert g.edge_count == 1

    def test_cycle_detection(self) -> None:
        g = DependencyGraph()
        g.add_task(TaskNode(task_id="A"))
        g.add_task(TaskNode(task_id="B"))
        g.add_dependency("A", "B")
        with pytest.raises(ValueError, match="cycle"):
            g.add_dependency("B", "A")
        # Edge should not have been added
        assert g.edge_count == 1

    def test_missing_node_raises_key_error(self) -> None:
        g = DependencyGraph()
        g.add_task(TaskNode(task_id="A"))
        with pytest.raises(KeyError):
            g.add_dependency("A", "nonexistent")

    def test_self_cycle_detection(self) -> None:
        g = DependencyGraph()
        g.add_task(TaskNode(task_id="A"))
        with pytest.raises(ValueError, match="cycle"):
            g.add_dependency("A", "A")


class TestReadiness:
    """Tests for Eq. 3: Ready_t(v_j) iff all predecessors in C_t."""

    def test_no_dependencies_is_ready(self) -> None:
        g = DependencyGraph()
        g.add_task(TaskNode(task_id="A"))
        assert g.is_ready("A")

    def test_unmet_dependency_not_ready(self, sample_graph: DependencyGraph) -> None:
        assert not sample_graph.is_ready("B")  # A not completed
        assert not sample_graph.is_ready("D")  # B,C not completed

    def test_met_dependency_is_ready(self, sample_graph: DependencyGraph) -> None:
        sample_graph.mark_completed("A")
        assert sample_graph.is_ready("B")
        assert sample_graph.is_ready("C")

    def test_partial_dependency_not_ready(self, sample_graph: DependencyGraph) -> None:
        sample_graph.mark_completed("A")
        sample_graph.mark_completed("B")
        # D depends on both B and C; C not done yet
        assert not sample_graph.is_ready("D")

    def test_all_dependencies_met(self, sample_graph: DependencyGraph) -> None:
        sample_graph.mark_completed("A")
        sample_graph.mark_completed("B")
        sample_graph.mark_completed("C")
        assert sample_graph.is_ready("D")


class TestAssignableTasks:
    """Tests for Eq. 4: AssignableAt_t."""

    def test_initial_assignable(self, sample_graph: DependencyGraph) -> None:
        assignable = sample_graph.get_assignable_tasks()
        assert assignable == ["A"]

    def test_after_completing_root(self, sample_graph: DependencyGraph) -> None:
        sample_graph.mark_completed("A")
        assignable = sample_graph.get_assignable_tasks()
        assert set(assignable) == {"B", "C"}

    def test_in_progress_excluded(self, sample_graph: DependencyGraph) -> None:
        sample_graph.mark_completed("A")
        sample_graph.mark_in_progress("B")
        assignable = sample_graph.get_assignable_tasks()
        assert assignable == ["C"]

    def test_completed_excluded(self, sample_graph: DependencyGraph) -> None:
        sample_graph.mark_completed("A")
        sample_graph.mark_completed("B")
        sample_graph.mark_completed("C")
        assignable = sample_graph.get_assignable_tasks()
        assert assignable == ["D"]

    def test_empty_when_all_done(self, sample_graph: DependencyGraph) -> None:
        for t in ["A", "B", "C", "D"]:
            sample_graph.mark_completed(t)
        assert sample_graph.get_assignable_tasks() == []

    def test_failed_task_returns_to_pool(self) -> None:
        g = DependencyGraph()
        g.add_task(TaskNode(task_id="T"))
        g.mark_in_progress("T")
        assert g.get_assignable_tasks() == []
        g.mark_failed("T")
        assert g.get_assignable_tasks() == ["T"]

    def test_upstream_priority(self) -> None:
        """Tasks unblocking more downstream work should be first."""
        g = DependencyGraph()
        g.add_task(TaskNode(task_id="root"))
        g.add_task(TaskNode(task_id="leaf1"))
        g.add_task(TaskNode(task_id="leaf2"))
        g.add_task(TaskNode(task_id="branch"))
        g.add_dependency("root", "branch")
        g.add_dependency("branch", "leaf1")
        g.add_dependency("branch", "leaf2")
        # root should be first (unblocks 3 downstream)
        # No other node is assignable initially
        assert g.get_assignable_tasks() == ["root"]


class TestTermination:
    """Tests for Eq. 5: Done iff C_t == V."""

    def test_not_done_initially(self, sample_graph: DependencyGraph) -> None:
        assert not sample_graph.is_done()

    def test_done_when_all_completed(self, sample_graph: DependencyGraph) -> None:
        for t in ["A", "B", "C", "D"]:
            sample_graph.mark_completed(t)
        assert sample_graph.is_done()

    def test_not_done_with_remaining(self, sample_graph: DependencyGraph) -> None:
        sample_graph.mark_completed("A")
        sample_graph.mark_completed("B")
        assert not sample_graph.is_done()


class TestTopologicalOrder:
    def test_linear_order(self, linear_graph: DependencyGraph) -> None:
        order = linear_graph.topological_order()
        assert order == ["A", "B", "C"]

    def test_diamond_order(self, sample_graph: DependencyGraph) -> None:
        order = sample_graph.topological_order()
        assert order.index("A") < order.index("B")
        assert order.index("A") < order.index("C")
        assert order.index("B") < order.index("D")
        assert order.index("C") < order.index("D")


class TestNodeData:
    def test_get_node_data(self) -> None:
        g = DependencyGraph()
        node = TaskNode(task_id="T", file_path="t.py", functions=["f1"])
        g.add_task(node)
        retrieved = g.get_node_data("T")
        assert retrieved.task_id == "T"
        assert retrieved.file_path == "t.py"
        assert retrieved.functions == ["f1"]

    def test_predecessors_and_successors(self, sample_graph: DependencyGraph) -> None:
        assert sample_graph.predecessors("D") == ["B", "C"] or set(
            sample_graph.predecessors("D")
        ) == {"B", "C"}
        assert "B" in sample_graph.successors("A")
        assert "C" in sample_graph.successors("A")
