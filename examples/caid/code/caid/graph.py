"""Dependency graph engine implementing CAID Equations 1-5.

Provides the core data structure for task tracking, readiness checking,
and assignable task selection based on the directed acyclic graph
described in CAID Section 2.1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import networkx as nx


@dataclass
class TaskNode:
    """A single work unit in the dependency graph.

    Attributes:
        task_id: Unique identifier for this task.
        file_path: Source file associated with this task (Commit0).
        functions: List of function names to implement.
        complexity: Estimated complexity level.
        metadata: Additional task-specific data.
    """

    task_id: str
    file_path: str = ""
    functions: list[str] = field(default_factory=list)
    complexity: str = "medium"
    metadata: dict[str, Any] = field(default_factory=dict)


class DependencyGraph:
    """Directed acyclic graph for CAID task dependency tracking.

    Implements Equations 1-5 from the CAID paper:
      - Eq. 1: G = (V, E) graph structure
      - Eq. 2: C_t completed set tracking
      - Eq. 3: Ready_t(v_j) readiness predicate
      - Eq. 4: AssignableAt_t task set selection
      - Eq. 5: Termination condition
    """

    def __init__(self) -> None:
        self._graph: nx.DiGraph = nx.DiGraph()
        self._completed: set[str] = set()
        self._in_progress: set[str] = set()

    def add_task(self, node: TaskNode) -> None:
        """Add a task node to the graph."""
        self._graph.add_node(node.task_id, data=node)

    def add_dependency(self, prerequisite_id: str, dependent_id: str) -> None:
        """Add a dependency edge: dependent depends on prerequisite.

        Raises:
            ValueError: If the edge would create a cycle.
            KeyError: If either node does not exist.
        """
        if prerequisite_id not in self._graph:
            raise KeyError(f"Prerequisite node not found: {prerequisite_id}")
        if dependent_id not in self._graph:
            raise KeyError(f"Dependent node not found: {dependent_id}")

        self._graph.add_edge(prerequisite_id, dependent_id)
        if not nx.is_directed_acyclic_graph(self._graph):
            self._graph.remove_edge(prerequisite_id, dependent_id)
            raise ValueError(
                f"Adding edge {prerequisite_id} -> {dependent_id} "
                f"would create a cycle"
            )

    def is_ready(self, task_id: str) -> bool:
        """Check if a task is ready for delegation (Eq. 3).

        Ready_t(v_j) iff for all (v_i, v_j) in E, v_i in C_t.
        A task with no predecessors is always ready.
        """
        return all(
            pred in self._completed
            for pred in self._graph.predecessors(task_id)
        )

    def get_assignable_tasks(self) -> list[str]:
        """Get tasks eligible for delegation (Eq. 4).

        AssignableAt_t = {v in V : Ready_t(v) and v not in C_t and v not in running_t}

        Results are sorted by priority: tasks with more downstream
        dependents are assigned first (upstream-first strategy).
        """
        candidates = set(self._graph.nodes) - self._completed - self._in_progress
        assignable = [v for v in candidates if self.is_ready(v)]
        # Prioritize tasks that unblock the most downstream work
        assignable.sort(
            key=lambda v: -len(nx.descendants(self._graph, v))
        )
        return assignable

    def mark_completed(self, task_id: str) -> None:
        """Mark a task as completed and integrated (add to C_t)."""
        self._in_progress.discard(task_id)
        self._completed.add(task_id)

    def mark_in_progress(self, task_id: str) -> None:
        """Mark a task as currently being worked on by an engineer."""
        self._in_progress.add(task_id)

    def mark_failed(self, task_id: str) -> None:
        """Return a task to the assignable pool after failure."""
        self._in_progress.discard(task_id)

    def is_done(self) -> bool:
        """Check termination condition (Eq. 5 partial): C_t == V."""
        return self._completed == set(self._graph.nodes)

    def get_node_data(self, task_id: str) -> TaskNode:
        """Retrieve the TaskNode data for a given task."""
        return self._graph.nodes[task_id]["data"]

    @property
    def completed(self) -> set[str]:
        """Copy of the current completed set C_t."""
        return self._completed.copy()

    @property
    def in_progress(self) -> set[str]:
        """Copy of the current in-progress set."""
        return self._in_progress.copy()

    @property
    def all_tasks(self) -> set[str]:
        """Set of all task IDs in the graph."""
        return set(self._graph.nodes)

    def topological_order(self) -> list[str]:
        """Return tasks in topological order (valid execution sequence)."""
        return list(nx.topological_sort(self._graph))

    @property
    def node_count(self) -> int:
        """Number of nodes in the graph."""
        return self._graph.number_of_nodes()

    @property
    def edge_count(self) -> int:
        """Number of edges in the graph."""
        return self._graph.number_of_edges()

    def predecessors(self, task_id: str) -> list[str]:
        """Get direct predecessors (dependencies) of a task."""
        return list(self._graph.predecessors(task_id))

    def successors(self, task_id: str) -> list[str]:
        """Get direct successors (dependents) of a task."""
        return list(self._graph.successors(task_id))


if __name__ == "__main__":
    # Build a sample dependency graph
    g = DependencyGraph()
    g.add_task(TaskNode(task_id="A", file_path="a.py", functions=["func_a"]))
    g.add_task(TaskNode(task_id="B", file_path="b.py", functions=["func_b"]))
    g.add_task(TaskNode(task_id="C", file_path="c.py", functions=["func_c"]))
    g.add_task(TaskNode(task_id="D", file_path="d.py", functions=["func_d"]))

    # D depends on B and C; B and C depend on A
    g.add_dependency("A", "B")
    g.add_dependency("A", "C")
    g.add_dependency("B", "D")
    g.add_dependency("C", "D")

    print(f"Nodes: {g.node_count}, Edges: {g.edge_count}")
    print(f"Topological order: {g.topological_order()}")

    # Initially only A is assignable
    assignable = g.get_assignable_tasks()
    print(f"Initially assignable: {assignable}")
    assert assignable == ["A"], f"Expected ['A'], got {assignable}"

    # Complete A -> B and C become assignable
    g.mark_completed("A")
    assignable = g.get_assignable_tasks()
    print(f"After A complete: {assignable}")
    assert set(assignable) == {"B", "C"}

    # Mark B in progress, C still assignable
    g.mark_in_progress("B")
    assignable = g.get_assignable_tasks()
    print(f"After B in progress: {assignable}")
    assert assignable == ["C"]

    # Complete B and C -> D assignable
    g.mark_completed("B")
    g.mark_completed("C")
    assignable = g.get_assignable_tasks()
    print(f"After B,C complete: {assignable}")
    assert assignable == ["D"]

    # Complete D -> done
    g.mark_completed("D")
    assert g.is_done()
    print(f"All done: {g.is_done()}")

    # Test cycle detection
    g2 = DependencyGraph()
    g2.add_task(TaskNode(task_id="X"))
    g2.add_task(TaskNode(task_id="Y"))
    g2.add_dependency("X", "Y")
    try:
        g2.add_dependency("Y", "X")
        print("ERROR: Should have raised ValueError")
    except ValueError as e:
        print(f"Cycle detection OK: {e}")

    # Test failed task re-entry
    g3 = DependencyGraph()
    g3.add_task(TaskNode(task_id="T1"))
    g3.mark_in_progress("T1")
    assert g3.get_assignable_tasks() == []
    g3.mark_failed("T1")
    assert g3.get_assignable_tasks() == ["T1"]
    print("Failed task re-entry OK")

    print("Graph module OK")
