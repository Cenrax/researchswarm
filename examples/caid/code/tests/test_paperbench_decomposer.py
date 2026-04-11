"""Tests for caid.paperbench.decomposer module."""

from __future__ import annotations

from caid.graph import DependencyGraph
from caid.paperbench.decomposer import _delegation_to_graph
from caid.schemas import (
    Complexity,
    PaperBenchDelegation,
    PaperBenchDelegationPlan,
    PaperBenchFirstRound,
    PaperBenchRemainingTask,
    PaperBenchTask,
    TaskCategory,
)


def _make_delegation() -> PaperBenchDelegation:
    return PaperBenchDelegation(
        delegation_plan=PaperBenchDelegationPlan(
            first_round=PaperBenchFirstRound(
                num_agents=2,
                reasoning="Independent tasks first",
                tasks=[
                    PaperBenchTask(
                        engineer_id="eng-0",
                        task_id="data-loader",
                        requirements="Build dataset loading",
                        task_category=TaskCategory.CODE_DEVELOPMENT,
                        estimated_complexity=Complexity.MEDIUM,
                        instruction="Implement data loader.",
                    ),
                    PaperBenchTask(
                        engineer_id="eng-1",
                        task_id="model-arch",
                        requirements="Build model",
                        task_category=TaskCategory.CODE_DEVELOPMENT,
                        estimated_complexity=Complexity.COMPLEX,
                        instruction="Build the model.",
                    ),
                ],
            ),
            remaining_tasks=[
                PaperBenchRemainingTask(
                    task_id="training",
                    requirements="Training loop",
                    task_category=TaskCategory.CODE_DEVELOPMENT,
                    estimated_complexity=Complexity.COMPLEX,
                    depends_on=["data-loader", "model-arch"],
                ),
                PaperBenchRemainingTask(
                    task_id="evaluation",
                    requirements="Run experiments",
                    task_category=TaskCategory.EXPERIMENT_RUNNING,
                    estimated_complexity=Complexity.MEDIUM,
                    depends_on=["training"],
                ),
            ],
        )
    )


class TestDelegationToGraph:
    def test_correct_node_count(self) -> None:
        graph = _delegation_to_graph(_make_delegation())
        assert graph.node_count == 4

    def test_correct_edge_count(self) -> None:
        graph = _delegation_to_graph(_make_delegation())
        assert graph.edge_count == 3  # data->train, model->train, train->eval

    def test_first_round_tasks_are_assignable(self) -> None:
        graph = _delegation_to_graph(_make_delegation())
        assignable = graph.get_assignable_tasks()
        assert "data-loader" in assignable
        assert "model-arch" in assignable

    def test_dependent_tasks_not_assignable(self) -> None:
        graph = _delegation_to_graph(_make_delegation())
        assignable = graph.get_assignable_tasks()
        assert "training" not in assignable
        assert "evaluation" not in assignable

    def test_dependency_chain(self) -> None:
        graph = _delegation_to_graph(_make_delegation())
        # Complete first round
        graph.mark_completed("data-loader")
        graph.mark_completed("model-arch")
        assert "training" in graph.get_assignable_tasks()
        assert "evaluation" not in graph.get_assignable_tasks()

        # Complete training
        graph.mark_completed("training")
        assert "evaluation" in graph.get_assignable_tasks()

    def test_is_acyclic(self) -> None:
        graph = _delegation_to_graph(_make_delegation())
        # Should be able to produce topological order
        order = graph.topological_order()
        assert len(order) == 4

    def test_topological_order_respects_deps(self) -> None:
        graph = _delegation_to_graph(_make_delegation())
        order = graph.topological_order()
        assert order.index("data-loader") < order.index("training")
        assert order.index("model-arch") < order.index("training")
        assert order.index("training") < order.index("evaluation")

    def test_metadata_preserved(self) -> None:
        graph = _delegation_to_graph(_make_delegation())
        node = graph.get_node_data("data-loader")
        assert node.metadata["requirements"] == "Build dataset loading"
        assert node.metadata["task_category"] == "Code Development"

    def test_handles_missing_dependency(self) -> None:
        """Dependencies referencing non-existent tasks should be skipped."""
        delegation = PaperBenchDelegation(
            delegation_plan=PaperBenchDelegationPlan(
                first_round=PaperBenchFirstRound(
                    num_agents=1,
                    reasoning="test",
                    tasks=[
                        PaperBenchTask(
                            engineer_id="eng-0",
                            task_id="task-a",
                            requirements="Build A",
                            task_category=TaskCategory.CODE_DEVELOPMENT,
                            estimated_complexity=Complexity.SIMPLE,
                            instruction="Build A.",
                        )
                    ],
                ),
                remaining_tasks=[
                    PaperBenchRemainingTask(
                        task_id="task-b",
                        requirements="Build B",
                        task_category=TaskCategory.CODE_DEVELOPMENT,
                        estimated_complexity=Complexity.SIMPLE,
                        depends_on=["nonexistent-task"],  # Should be skipped
                    )
                ],
            )
        )
        graph = _delegation_to_graph(delegation)
        assert graph.node_count == 2
        assert graph.edge_count == 0  # Edge skipped
