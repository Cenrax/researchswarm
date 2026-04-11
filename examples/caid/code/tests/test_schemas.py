"""Tests for caid.schemas module."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from caid.schemas import (
    Commit0Assignment,
    Commit0AssignTask,
    Commit0Delegation,
    Commit0DelegationPlan,
    Commit0FirstRound,
    Commit0Reassignment,
    Commit0RemainingTask,
    Commit0Task,
    Complexity,
    EngineerResult,
    PaperBenchDelegation,
    PaperBenchDelegationPlan,
    PaperBenchFirstRound,
    PaperBenchRemainingTask,
    PaperBenchTask,
    TaskCategory,
)


class TestComplexity:
    def test_values(self) -> None:
        assert Complexity.SIMPLE == "simple"
        assert Complexity.MEDIUM == "medium"
        assert Complexity.COMPLEX == "complex"


class TestCommit0Task:
    def test_round_trip(self) -> None:
        task = Commit0Task(
            engineer_id="eng-0",
            task_id="task-1",
            file_path="src/utils.py",
            functions_to_implement=["func_a", "func_b"],
            complexity=Complexity.MEDIUM,
            instruction="Implement both functions.",
        )
        json_str = task.model_dump_json()
        restored = Commit0Task.model_validate_json(json_str)
        assert restored == task

    def test_invalid_complexity(self) -> None:
        with pytest.raises(ValidationError):
            Commit0Task(
                engineer_id="eng-0",
                task_id="task-1",
                file_path="src/utils.py",
                functions_to_implement=["func"],
                complexity="invalid",  # type: ignore
                instruction="test",
            )

    def test_missing_required_field(self) -> None:
        with pytest.raises(ValidationError):
            Commit0Task(
                engineer_id="eng-0",
                task_id="task-1",
                # missing file_path
                functions_to_implement=["func"],
                complexity="medium",
                instruction="test",
            )  # type: ignore


class TestCommit0Delegation:
    def test_full_delegation_round_trip(self) -> None:
        delegation = Commit0Delegation(
            delegation_plan=Commit0DelegationPlan(
                first_round=Commit0FirstRound(
                    num_agents=2,
                    reasoning="Start with independent files",
                    tasks=[
                        Commit0Task(
                            engineer_id="eng-0",
                            task_id="t1",
                            file_path="a.py",
                            functions_to_implement=["func_a"],
                            complexity=Complexity.SIMPLE,
                            instruction="Implement func_a.",
                        )
                    ],
                ),
                remaining_tasks=[
                    Commit0RemainingTask(
                        task_id="t2",
                        file_path="b.py",
                        functions_to_implement=["func_b"],
                        complexity=Complexity.COMPLEX,
                        depends_on=["t1"],
                    )
                ],
            )
        )
        json_str = delegation.model_dump_json()
        restored = Commit0Delegation.model_validate_json(json_str)
        assert restored == delegation
        assert len(restored.delegation_plan.remaining_tasks[0].depends_on) == 1


class TestCommit0Reassignment:
    def test_round_trip(self) -> None:
        reassignment = Commit0Reassignment(
            assign_task=Commit0AssignTask(
                reasoning="Task failed, reassigning",
                assignments=[
                    Commit0Assignment(
                        engineer_id="eng-1",
                        task_id="fix-t1",
                        file_path="a.py",
                        functions_to_implement=["func_a"],
                        instruction="Fix the failing implementation.",
                        complexity=Complexity.MEDIUM,
                    )
                ],
            )
        )
        json_str = reassignment.model_dump_json()
        restored = Commit0Reassignment.model_validate_json(json_str)
        assert restored == reassignment


class TestPaperBenchTask:
    def test_round_trip(self) -> None:
        task = PaperBenchTask(
            engineer_id="eng-0",
            task_id="pb-1",
            task_node_id="node-1",
            requirements="Build the data loader",
            task_category=TaskCategory.CODE_DEVELOPMENT,
            estimated_complexity=Complexity.MEDIUM,
            instruction="Implement data loading pipeline.",
        )
        json_str = task.model_dump_json()
        restored = PaperBenchTask.model_validate_json(json_str)
        assert restored == task

    def test_optional_task_node_id(self) -> None:
        task = PaperBenchTask(
            engineer_id="eng-0",
            task_id="pb-1",
            requirements="Build something",
            task_category=TaskCategory.OTHER,
            estimated_complexity=Complexity.SIMPLE,
            instruction="Do it.",
        )
        assert task.task_node_id is None


class TestPaperBenchDelegation:
    def test_full_round_trip(self) -> None:
        delegation = PaperBenchDelegation(
            delegation_plan=PaperBenchDelegationPlan(
                first_round=PaperBenchFirstRound(
                    num_agents=2,
                    reasoning="Independent tasks first",
                    tasks=[
                        PaperBenchTask(
                            engineer_id="eng-0",
                            task_id="data",
                            requirements="Data loading",
                            task_category=TaskCategory.CODE_DEVELOPMENT,
                            estimated_complexity=Complexity.MEDIUM,
                            instruction="Build loader.",
                        )
                    ],
                ),
                remaining_tasks=[
                    PaperBenchRemainingTask(
                        task_id="train",
                        requirements="Training loop",
                        task_category=TaskCategory.CODE_DEVELOPMENT,
                        estimated_complexity=Complexity.COMPLEX,
                        depends_on=["data"],
                    )
                ],
            )
        )
        json_str = delegation.model_dump_json()
        restored = PaperBenchDelegation.model_validate_json(json_str)
        assert restored == delegation


class TestEngineerResult:
    def test_completed_result(self) -> None:
        result = EngineerResult(
            engineer_id="eng-0",
            task_id="task-1",
            status="completed",
            branch_name="caid/eng-0",
            commit_sha="abc123def",
            test_passed=True,
        )
        assert result.status == "completed"
        assert result.test_passed is True

    def test_partial_result(self) -> None:
        result = EngineerResult(
            engineer_id="eng-0",
            task_id="task-1",
            status="partial",
            branch_name="caid/eng-0",
            test_passed=False,
            error_log="Tests failed: 2 failures",
        )
        assert result.status == "partial"
        assert result.error_log is not None

    def test_round_trip(self) -> None:
        result = EngineerResult(
            engineer_id="eng-0",
            task_id="task-1",
            status="completed",
            branch_name="caid/eng-0",
            commit_sha="abc123",
            test_passed=True,
        )
        restored = EngineerResult.model_validate_json(result.model_dump_json())
        assert restored == result
