"""JSON message schemas for CAID manager-engineer communication.

All schemas are Pydantic v2 models matching the exact JSON formats
specified in the CAID paper appendix for both Commit0 and PaperBench modes.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# --- Enums ---


class Complexity(str, Enum):
    """Task complexity levels used for delegation prioritization."""

    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"


class TaskCategory(str, Enum):
    """PaperBench task categories."""

    CODE_DEVELOPMENT = "Code Development"
    EXPERIMENT_RUNNING = "Experiment Running"
    RESULTS_ANALYSIS = "Results Analysis"
    OTHER = "Other"


# --- Commit0 Schemas ---


class Commit0Task(BaseModel):
    """A single task assignment for Commit0 mode."""

    engineer_id: str
    task_id: str
    file_path: str
    functions_to_implement: list[str]
    complexity: Complexity
    instruction: str


class Commit0RemainingTask(BaseModel):
    """A task not yet assigned, waiting for dependencies."""

    task_id: str
    file_path: str
    functions_to_implement: list[str]
    complexity: Complexity
    depends_on: list[str] = Field(default_factory=list)


class Commit0FirstRound(BaseModel):
    """First round of Commit0 task delegation."""

    num_agents: int
    reasoning: str
    tasks: list[Commit0Task]


class Commit0DelegationPlan(BaseModel):
    """Complete Commit0 delegation plan from manager."""

    first_round: Commit0FirstRound
    remaining_tasks: list[Commit0RemainingTask]


class Commit0Delegation(BaseModel):
    """Top-level Commit0 delegation message."""

    delegation_plan: Commit0DelegationPlan


class Commit0Assignment(BaseModel):
    """A single task reassignment for Commit0 mode."""

    engineer_id: str
    task_id: str
    file_path: str
    functions_to_implement: list[str]
    instruction: str
    complexity: Complexity


class Commit0AssignTask(BaseModel):
    """Commit0 reassignment block."""

    reasoning: str
    assignments: list[Commit0Assignment]


class Commit0Reassignment(BaseModel):
    """Top-level Commit0 reassignment message."""

    assign_task: Commit0AssignTask


# --- PaperBench Schemas ---


class PaperBenchTask(BaseModel):
    """A single task assignment for PaperBench mode."""

    engineer_id: str
    task_id: str
    task_node_id: Optional[str] = None
    requirements: str
    task_category: TaskCategory
    estimated_complexity: Complexity
    instruction: str


class PaperBenchRemainingTask(BaseModel):
    """A PaperBench task waiting for dependencies."""

    task_id: str
    task_node_id: Optional[str] = None
    requirements: str
    task_category: TaskCategory
    estimated_complexity: Complexity
    depends_on: list[str] = Field(default_factory=list)


class PaperBenchFirstRound(BaseModel):
    """First round of PaperBench task delegation."""

    num_agents: int
    reasoning: str
    tasks: list[PaperBenchTask]


class PaperBenchDelegationPlan(BaseModel):
    """Complete PaperBench delegation plan."""

    first_round: PaperBenchFirstRound
    remaining_tasks: list[PaperBenchRemainingTask]


class PaperBenchDelegation(BaseModel):
    """Top-level PaperBench delegation message."""

    delegation_plan: PaperBenchDelegationPlan


class PaperBenchAssignTaskBlock(BaseModel):
    """PaperBench reassignment block."""

    reasoning: str
    tasks: list[PaperBenchTask]


class PaperBenchReassignment(BaseModel):
    """Top-level PaperBench reassignment message."""

    assign_task: PaperBenchAssignTaskBlock


# --- Engineer Result ---


class EngineerResult(BaseModel):
    """Result returned by an engineer agent after task execution."""

    engineer_id: str
    task_id: str
    status: str  # "completed", "partial", "conflict"
    branch_name: str
    commit_sha: Optional[str] = None
    test_passed: bool = False
    error_log: Optional[str] = None


if __name__ == "__main__":
    # Verify round-trip serialization for each schema
    task = Commit0Task(
        engineer_id="eng-0",
        task_id="task-1",
        file_path="src/utils.py",
        functions_to_implement=["parse_config", "validate_input"],
        complexity=Complexity.MEDIUM,
        instruction="Implement the utility functions.",
    )
    json_str = task.model_dump_json()
    restored = Commit0Task.model_validate_json(json_str)
    assert restored == task
    print(f"Commit0Task round-trip OK: {json_str[:80]}...")

    delegation = Commit0Delegation(
        delegation_plan=Commit0DelegationPlan(
            first_round=Commit0FirstRound(
                num_agents=2,
                reasoning="Start with independent files",
                tasks=[task],
            ),
            remaining_tasks=[
                Commit0RemainingTask(
                    task_id="task-2",
                    file_path="src/main.py",
                    functions_to_implement=["main"],
                    complexity=Complexity.COMPLEX,
                    depends_on=["task-1"],
                )
            ],
        )
    )
    json_str = delegation.model_dump_json()
    restored_del = Commit0Delegation.model_validate_json(json_str)
    assert restored_del == delegation
    print(f"Commit0Delegation round-trip OK")

    pb_task = PaperBenchTask(
        engineer_id="eng-0",
        task_id="pb-1",
        task_node_id="node-1",
        requirements="Implement the data loader",
        task_category=TaskCategory.CODE_DEVELOPMENT,
        estimated_complexity=Complexity.MEDIUM,
        instruction="Build the data loading pipeline.",
    )
    json_str = pb_task.model_dump_json()
    restored_pb = PaperBenchTask.model_validate_json(json_str)
    assert restored_pb == pb_task
    print(f"PaperBenchTask round-trip OK")

    result = EngineerResult(
        engineer_id="eng-0",
        task_id="task-1",
        status="completed",
        branch_name="caid/eng-0",
        commit_sha="abc123",
        test_passed=True,
    )
    json_str = result.model_dump_json()
    restored_r = EngineerResult.model_validate_json(json_str)
    assert restored_r == result
    print(f"EngineerResult round-trip OK")

    # Test validation error
    try:
        Commit0Task(
            engineer_id="eng-0",
            task_id="task-1",
            file_path="src/utils.py",
            functions_to_implement=["func"],
            complexity="invalid",  # type: ignore
            instruction="test",
        )
        print("ERROR: Should have raised ValidationError")
    except Exception as e:
        print(f"Validation error caught correctly: {type(e).__name__}")

    print("Schemas module OK")
