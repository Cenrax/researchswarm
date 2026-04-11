"""LLM-driven paper task decomposer for PaperBench mode.

Reads a paper description and uses the LLM to decompose it into
a dependency graph of implementation tasks (CAID Section 2.1).
"""

from __future__ import annotations

import logging
from pathlib import Path

from caid.config import CAIDConfig
from caid.graph import DependencyGraph, TaskNode
from caid.llm import LLMClient
from caid.schemas import PaperBenchDelegation

logger = logging.getLogger(__name__)

DECOMPOSE_PROMPT = """You are analyzing a research paper to create an implementation plan.
Break the paper's contributions into discrete, implementable tasks.
For each task, identify:
- A unique task_id (short, descriptive string)
- Requirements (what needs to be built)
- Category: one of "Code Development", "Experiment Running", "Results Analysis", "Other"
- Complexity: one of "simple", "medium", "complex"
- Dependencies on other tasks (by task_id)

Assign the first round of tasks (those with no dependencies) to engineers.
Use {num_engineers} engineers maximum.

Output as JSON matching this exact schema:
{{
  "delegation_plan": {{
    "first_round": {{
      "num_agents": <int>,
      "reasoning": "<string>",
      "tasks": [
        {{
          "engineer_id": "<eng-0, eng-1, etc>",
          "task_id": "<string>",
          "task_node_id": "<optional rubric node id>",
          "requirements": "<string>",
          "task_category": "<category>",
          "estimated_complexity": "<complexity>",
          "instruction": "<detailed instruction>"
        }}
      ]
    }},
    "remaining_tasks": [
      {{
        "task_id": "<string>",
        "task_node_id": "<optional>",
        "requirements": "<string>",
        "task_category": "<category>",
        "estimated_complexity": "<complexity>",
        "depends_on": ["<task_id>"]
      }}
    ]
  }}
}}

Paper content:
{paper_content}"""


async def decompose_paper(
    config: CAIDConfig,
    llm: LLMClient,
) -> DependencyGraph:
    """Decompose a paper into a dependency graph of tasks.

    Uses the LLM to read the paper content and produce a structured
    delegation plan, which is then converted to a DependencyGraph.

    Args:
        config: CAID configuration with task_description.
        llm: LLM client for making the decomposition call.

    Returns:
        A DependencyGraph with nodes for each paper task.
    """
    paper_content = _read_paper(config)

    messages = [
        {"role": "system", "content": "You are a research paper analyst."},
        {
            "role": "user",
            "content": DECOMPOSE_PROMPT.format(
                paper_content=paper_content,
                num_engineers=config.max_engineers,
            ),
        },
    ]

    delegation = await llm.call(messages, response_model=PaperBenchDelegation)
    return _delegation_to_graph(delegation)


def _delegation_to_graph(delegation: PaperBenchDelegation) -> DependencyGraph:
    """Convert a PaperBench delegation plan to a DependencyGraph."""
    graph = DependencyGraph()

    # Add first round tasks (no dependencies)
    for task in delegation.delegation_plan.first_round.tasks:
        graph.add_task(
            TaskNode(
                task_id=task.task_id,
                metadata={
                    "task_node_id": task.task_node_id,
                    "requirements": task.requirements,
                    "task_category": task.task_category.value,
                    "instruction": task.instruction,
                    "complexity": task.estimated_complexity.value,
                },
            )
        )

    # Pass 1: Add all remaining task nodes
    for task in delegation.delegation_plan.remaining_tasks:
        graph.add_task(
            TaskNode(
                task_id=task.task_id,
                metadata={
                    "task_node_id": task.task_node_id,
                    "requirements": task.requirements,
                    "task_category": task.task_category.value,
                    "complexity": task.estimated_complexity.value,
                },
            )
        )

    # Pass 2: Add all dependency edges for remaining tasks
    for task in delegation.delegation_plan.remaining_tasks:
        for dep in task.depends_on:
            if dep in graph.all_tasks:
                try:
                    graph.add_dependency(dep, task.task_id)
                except ValueError:
                    logger.warning(
                        "Skipping cyclic dependency: %s -> %s",
                        dep,
                        task.task_id,
                    )

    logger.info(
        "Paper decomposition: %d tasks, %d edges",
        graph.node_count,
        graph.edge_count,
    )
    return graph


def _read_paper(config: CAIDConfig) -> str:
    """Read paper content from task description or file path.

    Supports:
      - Direct text in task_description
      - Path to .md or .txt file
    """
    desc = config.task_description

    if desc.endswith((".md", ".txt")):
        paper_path = Path(desc)
        if paper_path.exists():
            return paper_path.read_text(encoding="utf-8")

    # Also check if it looks like an absolute path
    if desc.startswith("/") or desc.startswith("~"):
        paper_path = Path(desc).expanduser()
        if paper_path.exists():
            return paper_path.read_text(encoding="utf-8")

    return desc


if __name__ == "__main__":
    from caid.schemas import (
        Complexity,
        PaperBenchDelegation,
        PaperBenchDelegationPlan,
        PaperBenchFirstRound,
        PaperBenchRemainingTask,
        PaperBenchTask,
        TaskCategory,
    )

    # Test conversion without LLM by creating a mock delegation
    delegation = PaperBenchDelegation(
        delegation_plan=PaperBenchDelegationPlan(
            first_round=PaperBenchFirstRound(
                num_agents=2,
                reasoning="Start with data loader and model architecture",
                tasks=[
                    PaperBenchTask(
                        engineer_id="eng-0",
                        task_id="data-loader",
                        requirements="Build dataset and data loading pipeline",
                        task_category=TaskCategory.CODE_DEVELOPMENT,
                        estimated_complexity=Complexity.MEDIUM,
                        instruction="Implement the data loading module.",
                    ),
                    PaperBenchTask(
                        engineer_id="eng-1",
                        task_id="model-arch",
                        requirements="Implement the neural network architecture",
                        task_category=TaskCategory.CODE_DEVELOPMENT,
                        estimated_complexity=Complexity.COMPLEX,
                        instruction="Build the model as described in Section 3.",
                    ),
                ],
            ),
            remaining_tasks=[
                PaperBenchRemainingTask(
                    task_id="training",
                    requirements="Training loop with optimization",
                    task_category=TaskCategory.CODE_DEVELOPMENT,
                    estimated_complexity=Complexity.COMPLEX,
                    depends_on=["data-loader", "model-arch"],
                ),
                PaperBenchRemainingTask(
                    task_id="evaluation",
                    requirements="Run experiments and generate results",
                    task_category=TaskCategory.EXPERIMENT_RUNNING,
                    estimated_complexity=Complexity.MEDIUM,
                    depends_on=["training"],
                ),
            ],
        )
    )

    graph = _delegation_to_graph(delegation)
    print(f"Tasks: {sorted(graph.all_tasks)}")
    print(f"Nodes: {graph.node_count}, Edges: {graph.edge_count}")
    print(f"Topological order: {graph.topological_order()}")

    # Verify initial tasks are assignable
    assignable = graph.get_assignable_tasks()
    print(f"Initially assignable: {assignable}")
    assert "data-loader" in assignable
    assert "model-arch" in assignable

    # Training should not be assignable yet
    assert "training" not in assignable

    # Complete first round -> training becomes assignable
    graph.mark_completed("data-loader")
    graph.mark_completed("model-arch")
    assignable = graph.get_assignable_tasks()
    print(f"After first round: {assignable}")
    assert "training" in assignable

    print("Decomposer module OK")
