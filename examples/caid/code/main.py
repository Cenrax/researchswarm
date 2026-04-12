"""CAID framework integration entry point.

Demonstrates the core components working together:
  1. Configuration loading
  2. Dependency graph construction
  3. Graph readiness and assignable task logic
  4. Git operations (worktree, commit, merge)
  5. Schema serialization

For actual multi-agent execution, use the CLI (caid run) or
the Python API (caid.api.run_caid).
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path


def demo_config() -> None:
    """Demonstrate configuration system."""
    from caid.config import CAIDConfig, LLMConfig

    config = CAIDConfig(
        mode="commit0",
        max_engineers=4,
        manager_max_iterations=50,
        engineer_max_iterations=80,
    )
    print(f"[Config] mode={config.mode}, engineers={config.max_engineers}")
    print(f"[Config] manager_iters={config.manager_max_iterations}, "
          f"engineer_iters={config.engineer_max_iterations}")
    print(f"[Config] LLM model={config.llm.model}")


def demo_schemas() -> None:
    """Demonstrate JSON schema round-trip."""
    from caid.schemas import (
        Commit0Delegation,
        Commit0DelegationPlan,
        Commit0FirstRound,
        Commit0RemainingTask,
        Commit0Task,
        Complexity,
        EngineerResult,
    )

    task = Commit0Task(
        engineer_id="eng-0",
        task_id="task-1",
        file_path="src/utils.py",
        functions_to_implement=["parse_config"],
        complexity=Complexity.MEDIUM,
        instruction="Implement parse_config.",
    )
    json_str = task.model_dump_json(indent=2)
    restored = Commit0Task.model_validate_json(json_str)
    assert restored == task
    print(f"[Schemas] Commit0Task round-trip OK")

    result = EngineerResult(
        engineer_id="eng-0",
        task_id="task-1",
        status="completed",
        branch_name="caid/eng-0",
        test_passed=True,
    )
    print(f"[Schemas] EngineerResult: {result.status}")


def demo_graph() -> None:
    """Demonstrate dependency graph with readiness checking."""
    from caid.graph import DependencyGraph, TaskNode

    graph = DependencyGraph()
    for tid in ["A", "B", "C", "D"]:
        graph.add_task(TaskNode(task_id=tid, file_path=f"{tid.lower()}.py"))

    graph.add_dependency("A", "B")
    graph.add_dependency("A", "C")
    graph.add_dependency("B", "D")
    graph.add_dependency("C", "D")

    print(f"[Graph] {graph.node_count} nodes, {graph.edge_count} edges")
    print(f"[Graph] Topological order: {graph.topological_order()}")

    # Simulate execution
    print(f"[Graph] Assignable: {graph.get_assignable_tasks()}")
    graph.mark_completed("A")
    print(f"[Graph] After A done: {graph.get_assignable_tasks()}")
    graph.mark_completed("B")
    graph.mark_completed("C")
    print(f"[Graph] After B,C done: {graph.get_assignable_tasks()}")
    graph.mark_completed("D")
    print(f"[Graph] All done: {graph.is_done()}")


def demo_git_ops() -> None:
    """Demonstrate git operations."""
    from caid.git_ops import GitOps

    tmp = Path(tempfile.mkdtemp(prefix="caid-demo-"))
    try:
        repo_dir = tmp / "repo"
        repo_dir.mkdir()
        git = GitOps(repo_dir)
        git.init_repo()

        (repo_dir / "main.py").write_text("print('hello')\n")
        sha = git.commit_all("Initial file")
        print(f"[Git] Committed: {sha[:8]}")

        wt_path = tmp / "wt-0"
        git.worktree_add(wt_path, "caid/eng-0")
        print(f"[Git] Worktree created: {wt_path.name}")

        (wt_path / "main.py").write_text("print('modified')\n")
        from caid.git_ops import GitOps as GO
        wt_git = GO(wt_path)
        wt_git.commit_all("[eng-0] Update main.py")

        result = git.merge_branch("caid/eng-0")
        print(f"[Git] Merge: success={result.success}")

        git.worktree_remove(wt_path)
        print(f"[Git] Worktree cleaned up")
    finally:
        shutil.rmtree(tmp)


def demo_extractor() -> None:
    """Demonstrate Commit0 dependency extraction."""
    from caid.commit0.extractor import extract_dependencies

    tmp = Path(tempfile.mkdtemp(prefix="caid-extract-demo-"))
    try:
        (tmp / "base.py").write_text("def setup():\n    pass\n")
        (tmp / "app.py").write_text(
            "from base import setup\n\ndef run():\n    pass\n"
        )

        graph = extract_dependencies(tmp)
        print(f"[Extractor] Tasks: {sorted(graph.all_tasks)}")
        print(f"[Extractor] Order: {graph.topological_order()}")
    finally:
        shutil.rmtree(tmp)


def demo_llm() -> None:
    """Demonstrate LLM client and JSON extraction."""
    from caid.llm import LLMClient, extract_json
    from caid.config import LLMConfig

    result = extract_json('```json\n{"key": "value"}\n```')
    print(f"[LLM] JSON extraction: {result}")

    client = LLMClient(LLMConfig(model="test"))
    print(f"[LLM] Token usage: {client.token_usage}")


def main() -> None:
    """Run all component demonstrations."""
    print("=" * 60)
    print("CAID Framework - Integration Demo")
    print("=" * 60)

    demos = [
        ("Configuration", demo_config),
        ("Schemas", demo_schemas),
        ("Dependency Graph", demo_graph),
        ("Git Operations", demo_git_ops),
        ("Dependency Extractor", demo_extractor),
        ("LLM Client", demo_llm),
    ]

    for name, func in demos:
        print(f"\n--- {name} ---")
        try:
            func()
        except Exception as e:
            print(f"[ERROR] {name}: {e}")

    print("\n" + "=" * 60)
    print("All demos completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()
