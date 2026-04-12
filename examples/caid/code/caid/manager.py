"""Manager agent event loop for CAID.

Implements the centralized manager described in CAID Section 2.2:
explore repository -> build DAG -> delegate tasks -> await engineers
(FIRST_COMPLETED) -> merge -> reassign -> final review.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Optional

from caid.condenser import LLMSummarizingCondenser
from caid.config import CAIDConfig
from caid.engineer import EngineerAgent
from caid.git_ops import GitOps, MergeResult
from caid.graph import DependencyGraph, TaskNode
from caid.llm import LLMClient
from caid.schemas import (
    Commit0Task,
    EngineerResult,
    PaperBenchTask,
)

logger = logging.getLogger(__name__)

MANAGER_SYSTEM_PROMPT = """You are the Manager agent in a CAID multi-agent system.
Your role is to:
1. Analyze the repository/paper and build a dependency graph of tasks.
2. Delegate tasks to engineer agents as structured JSON.
3. Monitor progress and reassign tasks as dependencies are satisfied.

Mode: {mode}
Repository: {repo_path}
Task: {task_description}
Number of engineers: {max_engineers}

Respond with valid JSON matching the delegation schema."""


class ManagerAgent:
    """Central CAID manager that orchestrates engineer agents.

    Runs the full CAID lifecycle:
      1. Build dependency graph (import analysis or LLM decomposition)
      2. Create worktrees for engineers
      3. Delegate tasks via asyncio event loop
      4. Merge completed work, handle conflicts
      5. Reassign newly-ready tasks
      6. Condense context periodically
      7. Final review pass
    """

    def __init__(self, config: CAIDConfig) -> None:
        self.config = config
        self.llm = LLMClient(config.llm)
        self.git = GitOps(config.repo_path)
        self.graph: Optional[DependencyGraph] = None
        self.condenser = LLMSummarizingCondenser(self.llm)
        self.engineers: dict[str, EngineerAgent] = {}
        self.worktree_paths: dict[str, Path] = {}
        self.messages: list[dict[str, str]] = []

    async def run(self) -> dict[str, Any]:
        """Execute the full CAID manager event loop.

        Returns:
            Summary dict with completed/total tasks and token usage.
        """
        logger.info("CAID Manager starting (mode=%s)", self.config.mode)

        # Phase 1: Build dependency graph
        self.graph = await self._build_dependency_graph()
        logger.info(
            "Dependency graph: %d tasks, %d edges",
            self.graph.node_count,
            self.graph.edge_count,
        )

        if self.graph.node_count == 0:
            logger.warning("No tasks found. Nothing to do.")
            return self._build_summary()

        # Phase 2: Preparatory commit
        await self._preparatory_commit()

        # Phase 3: Setup engineers and worktrees
        self._setup_engineers()

        # Phase 4: Main delegation loop (per implementation round)
        try:
            for impl_round in range(self.config.implementation_rounds):
                logger.info(
                    "=== Implementation round %d/%d ===",
                    impl_round + 1,
                    self.config.implementation_rounds,
                )
                await self._delegation_loop()
                if self.graph.is_done():
                    break
        finally:
            # Phase 5: Cleanup worktrees even on error
            self._cleanup_worktrees()

        # Phase 6: Final review
        await self._final_review()

        return self._build_summary()

    async def _build_dependency_graph(self) -> DependencyGraph:
        """Build the dependency graph based on operating mode."""
        if self.config.mode == "commit0":
            from caid.commit0.extractor import extract_dependencies

            return extract_dependencies(self.config.repo_path)
        else:
            from caid.paperbench.decomposer import decompose_paper

            return await decompose_paper(self.config, self.llm)

    async def _preparatory_commit(self) -> None:
        """Commit any stubs or structural changes to main."""
        self.git.commit_all("CAID: preparatory commit (stubs)")

    def _setup_engineers(self) -> None:
        """Create worktrees and engineer agent instances."""
        self.config.worktree_base_dir.mkdir(parents=True, exist_ok=True)

        for i in range(self.config.max_engineers):
            eng_id = f"eng-{i}"
            worktree_path = self.config.worktree_base_dir / eng_id
            branch_name = f"caid/{eng_id}"

            try:
                self.git.worktree_add(worktree_path, branch_name)
                self.worktree_paths[eng_id] = worktree_path
                self.engineers[eng_id] = EngineerAgent(
                    engineer_id=eng_id,
                    config=self.config,
                    llm_client=LLMClient(self.config.llm),
                    git_ops=GitOps(worktree_path),
                )
                logger.info("Engineer %s ready at %s", eng_id, worktree_path)
            except Exception as e:
                logger.error("Failed to create engineer %s: %s", eng_id, e)

    async def _delegation_loop(self) -> None:
        """Core asyncio event loop for task delegation and integration.

        Uses asyncio.wait(FIRST_COMPLETED) to process engineer results
        as they arrive, then merges, updates the graph, and reassigns.
        """
        running_tasks: dict[str, asyncio.Task[EngineerResult]] = {}
        task_assignments: dict[str, str] = {}  # engineer_id -> task_id

        iteration = 0
        while iteration < self.config.manager_max_iterations:
            # Get assignable tasks
            if self.graph is None:
                raise RuntimeError("Graph not initialized")
            assignable = self.graph.get_assignable_tasks()

            if not assignable and not running_tasks:
                logger.info("No more tasks to assign and none running.")
                break

            # Assign to idle engineers
            idle_engineers = [
                eid for eid in self.engineers if eid not in running_tasks
            ]

            for eng_id, task_id in zip(idle_engineers, assignable):
                task_node = self.graph.get_node_data(task_id)
                task_spec = self._build_task_spec(eng_id, task_node)
                self.graph.mark_in_progress(task_id)
                task_assignments[eng_id] = task_id

                # Resolve relevant tests for Commit0 mode
                relevant_tests = self._get_relevant_tests(task_node)

                coro = self.engineers[eng_id].run(
                    task=task_spec,
                    worktree_path=self.worktree_paths[eng_id],
                    relevant_tests=relevant_tests,
                )
                running_tasks[eng_id] = asyncio.create_task(coro)
                logger.info("Assigned task %s to %s", task_id, eng_id)

            if not running_tasks:
                logger.info("No running tasks and no assignable tasks.")
                break

            # Await first completion (FIRST_COMPLETED per CAID design)
            done, _ = await asyncio.wait(
                running_tasks.values(),
                return_when=asyncio.FIRST_COMPLETED,
            )

            for finished in done:
                eng_id = next(
                    eid for eid, t in running_tasks.items() if t is finished
                )
                try:
                    result: EngineerResult = finished.result()
                except Exception as e:
                    logger.error("Engineer %s raised exception: %s", eng_id, e)
                    task_id = task_assignments.pop(eng_id, None)
                    if task_id:
                        self.graph.mark_failed(task_id)
                    del running_tasks[eng_id]
                    continue

                task_id = task_assignments.pop(eng_id, result.task_id)
                await self._integrate_result(eng_id, task_id, result)
                del running_tasks[eng_id]

            # Sync active worktrees to latest main
            self._sync_worktrees(running_tasks)

            # Condense context periodically (every 5 iterations)
            if iteration % 5 == 0 and iteration > 0:
                self.messages = self.condenser.condense_sync(
                    self.messages, self.graph
                )

            iteration += 1

            if self.graph.is_done():
                logger.info("All tasks completed!")
                break

    async def _integrate_result(
        self,
        eng_id: str,
        task_id: str,
        result: EngineerResult,
    ) -> None:
        """Merge an engineer's completed branch into main."""
        if self.graph is None:
            raise RuntimeError("Graph not initialized")

        merge_result: MergeResult = self.git.merge_branch(result.branch_name)

        if merge_result.success:
            self.graph.mark_completed(task_id)
            logger.info(
                "Merged task %s from %s (sha=%s)",
                task_id,
                eng_id,
                merge_result.commit_sha,
            )
            return

        # Conflict: route back to engineer for resolution
        logger.warning(
            "Merge conflict for %s: %s",
            task_id,
            merge_result.conflict_files,
        )
        self.git.abort_merge()

        try:
            await self.engineers[eng_id].resolve_conflict(
                self.worktree_paths[eng_id],
                merge_result.conflict_files,
            )
            # Retry merge after resolution
            retry_result = self.git.merge_branch(result.branch_name)
            if retry_result.success:
                self.graph.mark_completed(task_id)
                logger.info("Merged %s after conflict resolution", task_id)
            else:
                self.graph.mark_failed(task_id)
                logger.error(
                    "Failed to merge %s after conflict resolution", task_id
                )
        except Exception as e:
            self.graph.mark_failed(task_id)
            logger.error("Conflict resolution failed for %s: %s", task_id, e)

    def _sync_worktrees(
        self,
        running_tasks: dict[str, asyncio.Task[EngineerResult]],
    ) -> None:
        """Sync all active worktrees to latest main."""
        for eng_id, wt_path in self.worktree_paths.items():
            if eng_id not in running_tasks:
                try:
                    self.git.reset_hard("main", cwd=wt_path)
                except Exception as e:
                    logger.warning("Failed to sync worktree %s: %s", eng_id, e)

    def _build_task_spec(
        self,
        engineer_id: str,
        task_node: TaskNode,
    ) -> Commit0Task | PaperBenchTask:
        """Build the appropriate task spec based on mode."""
        if self.config.mode == "commit0":
            return Commit0Task(
                engineer_id=engineer_id,
                task_id=task_node.task_id,
                file_path=task_node.file_path,
                functions_to_implement=task_node.functions,
                complexity=task_node.complexity,
                instruction=task_node.metadata.get(
                    "instruction", "Implement the specified functions."
                ),
            )
        else:
            return PaperBenchTask(
                engineer_id=engineer_id,
                task_id=task_node.task_id,
                task_node_id=task_node.metadata.get("task_node_id"),
                requirements=task_node.metadata.get("requirements", ""),
                task_category=task_node.metadata.get(
                    "task_category", "Code Development"
                ),
                estimated_complexity=task_node.complexity,
                instruction=task_node.metadata.get(
                    "instruction", "Implement the task."
                ),
            )

    def _get_relevant_tests(self, task_node: TaskNode) -> list[str] | None:
        """Find relevant tests for a task (Commit0 mode only)."""
        if self.config.mode != "commit0" or not task_node.file_path:
            return None
        try:
            from caid.commit0.test_mapper import get_relevant_tests

            return get_relevant_tests(
                self.config.repo_path, task_node.file_path
            )
        except Exception:
            return None

    async def _final_review(self) -> None:
        """Manager final review pass before submission."""
        if self.graph is None:
            raise RuntimeError("Graph not initialized")
        completed = sorted(self.graph.completed)
        total = len(self.graph.all_tasks)
        logger.info(
            "Final review: %d/%d tasks completed",
            len(completed),
            total,
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are reviewing the final state of the repository "
                    "after all engineer agents have completed their tasks."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Completed tasks: {completed}\n"
                    f"Total tasks: {total}\n"
                    f"Review the repository and report any remaining issues."
                ),
            },
        ]
        try:
            await self.llm.call(messages)
        except Exception as e:
            logger.warning("Final review LLM call failed: %s", e)

    def _cleanup_worktrees(self) -> None:
        """Remove all engineer worktrees."""
        for eng_id, path in self.worktree_paths.items():
            try:
                self.git.worktree_remove(path)
                logger.debug("Removed worktree for %s", eng_id)
            except Exception as e:
                logger.warning("Failed to remove worktree %s: %s", path, e)

    def _build_summary(self) -> dict[str, Any]:
        """Build the final result summary."""
        if self.graph is None:
            raise RuntimeError("Graph not initialized")
        return {
            "completed_tasks": len(self.graph.completed),
            "total_tasks": len(self.graph.all_tasks),
            "completed_task_ids": sorted(self.graph.completed),
            "token_usage": {
                "prompt": self.llm.total_prompt_tokens,
                "completion": self.llm.total_completion_tokens,
            },
        }


if __name__ == "__main__":
    # Test manager initialization (no LLM calls)
    config = CAIDConfig(
        mode="commit0",
        repo_path=Path("/tmp/test-repo"),
        max_engineers=2,
    )
    manager = ManagerAgent(config)
    print(f"Manager created: mode={config.mode}, engineers={config.max_engineers}")
    print(f"Manager iterations: {config.manager_max_iterations}")

    # Test task spec building
    node = TaskNode(
        task_id="test-task",
        file_path="src/utils.py",
        functions=["func_a", "func_b"],
        complexity="medium",
        metadata={"instruction": "Implement these functions."},
    )
    spec = manager._build_task_spec("eng-0", node)
    print(f"Task spec type: {type(spec).__name__}")
    print(f"Task spec task_id: {spec.task_id}")

    # Test PaperBench mode
    config_pb = CAIDConfig(
        mode="paperbench",
        max_engineers=2,
    )
    manager_pb = ManagerAgent(config_pb)
    node_pb = TaskNode(
        task_id="pb-task",
        metadata={
            "task_node_id": "node-1",
            "requirements": "Build data loader",
            "task_category": "Code Development",
            "instruction": "Implement data pipeline.",
        },
    )
    spec_pb = manager_pb._build_task_spec("eng-0", node_pb)
    print(f"PaperBench spec type: {type(spec_pb).__name__}")

    print("Manager module OK")
