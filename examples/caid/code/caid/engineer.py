"""Engineer agent coroutine for CAID.

Implements the engineer lifecycle from CAID Section 2.4-2.5:
receive task spec -> implement code -> run tests -> fix failures -> commit.
Each engineer operates in an isolated git worktree.
"""

from __future__ import annotations

import logging
import re
import subprocess
import sys
from pathlib import Path
from typing import Union

from caid.config import CAIDConfig
from caid.git_ops import GitOps
from caid.llm import LLMClient
from caid.schemas import Commit0Task, EngineerResult, PaperBenchTask

logger = logging.getLogger(__name__)

ENGINEER_SYSTEM_PROMPT = """You are a software engineer agent in the CAID framework.
You receive a task specification and must implement the required code changes.
You operate in an isolated git worktree.

Rules:
- Only modify the files specified in your task.
- Do NOT modify any restricted files: {restricted_files}
- Write clean, well-documented code with type hints.
- Return the COMPLETE updated file content wrapped in ```python ... ``` code blocks.
- If fixing test failures, address all reported errors."""


class EngineerAgent:
    """Async engineer agent that implements tasks in isolated worktrees.

    Follows the implement -> test -> fix loop up to max_iterations,
    then commits (complete or partial) and returns a result.
    """

    def __init__(
        self,
        engineer_id: str,
        config: CAIDConfig,
        llm_client: LLMClient,
        git_ops: GitOps,
    ) -> None:
        self.engineer_id = engineer_id
        self.config = config
        self.llm = llm_client
        self.git = git_ops

    def _safe_resolve_path(self, worktree_path: Path, fname: str) -> Path | None:
        """Resolve fname within worktree_path, rejecting traversal attempts."""
        try:
            target = (worktree_path / fname).resolve()
            wt_base = worktree_path.resolve()
            if target == wt_base or str(target).startswith(str(wt_base) + "/"):
                return target
        except Exception:
            pass
        logger.error(
            "Engineer %s: rejected unsafe path: %r", self.engineer_id, fname
        )
        return None

    async def run(
        self,
        task: Union[Commit0Task, PaperBenchTask],
        worktree_path: Path,
        relevant_tests: list[str] | None = None,
    ) -> EngineerResult:
        """Main engineer coroutine.

        Args:
            task: The task specification from the manager.
            worktree_path: Path to the isolated git worktree.
            relevant_tests: Test files to run for self-verification.

        Returns:
            EngineerResult with status, branch, and commit info.
        """
        safe_task_id = task.task_id.replace("/", "-").replace("::", "-").replace(".", "-")
        branch_name = f"engineer-{self.engineer_id}-{safe_task_id}"
        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": ENGINEER_SYSTEM_PROMPT.format(
                    restricted_files=self.config.restricted_files
                ),
            },
            {
                "role": "user",
                "content": self._format_task_prompt(task, worktree_path),
            },
        ]

        for iteration in range(self.config.engineer_max_iterations):
            logger.info(
                "Engineer %s: iteration %d/%d for %s",
                self.engineer_id,
                iteration + 1,
                self.config.engineer_max_iterations,
                task.task_id,
            )

            # Step 1: Ask LLM for implementation
            response = await self.llm.call(messages)
            messages.append({"role": "assistant", "content": response})

            # Step 2: Apply code changes to worktree
            self._apply_code_changes(response, worktree_path, task)

            # Step 3: Run tests (self-verification)
            test_result = await self._run_tests(worktree_path, relevant_tests)

            if test_result["passed"]:
                sha = self.git.commit_all(
                    message=f"[{self.engineer_id}] Implement {task.task_id}",
                    cwd=worktree_path,
                )
                logger.info(
                    "Engineer %s: task %s completed (iteration %d)",
                    self.engineer_id,
                    task.task_id,
                    iteration + 1,
                )
                return EngineerResult(
                    engineer_id=self.engineer_id,
                    task_id=task.task_id,
                    status="completed",
                    branch_name=branch_name,
                    commit_sha=sha,
                    test_passed=True,
                )

            # Step 4: Feed errors back to LLM for next iteration
            error_msg = (
                f"Tests failed (iteration {iteration + 1}). "
                f"Output:\n{test_result['output']}\n\n"
                f"Fix the issues and return the complete corrected file."
            )
            messages.append({"role": "user", "content": error_msg})

        # Iteration limit reached -- partial commit
        logger.warning(
            "Engineer %s: iteration limit reached for %s",
            self.engineer_id,
            task.task_id,
        )
        sha = self.git.commit_all(
            message=f"[{self.engineer_id}] Partial {task.task_id}",
            cwd=worktree_path,
        )
        return EngineerResult(
            engineer_id=self.engineer_id,
            task_id=task.task_id,
            status="partial",
            branch_name=branch_name,
            commit_sha=sha,
            test_passed=False,
        )

    async def resolve_conflict(
        self,
        worktree_path: Path,
        conflict_files: list[str],
    ) -> EngineerResult:
        """Handle merge conflict re-routing from manager.

        Reads conflict markers, asks LLM to resolve, writes resolved
        files, and commits.
        """
        # Pull latest main into worktree
        self.git.pull_main(worktree_path)

        # Read conflict markers
        conflict_contents: dict[str, str] = {}
        for f in conflict_files:
            fpath = worktree_path / f
            if fpath.exists():
                conflict_contents[f] = fpath.read_text()

        messages = [
            {
                "role": "system",
                "content": (
                    "You are resolving a git merge conflict. "
                    "For each file, output the resolved content "
                    "(remove all conflict markers). "
                    "Wrap each file in ```python ... ``` with "
                    "# filename: <path> as the first line."
                ),
            },
            {
                "role": "user",
                "content": f"Conflict files:\n{_format_conflicts(conflict_contents)}",
            },
        ]
        response = await self.llm.call(messages)
        self._apply_resolved_files(response, worktree_path)

        sha = self.git.commit_all(
            message=f"[{self.engineer_id}] Resolve conflicts",
            cwd=worktree_path,
        )
        return EngineerResult(
            engineer_id=self.engineer_id,
            task_id="conflict-resolution",
            status="completed",
            branch_name=self.git.get_current_branch(worktree_path),
            commit_sha=sha,
            test_passed=False,
        )

    def _format_task_prompt(
        self,
        task: Union[Commit0Task, PaperBenchTask],
        worktree_path: Path,
    ) -> str:
        """Build the implementation prompt from the task spec."""
        if isinstance(task, Commit0Task):
            target = worktree_path / task.file_path
            existing = "(file does not exist)"
            if target.exists():
                existing = target.read_text()
            return (
                f"Task: {task.task_id}\n"
                f"File: {task.file_path}\n"
                f"Functions to implement: {task.functions_to_implement}\n"
                f"Complexity: {task.complexity.value}\n"
                f"Instruction: {task.instruction}\n\n"
                f"Current file content:\n```python\n{existing}\n```\n\n"
                f"Implement the specified functions. Return the COMPLETE "
                f"updated file content wrapped in ```python ... ``` code blocks."
            )
        else:
            return (
                f"Task: {task.task_id}\n"
                f"Category: {task.task_category.value}\n"
                f"Requirements: {task.requirements}\n"
                f"Complexity: {task.estimated_complexity.value}\n"
                f"Instruction: {task.instruction}\n\n"
                f"Implement this task. Return any code wrapped in "
                f"```python ... ``` blocks with # filename: <path> "
                f"as the first line of each block."
            )

    def _apply_code_changes(
        self,
        llm_response: str,
        worktree_path: Path,
        task: Union[Commit0Task, PaperBenchTask],
    ) -> None:
        """Extract code blocks from LLM response and write to files."""
        blocks = re.findall(r"```python\n(.*?)```", llm_response, re.DOTALL)

        if not blocks:
            logger.warning(
                "Engineer %s: no code blocks found in response",
                self.engineer_id,
            )
            return

        if isinstance(task, Commit0Task):
            # Write the first block to the task's target file
            target = self._safe_resolve_path(worktree_path, task.file_path)
            if target is None:
                return
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(blocks[0], encoding="utf-8")
            logger.debug("Wrote %d bytes to %s", len(blocks[0]), target)
        else:
            # PaperBench: look for filename hints
            for block in blocks:
                lines = block.split("\n")
                if lines[0].startswith("# filename:"):
                    fname = lines[0].split(":", 1)[1].strip()
                    target = self._safe_resolve_path(worktree_path, fname)
                    if target is None:
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text("\n".join(lines[1:]), encoding="utf-8")
                    logger.debug("Wrote %d bytes to %s", len(block), target)

    def _apply_resolved_files(
        self,
        llm_response: str,
        worktree_path: Path,
    ) -> None:
        """Apply conflict resolution output to worktree files."""
        blocks = re.findall(r"```python\n(.*?)```", llm_response, re.DOTALL)
        for block in blocks:
            lines = block.split("\n")
            if lines[0].startswith("# filename:"):
                fname = lines[0].split(":", 1)[1].strip()
                target = self._safe_resolve_path(worktree_path, fname)
                if target is None:
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("\n".join(lines[1:]), encoding="utf-8")

    async def _run_tests(
        self,
        worktree_path: Path,
        relevant_tests: list[str] | None,
    ) -> dict[str, object]:
        """Run pytest on relevant tests in the worktree.

        Returns a dict with 'passed' (bool) and 'output' (str).
        """
        cmd = [sys.executable, "-m", "pytest", "-x", "--tb=short", "-q"]
        if relevant_tests:
            cmd.extend(relevant_tests)

        try:
            result = subprocess.run(
                cmd,
                cwd=str(worktree_path),
                capture_output=True,
                text=True,
                timeout=300,
            )
            passed = result.returncode == 0
            output = result.stdout + "\n" + result.stderr
            # Truncate to avoid context explosion
            return {"passed": passed, "output": output[-3000:]}
        except subprocess.TimeoutExpired:
            return {"passed": False, "output": "Test execution timed out (300s)"}
        except FileNotFoundError:
            return {"passed": False, "output": "pytest not found. Install: pip install pytest"}


def _format_conflicts(conflicts: dict[str, str]) -> str:
    """Format conflict file contents for the LLM prompt."""
    parts: list[str] = []
    for path, content in conflicts.items():
        parts.append(f"--- {path} ---\n{content}")
    return "\n\n".join(parts)


if __name__ == "__main__":
    import asyncio
    import tempfile
    import shutil

    # Test the engineer agent with mock components
    tmp = Path(tempfile.mkdtemp(prefix="caid-eng-test-"))
    try:
        # Create a mock worktree
        worktree = tmp / "worktree"
        worktree.mkdir()
        (worktree / "src").mkdir()
        (worktree / "src" / "utils.py").write_text(
            "def parse_config():\n    pass\n\ndef validate_input():\n    pass\n"
        )

        # Create a mock git repo in worktree
        git = GitOps(worktree)
        git.init_repo()
        git.commit_all("Initial commit")

        # Test task prompt formatting
        config = CAIDConfig(restricted_files=["__init__.py"])
        from caid.config import LLMConfig

        llm = LLMClient(LLMConfig(model="test"))
        agent = EngineerAgent("eng-0", config, llm, git)

        task = Commit0Task(
            engineer_id="eng-0",
            task_id="task-1",
            file_path="src/utils.py",
            functions_to_implement=["parse_config", "validate_input"],
            complexity="medium",
            instruction="Implement the utility functions.",
        )

        prompt = agent._format_task_prompt(task, worktree)
        assert "parse_config" in prompt
        assert "validate_input" in prompt
        print(f"Task prompt formatted: {len(prompt)} chars")

        # Test code change application
        mock_response = '```python\ndef parse_config():\n    return {}\n\ndef validate_input():\n    return True\n```'
        agent._apply_code_changes(mock_response, worktree, task)
        content = (worktree / "src" / "utils.py").read_text()
        assert "return {}" in content
        print(f"Code changes applied: {len(content)} chars")

        # Test EngineerResult creation
        result = EngineerResult(
            engineer_id="eng-0",
            task_id="task-1",
            status="completed",
            branch_name="engineer-eng-0-task-1",
            commit_sha="abc123",
            test_passed=True,
        )
        print(f"Engineer result: {result.status}")

        print("Engineer module OK")
    finally:
        shutil.rmtree(tmp)
