"""Git operations for CAID worktree isolation and integration.

Wraps git CLI commands for worktree management, commits, merges,
and conflict detection. Uses subprocess for worktree operations
(better support than GitPython) and direct CLI for all operations.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class GitError(Exception):
    """Raised when a git operation fails."""

    def __init__(self, command: str, stderr: str, returncode: int) -> None:
        self.command = command
        self.stderr = stderr
        self.returncode = returncode
        super().__init__(f"git {command} failed (rc={returncode}): {stderr}")


@dataclass
class MergeResult:
    """Result of a git merge operation."""

    success: bool
    conflict_files: list[str] = field(default_factory=list)
    commit_sha: Optional[str] = None
    error_message: Optional[str] = None


class GitOps:
    """Git operations manager for a repository.

    Provides worktree creation/removal, commit, merge with conflict
    detection, and state synchronization primitives used by CAID.
    """

    def __init__(self, repo_path: Path) -> None:
        self.repo_path = Path(repo_path).resolve()

    def _run(
        self,
        args: list[str],
        cwd: Optional[Path] = None,
        check: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        """Run a git command and return the result.

        Args:
            args: Git subcommand and arguments.
            cwd: Working directory (defaults to repo_path).
            check: If True, raise GitError on non-zero exit.
        """
        cwd = cwd or self.repo_path
        result = subprocess.run(
            ["git"] + args,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=120,
        )
        if check and result.returncode != 0:
            raise GitError(
                command=" ".join(args),
                stderr=result.stderr.strip(),
                returncode=result.returncode,
            )
        if result.returncode != 0:
            logger.debug(
                "git %s (rc=%d): %s",
                " ".join(args),
                result.returncode,
                result.stderr.strip(),
            )
        return result

    def init_repo(self, initial_branch: str = "main") -> None:
        """Initialize a new git repository with an initial commit."""
        self._run(["init", "-b", initial_branch], check=True)
        self._run(["config", "user.email", "caid@test.local"], check=True)
        self._run(["config", "user.name", "CAID"], check=True)
        # Create initial commit so HEAD exists
        self._run(["commit", "--allow-empty", "-m", "Initial commit"], check=True)

    def worktree_add(self, worktree_path: Path, branch_name: str) -> Path:
        """Create a new worktree with a new branch based on current HEAD.

        Returns the resolved worktree path.
        """
        worktree_path = Path(worktree_path).resolve()
        self._run(
            ["worktree", "add", "-b", branch_name, str(worktree_path)],
            check=True,
        )
        logger.info("Created worktree at %s (branch: %s)", worktree_path, branch_name)
        return worktree_path

    def worktree_remove(self, worktree_path: Path) -> None:
        """Remove a worktree and prune stale references."""
        worktree_path = Path(worktree_path).resolve()
        self._run(["worktree", "remove", str(worktree_path), "--force"])
        self._run(["worktree", "prune"])

    def worktree_list(self) -> list[str]:
        """List all worktree paths."""
        result = self._run(["worktree", "list", "--porcelain"], check=True)
        paths: list[str] = []
        for line in result.stdout.splitlines():
            if line.startswith("worktree "):
                paths.append(line.split(" ", 1)[1])
        return paths

    def commit_all(
        self,
        message: str,
        cwd: Optional[Path] = None,
    ) -> Optional[str]:
        """Stage all changes and commit.

        Returns the commit SHA, or None if there was nothing to commit.
        """
        work_dir = cwd or self.repo_path
        self._run(["add", "-A"], cwd=work_dir)
        result = self._run(
            ["commit", "-m", message, "--allow-empty"],
            cwd=work_dir,
        )
        if result.returncode == 0:
            sha_result = self._run(["rev-parse", "HEAD"], cwd=work_dir)
            return sha_result.stdout.strip()
        return None

    def merge_branch(
        self,
        branch_name: str,
        target_branch: str = "main",
    ) -> MergeResult:
        """Merge a branch into the target branch.

        Returns a MergeResult indicating success or conflict details.
        """
        # Checkout target branch in the main repo
        self._run(["checkout", target_branch], check=True)

        result = self._run(
            ["merge", branch_name, "--no-ff", "-m", f"Merge {branch_name}"]
        )

        if result.returncode == 0:
            sha_result = self._run(["rev-parse", "HEAD"])
            return MergeResult(
                success=True,
                conflict_files=[],
                commit_sha=sha_result.stdout.strip(),
            )

        # Detect conflict files
        status = self._run(["diff", "--name-only", "--diff-filter=U"])
        conflicts = [f.strip() for f in status.stdout.splitlines() if f.strip()]

        return MergeResult(
            success=False,
            conflict_files=conflicts,
            error_message=result.stderr.strip(),
        )

    def abort_merge(self) -> None:
        """Abort an in-progress merge."""
        self._run(["merge", "--abort"])

    def reset_hard(self, ref: str = "HEAD", cwd: Optional[Path] = None) -> None:
        """Reset the working tree to match a ref."""
        self._run(["reset", "--hard", ref], cwd=cwd or self.repo_path)

    def pull_main(self, cwd: Path) -> subprocess.CompletedProcess[str]:
        """Pull latest main into a worktree for conflict resolution.

        Uses merge (not rebase) to match CAID's merge-based integration.
        """
        return self._run(
            ["merge", "main", "--no-ff", "-m", "Merge main for sync"],
            cwd=cwd,
        )

    def get_current_branch(self, cwd: Optional[Path] = None) -> str:
        """Get the name of the currently checked out branch."""
        result = self._run(
            ["branch", "--show-current"],
            cwd=cwd or self.repo_path,
        )
        return result.stdout.strip()

    def checkout(self, branch: str, cwd: Optional[Path] = None) -> None:
        """Checkout a branch."""
        self._run(["checkout", branch], cwd=cwd or self.repo_path, check=True)

    def delete_branch(self, branch_name: str) -> None:
        """Delete a local branch."""
        self._run(["branch", "-D", branch_name])


if __name__ == "__main__":
    import shutil
    import tempfile

    # Create a temp directory for testing
    tmp = Path(tempfile.mkdtemp(prefix="caid-git-test-"))
    try:
        repo_dir = tmp / "repo"
        repo_dir.mkdir()
        git = GitOps(repo_dir)

        # Init repo
        git.init_repo()
        print(f"Repo initialized at {repo_dir}")
        print(f"Current branch: {git.get_current_branch()}")

        # Create a file and commit
        (repo_dir / "hello.py").write_text("print('hello')\n")
        sha = git.commit_all("Add hello.py")
        print(f"Committed: {sha}")

        # Create worktree
        wt_path = tmp / "wt-eng-0"
        git.worktree_add(wt_path, "caid/eng-0")
        print(f"Worktree created: {wt_path}")
        print(f"Worktrees: {git.worktree_list()}")

        # Make changes in worktree
        (wt_path / "hello.py").write_text("print('hello from engineer')\n")
        wt_git = GitOps(wt_path)
        sha2 = wt_git.commit_all("[eng-0] Update hello.py")
        print(f"Worktree commit: {sha2}")

        # Merge back to main
        merge_result = git.merge_branch("caid/eng-0")
        print(f"Merge result: success={merge_result.success}, sha={merge_result.commit_sha}")
        assert merge_result.success

        # Create a conflict scenario
        wt2_path = tmp / "wt-eng-1"
        git.worktree_add(wt2_path, "caid/eng-1")

        # Edit same file in main
        git.checkout("main")
        (repo_dir / "hello.py").write_text("print('hello from main')\n")
        git.commit_all("Update hello.py in main")

        # Edit same file in worktree
        (wt2_path / "hello.py").write_text("print('hello from engineer 1')\n")
        wt2_git = GitOps(wt2_path)
        wt2_git.commit_all("[eng-1] Update hello.py")

        # Attempt merge (should conflict)
        conflict_result = git.merge_branch("caid/eng-1")
        print(f"Conflict test: success={conflict_result.success}, "
              f"conflicts={conflict_result.conflict_files}")

        if not conflict_result.success:
            git.abort_merge()
            print("Merge aborted successfully")

        # Cleanup worktrees
        git.worktree_remove(wt_path)
        git.worktree_remove(wt2_path)
        print(f"Worktrees after cleanup: {git.worktree_list()}")

        print("Git ops module OK")
    finally:
        shutil.rmtree(tmp)
