"""Tests for caid.git_ops module."""

from __future__ import annotations

from pathlib import Path

import pytest

from caid.git_ops import GitError, GitOps, MergeResult


class TestGitInit:
    def test_init_creates_repo(self, git_repo: tuple[Path, GitOps]) -> None:
        repo_dir, git = git_repo
        assert (repo_dir / ".git").exists()

    def test_initial_branch_is_main(self, git_repo: tuple[Path, GitOps]) -> None:
        _, git = git_repo
        assert git.get_current_branch() == "main"


class TestCommit:
    def test_commit_returns_sha(self, git_repo: tuple[Path, GitOps]) -> None:
        repo_dir, git = git_repo
        (repo_dir / "file.txt").write_text("content")
        sha = git.commit_all("Add file")
        assert sha is not None
        assert len(sha) >= 7

    def test_empty_commit(self, git_repo: tuple[Path, GitOps]) -> None:
        _, git = git_repo
        sha = git.commit_all("Empty commit")
        assert sha is not None  # --allow-empty


class TestWorktree:
    def test_worktree_add_and_remove(
        self, git_repo: tuple[Path, GitOps], tmp_dir: Path
    ) -> None:
        repo_dir, git = git_repo
        wt_path = tmp_dir / "wt-test"
        git.worktree_add(wt_path, "test-branch")

        assert wt_path.exists()
        worktrees = git.worktree_list()
        assert any("wt-test" in w for w in worktrees)

        git.worktree_remove(wt_path)
        assert not wt_path.exists()

    def test_worktree_has_files(
        self, git_repo: tuple[Path, GitOps], tmp_dir: Path
    ) -> None:
        repo_dir, git = git_repo
        (repo_dir / "main.py").write_text("print('hello')\n")
        git.commit_all("Add main.py")

        wt_path = tmp_dir / "wt-files"
        git.worktree_add(wt_path, "branch-files")
        assert (wt_path / "main.py").exists()
        content = (wt_path / "main.py").read_text()
        assert content == "print('hello')\n"

        git.worktree_remove(wt_path)

    def test_worktree_isolation(
        self, git_repo: tuple[Path, GitOps], tmp_dir: Path
    ) -> None:
        repo_dir, git = git_repo
        (repo_dir / "shared.py").write_text("original\n")
        git.commit_all("Add shared.py")

        wt_path = tmp_dir / "wt-iso"
        git.worktree_add(wt_path, "iso-branch")

        # Modify in worktree
        (wt_path / "shared.py").write_text("modified in worktree\n")

        # Main repo should still have original
        assert (repo_dir / "shared.py").read_text() == "original\n"

        git.worktree_remove(wt_path)


class TestMerge:
    def test_successful_merge(
        self, git_repo: tuple[Path, GitOps], tmp_dir: Path
    ) -> None:
        repo_dir, git = git_repo
        (repo_dir / "base.py").write_text("base\n")
        git.commit_all("Add base")

        wt_path = tmp_dir / "wt-merge"
        git.worktree_add(wt_path, "merge-branch")

        (wt_path / "new_file.py").write_text("new content\n")
        wt_git = GitOps(wt_path)
        wt_git.commit_all("Add new file")

        git.worktree_remove(wt_path)

        result = git.merge_branch("merge-branch")
        assert result.success
        assert result.commit_sha is not None
        assert (repo_dir / "new_file.py").exists()

    def test_merge_conflict_detection(
        self, git_repo: tuple[Path, GitOps], tmp_dir: Path
    ) -> None:
        repo_dir, git = git_repo
        (repo_dir / "conflict.py").write_text("original\n")
        git.commit_all("Add conflict.py")

        wt_path = tmp_dir / "wt-conflict"
        git.worktree_add(wt_path, "conflict-branch")

        # Modify in worktree
        (wt_path / "conflict.py").write_text("worktree version\n")
        wt_git = GitOps(wt_path)
        wt_git.commit_all("Worktree change")

        # Modify in main
        git.checkout("main")
        (repo_dir / "conflict.py").write_text("main version\n")
        git.commit_all("Main change")

        git.worktree_remove(wt_path)

        result = git.merge_branch("conflict-branch")
        assert not result.success
        assert len(result.conflict_files) > 0

        # Clean up
        git.abort_merge()

    def test_abort_merge(
        self, git_repo: tuple[Path, GitOps], tmp_dir: Path
    ) -> None:
        repo_dir, git = git_repo
        (repo_dir / "f.py").write_text("v1\n")
        git.commit_all("v1")

        wt_path = tmp_dir / "wt-abort"
        git.worktree_add(wt_path, "abort-branch")
        (wt_path / "f.py").write_text("v2-wt\n")
        GitOps(wt_path).commit_all("v2 worktree")

        git.checkout("main")
        (repo_dir / "f.py").write_text("v2-main\n")
        git.commit_all("v2 main")

        git.worktree_remove(wt_path)

        result = git.merge_branch("abort-branch")
        assert not result.success
        git.abort_merge()

        # Should be back on main cleanly
        assert git.get_current_branch() == "main"


class TestResetHard:
    def test_reset_hard(self, git_repo: tuple[Path, GitOps]) -> None:
        repo_dir, git = git_repo
        (repo_dir / "temp.py").write_text("temp\n")
        git.commit_all("Add temp")
        (repo_dir / "temp.py").write_text("modified\n")

        git.reset_hard()
        assert (repo_dir / "temp.py").read_text() == "temp\n"


class TestMergeResult:
    def test_success_result(self) -> None:
        result = MergeResult(success=True, commit_sha="abc123")
        assert result.success
        assert result.conflict_files == []

    def test_conflict_result(self) -> None:
        result = MergeResult(
            success=False,
            conflict_files=["a.py", "b.py"],
            error_message="CONFLICT",
        )
        assert not result.success
        assert len(result.conflict_files) == 2
