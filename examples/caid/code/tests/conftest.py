"""Shared fixtures for CAID tests."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

from caid.config import CAIDConfig, LLMConfig
from caid.git_ops import GitOps
from caid.graph import DependencyGraph, TaskNode
from caid.llm import LLMClient


@pytest.fixture
def sample_config() -> CAIDConfig:
    """Default CAID config matching paper values."""
    return CAIDConfig(
        mode="commit0",
        max_engineers=4,
        manager_max_iterations=50,
        engineer_max_iterations=80,
        implementation_rounds=2,
    )


@pytest.fixture
def llm_config() -> LLMConfig:
    """LLM config for testing (no real API key)."""
    return LLMConfig(model="test-model", api_key="test-key")


@pytest.fixture
def llm_client(llm_config: LLMConfig) -> LLMClient:
    """LLM client instance for testing."""
    return LLMClient(llm_config)


@pytest.fixture
def sample_graph() -> DependencyGraph:
    """A 4-node diamond-shaped dependency graph: A -> B,C -> D."""
    g = DependencyGraph()
    for tid in ["A", "B", "C", "D"]:
        g.add_task(TaskNode(task_id=tid, file_path=f"{tid.lower()}.py"))
    g.add_dependency("A", "B")
    g.add_dependency("A", "C")
    g.add_dependency("B", "D")
    g.add_dependency("C", "D")
    return g


@pytest.fixture
def linear_graph() -> DependencyGraph:
    """A 3-node linear dependency graph: A -> B -> C."""
    g = DependencyGraph()
    for tid in ["A", "B", "C"]:
        g.add_task(TaskNode(task_id=tid))
    g.add_dependency("A", "B")
    g.add_dependency("B", "C")
    return g


@pytest.fixture
def tmp_dir() -> Path:
    """Temporary directory, cleaned up after test."""
    d = Path(tempfile.mkdtemp(prefix="caid-test-"))
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def git_repo(tmp_dir: Path) -> tuple[Path, GitOps]:
    """Initialized git repo in a temp directory."""
    repo_dir = tmp_dir / "repo"
    repo_dir.mkdir()
    git = GitOps(repo_dir)
    git.init_repo()
    return repo_dir, git


@pytest.fixture
def sample_repo(tmp_dir: Path) -> Path:
    """Sample Python repo with stubs for extractor testing."""
    repo = tmp_dir / "sample-repo"
    repo.mkdir()
    (repo / "src").mkdir()
    (repo / "src" / "__init__.py").write_text("")
    (repo / "src" / "base.py").write_text(
        "def setup():\n    pass\n\ndef validate():\n    pass\n"
    )
    (repo / "src" / "processor.py").write_text(
        "from src.base import setup\n\n"
        "def process():\n    pass\n"
    )
    (repo / "tests").mkdir()
    (repo / "tests" / "test_base.py").write_text(
        "from src.base import setup\n\n"
        "def test_setup():\n    assert True\n"
    )
    (repo / "tests" / "test_processor.py").write_text(
        "from src.processor import process\n\n"
        "def test_process():\n    assert True\n"
    )
    return repo
