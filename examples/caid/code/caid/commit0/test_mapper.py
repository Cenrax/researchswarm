"""Test-to-file mapper for Commit0 mode.

Maps test files to source files so each engineer knows which tests
are relevant to their assigned task (CAID Section 2.4).
Uses both naming conventions and import analysis.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

SKIP_DIRS = {".git", "venv", ".venv", "__pycache__", "node_modules", ".tox"}


def get_relevant_tests(repo_path: Path, source_file: str) -> list[str]:
    """Find test files that exercise a given source file.

    Uses two strategies:
      1. Convention-based: test_<name>.py for <name>.py
      2. Import-based: test files that import the source module

    Args:
        repo_path: Repository root path.
        source_file: Relative path to the source file.

    Returns:
        List of relative paths to relevant test files.
    """
    repo_path = Path(repo_path).resolve()
    source_module = _file_to_module(source_file)
    source_stem = Path(source_file).stem
    relevant: set[str] = set()

    test_files = _find_test_files(repo_path)

    for test_file in test_files:
        rel = str(test_file.relative_to(repo_path))

        # Strategy 1: Name convention match
        if source_stem in test_file.stem:
            relevant.add(rel)
            continue

        # Strategy 2: Import-based match
        if _test_imports_module(test_file, source_module):
            relevant.add(rel)

    return sorted(relevant)


def build_test_map(repo_path: Path) -> dict[str, list[str]]:
    """Build a complete mapping of source files to their test files.

    Args:
        repo_path: Repository root path.

    Returns:
        Dict mapping source file relative paths to lists of test file paths.
    """
    repo_path = Path(repo_path).resolve()
    test_map: dict[str, list[str]] = {}

    source_files = [
        str(f.relative_to(repo_path))
        for f in repo_path.rglob("*.py")
        if "test" not in f.name
        and not any(part in SKIP_DIRS for part in f.parts)
    ]

    for src in source_files:
        tests = get_relevant_tests(repo_path, src)
        if tests:
            test_map[src] = tests

    return test_map


def _find_test_files(repo_path: Path) -> list[Path]:
    """Find all test files in the repository."""
    test_files: list[Path] = []
    for py_file in repo_path.rglob("test_*.py"):
        if not any(part in SKIP_DIRS for part in py_file.parts):
            test_files.append(py_file)
    # Also check for *_test.py pattern
    for py_file in repo_path.rglob("*_test.py"):
        if not any(part in SKIP_DIRS for part in py_file.parts):
            if py_file not in test_files:
                test_files.append(py_file)
    return test_files


def _test_imports_module(test_file: Path, source_module: str) -> bool:
    """Check if a test file imports the given source module."""
    try:
        tree = ast.parse(test_file.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return False

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if source_module in alias.name:
                    return True
        elif isinstance(node, ast.ImportFrom):
            if node.module and source_module in node.module:
                return True
    return False


def _file_to_module(file_path: str) -> str:
    """Convert a file path to a Python module name.

    Example: src/utils/parser.py -> src.utils.parser
    """
    parts = list(Path(file_path).parts)
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        parts[-1] = parts[-1].replace(".py", "")
    return ".".join(parts)


if __name__ == "__main__":
    import shutil
    import tempfile

    # Create a sample repo structure
    tmp = Path(tempfile.mkdtemp(prefix="caid-testmap-test-"))
    try:
        (tmp / "src").mkdir()
        (tmp / "src" / "utils.py").write_text("def helper():\n    pass\n")
        (tmp / "src" / "core.py").write_text("def main_logic():\n    pass\n")

        (tmp / "tests").mkdir()
        (tmp / "tests" / "test_utils.py").write_text(
            "from src.utils import helper\n\n"
            "def test_helper():\n    assert True\n"
        )
        (tmp / "tests" / "test_core.py").write_text(
            "from src.core import main_logic\n\n"
            "def test_main():\n    assert True\n"
        )
        (tmp / "tests" / "test_integration.py").write_text(
            "from src.utils import helper\n"
            "from src.core import main_logic\n\n"
            "def test_all():\n    assert True\n"
        )

        # Test get_relevant_tests
        tests_for_utils = get_relevant_tests(tmp, "src/utils.py")
        print(f"Tests for utils.py: {tests_for_utils}")
        assert "tests/test_utils.py" in tests_for_utils
        assert "tests/test_integration.py" in tests_for_utils

        tests_for_core = get_relevant_tests(tmp, "src/core.py")
        print(f"Tests for core.py: {tests_for_core}")
        assert "tests/test_core.py" in tests_for_core
        assert "tests/test_integration.py" in tests_for_core

        # Test build_test_map
        test_map = build_test_map(tmp)
        print(f"Test map: {test_map}")
        assert "src/utils.py" in test_map
        assert "src/core.py" in test_map

        # Test module conversion
        assert _file_to_module("src/utils.py") == "src.utils"
        assert _file_to_module("src/__init__.py") == "src"
        print("Module conversion OK")

        # Test empty case
        empty_tests = get_relevant_tests(tmp, "nonexistent.py")
        assert empty_tests == []
        print("Empty case OK")

        print("Test mapper module OK")
    finally:
        shutil.rmtree(tmp)
