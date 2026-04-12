"""Tests for caid.commit0.test_mapper module."""

from __future__ import annotations

from pathlib import Path

from caid.commit0.test_mapper import (
    _file_to_module,
    build_test_map,
    get_relevant_tests,
)


class TestFileToModule:
    def test_regular_file(self) -> None:
        assert _file_to_module("src/utils.py") == "src.utils"

    def test_init_file(self) -> None:
        assert _file_to_module("src/__init__.py") == "src"

    def test_nested_path(self) -> None:
        assert _file_to_module("a/b/c/module.py") == "a.b.c.module"

    def test_root_file(self) -> None:
        assert _file_to_module("main.py") == "main"


class TestGetRelevantTests:
    def test_convention_based_match(self, sample_repo: Path) -> None:
        tests = get_relevant_tests(sample_repo, "src/base.py")
        assert "tests/test_base.py" in tests

    def test_import_based_match(self, sample_repo: Path) -> None:
        tests = get_relevant_tests(sample_repo, "src/processor.py")
        assert "tests/test_processor.py" in tests

    def test_no_matching_tests(self, sample_repo: Path) -> None:
        tests = get_relevant_tests(sample_repo, "nonexistent.py")
        assert tests == []

    def test_returns_sorted_list(self, sample_repo: Path) -> None:
        tests = get_relevant_tests(sample_repo, "src/base.py")
        assert tests == sorted(tests)


class TestBuildTestMap:
    def test_builds_complete_map(self, sample_repo: Path) -> None:
        test_map = build_test_map(sample_repo)
        assert "src/base.py" in test_map
        assert "src/processor.py" in test_map

    def test_test_files_not_mapped(self, sample_repo: Path) -> None:
        test_map = build_test_map(sample_repo)
        # Test files should not appear as keys
        assert not any("test_" in k for k in test_map)
