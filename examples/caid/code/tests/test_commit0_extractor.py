"""Tests for caid.commit0.extractor module."""

from __future__ import annotations

from pathlib import Path

from caid.commit0.extractor import (
    _build_module_map,
    _find_stub_files,
    _is_stub,
    extract_dependencies,
)


class TestFindStubFiles:
    def test_finds_pass_stubs(self, sample_repo: Path) -> None:
        stubs = _find_stub_files(sample_repo)
        paths = {str(p.relative_to(sample_repo)) for p in stubs}
        assert "src/base.py" in paths
        assert "src/processor.py" in paths

    def test_excludes_non_stub_files(self, tmp_dir: Path) -> None:
        (tmp_dir / "real.py").write_text("def func():\n    return 42\n")
        stubs = _find_stub_files(tmp_dir)
        assert len(stubs) == 0

    def test_finds_ellipsis_stubs(self, tmp_dir: Path) -> None:
        (tmp_dir / "stub.py").write_text("def func():\n    ...\n")
        stubs = _find_stub_files(tmp_dir)
        assert len(stubs) == 1

    def test_finds_docstring_plus_pass(self, tmp_dir: Path) -> None:
        (tmp_dir / "stub.py").write_text(
            'def func():\n    """Docstring."""\n    pass\n'
        )
        stubs = _find_stub_files(tmp_dir)
        assert len(stubs) == 1

    def test_skips_git_directory(self, tmp_dir: Path) -> None:
        git_dir = tmp_dir / ".git"
        git_dir.mkdir()
        (git_dir / "stub.py").write_text("def func():\n    pass\n")
        stubs = _find_stub_files(tmp_dir)
        assert len(stubs) == 0

    def test_handles_syntax_errors(self, tmp_dir: Path) -> None:
        (tmp_dir / "bad.py").write_text("def func(\n")
        stubs = _find_stub_files(tmp_dir)
        assert len(stubs) == 0


class TestBuildModuleMap:
    def test_regular_files(self, sample_repo: Path) -> None:
        module_map = _build_module_map(sample_repo)
        assert "src.base" in module_map
        assert module_map["src.base"] == "src/base.py"

    def test_init_files(self, sample_repo: Path) -> None:
        module_map = _build_module_map(sample_repo)
        assert "src" in module_map


class TestExtractDependencies:
    def test_builds_correct_graph(self, sample_repo: Path) -> None:
        graph = extract_dependencies(sample_repo)
        assert "src/base.py" in graph.all_tasks
        assert "src/processor.py" in graph.all_tasks

    def test_correct_dependency_direction(self, sample_repo: Path) -> None:
        graph = extract_dependencies(sample_repo)
        # processor imports from base, so base -> processor
        assert graph.is_ready("src/base.py")
        assert not graph.is_ready("src/processor.py")

    def test_topological_order(self, sample_repo: Path) -> None:
        graph = extract_dependencies(sample_repo)
        order = graph.topological_order()
        base_idx = order.index("src/base.py")
        proc_idx = order.index("src/processor.py")
        assert base_idx < proc_idx

    def test_empty_repo(self, tmp_dir: Path) -> None:
        graph = extract_dependencies(tmp_dir)
        assert graph.node_count == 0

    def test_independent_files(self, tmp_dir: Path) -> None:
        (tmp_dir / "a.py").write_text("def func_a():\n    pass\n")
        (tmp_dir / "b.py").write_text("def func_b():\n    pass\n")
        graph = extract_dependencies(tmp_dir)
        assert graph.node_count == 2
        assert graph.edge_count == 0
        # Both should be assignable
        assert set(graph.get_assignable_tasks()) == {"a.py", "b.py"}

    def test_function_level_splitting(self, tmp_dir: Path) -> None:
        # Create a file with more than 5 stubs
        stubs = "\n\n".join(
            f"def func_{i}():\n    pass" for i in range(7)
        )
        (tmp_dir / "big.py").write_text(stubs)
        graph = extract_dependencies(tmp_dir)
        # Should be split into function-level nodes
        assert graph.node_count == 7
        assert all("::" in t for t in graph.all_tasks)
