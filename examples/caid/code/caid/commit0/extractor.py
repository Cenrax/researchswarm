"""Import-based dependency extractor for Commit0 mode.

Parses Python source files to find stub functions and extract
import-level dependencies, building the CAID dependency graph
automatically from repository structure (CAID Section 2.1).
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path

from caid.graph import DependencyGraph, TaskNode

logger = logging.getLogger(__name__)

# Threshold for splitting a file into function-level nodes
FUNCTION_SPLIT_THRESHOLD = 5


def extract_dependencies(repo_path: Path) -> DependencyGraph:
    """Build a dependency graph from Python import analysis.

    Steps:
      1. Find all .py files with stub functions (body is pass or ...).
      2. Extract intra-repo imports for each stub file.
      3. Build graph nodes (file-level or function-level).
      4. Add dependency edges based on imports.

    Args:
        repo_path: Path to the git repository root.

    Returns:
        A DependencyGraph with nodes for each implementation task.
    """
    repo_path = Path(repo_path).resolve()
    graph = DependencyGraph()

    # Step 1: Find stub files
    stub_files = _find_stub_files(repo_path)
    logger.info("Found %d files with stubs", len(stub_files))

    if not stub_files:
        return graph

    # Step 2: Build module-to-file mapping
    module_to_file = _build_module_map(repo_path)

    # Step 3: Extract imports and build nodes
    import_map: dict[str, list[str]] = {}  # rel_path -> list of imported rel_paths

    for file_path, stubs in stub_files.items():
        rel_path = str(file_path.relative_to(repo_path))
        imports = _extract_imports(file_path, module_to_file, repo_path)
        import_map[rel_path] = imports

        # Decide granularity
        if len(stubs) > FUNCTION_SPLIT_THRESHOLD:
            for func_name in stubs:
                task_id = f"{rel_path}::{func_name}"
                graph.add_task(
                    TaskNode(
                        task_id=task_id,
                        file_path=rel_path,
                        functions=[func_name],
                        complexity=_estimate_complexity(func_name, file_path),
                    )
                )
        else:
            graph.add_task(
                TaskNode(
                    task_id=rel_path,
                    file_path=rel_path,
                    functions=stubs,
                    complexity="medium",
                )
            )

    # Step 4: Add dependency edges
    all_task_files = {
        node_id.split("::")[0] if "::" in node_id else node_id
        for node_id in graph.all_tasks
    }

    for file_path, imports in import_map.items():
        for imported_file in imports:
            if imported_file in all_task_files:
                source_nodes = [
                    t
                    for t in graph.all_tasks
                    if t == imported_file or t.startswith(imported_file + "::")
                ]
                target_nodes = [
                    t
                    for t in graph.all_tasks
                    if t == file_path or t.startswith(file_path + "::")
                ]
                for src in source_nodes:
                    for tgt in target_nodes:
                        if src != tgt:
                            try:
                                graph.add_dependency(src, tgt)
                            except (ValueError, KeyError):
                                pass  # Skip cycles or missing nodes

    logger.info(
        "Dependency graph: %d tasks, %d edges",
        graph.node_count,
        graph.edge_count,
    )
    return graph


def _find_stub_files(repo_path: Path) -> dict[Path, list[str]]:
    """Find Python files containing stub functions.

    A stub function has a body consisting only of ``pass`` or ``...``.
    Skips files in .git, venv, __pycache__, and node_modules directories.
    """
    stubs: dict[Path, list[str]] = {}
    skip_dirs = {".git", "venv", ".venv", "__pycache__", "node_modules", ".tox"}

    for py_file in repo_path.rglob("*.py"):
        if any(part in skip_dirs for part in py_file.parts):
            continue
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue

        file_stubs: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if _is_stub(node):
                    file_stubs.append(node.name)

        if file_stubs:
            stubs[py_file] = file_stubs

    return stubs


def _is_stub(func_node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Check if a function body is just ``pass`` or ``...`` (Ellipsis).

    Also considers functions with only a docstring + pass/ellipsis.
    """
    body = func_node.body

    # Filter out docstrings
    meaningful = []
    for stmt in body:
        if (
            isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Constant)
            and isinstance(stmt.value.value, str)
        ):
            continue  # Skip docstring
        meaningful.append(stmt)

    if len(meaningful) == 0:
        return True  # Only docstring

    if len(meaningful) == 1:
        stmt = meaningful[0]
        if isinstance(stmt, ast.Pass):
            return True
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
            if stmt.value.value is ...:
                return True

    return False


def _build_module_map(repo_path: Path) -> dict[str, str]:
    """Build a mapping from module names to relative file paths.

    Example: foo/bar/baz.py -> {"foo.bar.baz": "foo/bar/baz.py"}
    """
    module_map: dict[str, str] = {}
    skip_dirs = {".git", "venv", ".venv", "__pycache__", "node_modules"}

    for py_file in repo_path.rglob("*.py"):
        if any(part in skip_dirs for part in py_file.parts):
            continue
        rel = py_file.relative_to(repo_path)
        parts = list(rel.parts)
        if parts[-1] == "__init__.py":
            parts = parts[:-1]
        else:
            parts[-1] = parts[-1].replace(".py", "")

        if parts:  # Non-empty
            module_name = ".".join(parts)
            module_map[module_name] = str(rel)

    return module_map


def _extract_imports(
    file_path: Path,
    module_map: dict[str, str],
    repo_path: Path,
) -> list[str]:
    """Extract intra-repo imports from a Python file.

    Only returns imports that map to files within the repository.
    External and stdlib imports are ignored.
    """
    try:
        tree = ast.parse(file_path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return []

    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in module_map:
                    imports.add(module_map[alias.name])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                if node.module in module_map:
                    imports.add(module_map[node.module])
                else:
                    # Try partial match (parent package)
                    parts = node.module.split(".")
                    for i in range(len(parts), 0, -1):
                        candidate = ".".join(parts[:i])
                        if candidate in module_map:
                            imports.add(module_map[candidate])
                            break

    return list(imports)


def _estimate_complexity(func_name: str, file_path: Path) -> str:
    """Heuristic complexity estimation for a function.

    Uses function name patterns and file size as signals.
    """
    # Simple heuristics
    if any(kw in func_name.lower() for kw in ["init", "setup", "config", "get"]):
        return "simple"
    if any(kw in func_name.lower() for kw in ["process", "transform", "compute"]):
        return "complex"
    return "medium"


if __name__ == "__main__":
    import shutil
    import tempfile

    # Create a sample repository structure
    tmp = Path(tempfile.mkdtemp(prefix="caid-extractor-test-"))
    try:
        # Create source files with stubs
        (tmp / "src").mkdir()

        (tmp / "src" / "__init__.py").write_text("")

        (tmp / "src" / "base.py").write_text(
            "def setup_config():\n    pass\n\n"
            "def validate():\n    pass\n"
        )

        (tmp / "src" / "processor.py").write_text(
            "from src.base import setup_config\n\n"
            "def process_data():\n    pass\n\n"
            "def transform_data():\n    pass\n"
        )

        (tmp / "src" / "main.py").write_text(
            "from src.processor import process_data\n"
            "from src.base import validate\n\n"
            "def run():\n    pass\n"
        )

        # Extract dependencies
        graph = extract_dependencies(tmp)

        print(f"Tasks: {sorted(graph.all_tasks)}")
        print(f"Node count: {graph.node_count}")
        print(f"Edge count: {graph.edge_count}")
        print(f"Topological order: {graph.topological_order()}")
        print(f"Initially assignable: {graph.get_assignable_tasks()}")

        # Verify base.py has no dependencies (should be first)
        topo = graph.topological_order()
        assert "src/base.py" in topo
        assert graph.is_ready("src/base.py")

        # Verify main.py depends on processor and base
        assert not graph.is_ready("src/main.py")

        # Complete base -> processor becomes ready
        graph.mark_completed("src/base.py")
        assignable = graph.get_assignable_tasks()
        print(f"After base complete: {assignable}")
        assert "src/processor.py" in assignable

        print("Extractor module OK")
    finally:
        shutil.rmtree(tmp)
