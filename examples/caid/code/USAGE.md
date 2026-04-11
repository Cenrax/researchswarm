# CAID Framework -- Installation and Usage Guide

CAID (Centralized Asynchronous Isolated Delegation) is a multi-agent framework
for automated software engineering. It uses a central Manager agent that
decomposes work into a dependency graph and delegates tasks to parallel
Engineer agents, each operating in isolated git worktrees.

---

## 1. Prerequisites

Before installing CAID, ensure you have the following:

- **Python 3.11 or later** (check with `python3 --version`)
- **git 2.5 or later** (required for `git worktree` support; check with `git --version`)
- **pip** (bundled with Python 3.11+)
- **pytest** installed in the target repository's environment (for engineer self-verification)

---

## 2. Installation

### Clone and install in a virtual environment

```bash
# Clone the repository
git clone <your-caid-repo-url>
cd caid

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate

# Install CAID and its dependencies
pip install -e .

# Or install from requirements.txt without the package:
pip install -r requirements.txt
```

### Verify the installation

```bash
# Check the CLI is available
caid --help

# Run the test suite
python -m pytest tests/ -v
```

---

## 3. LLM Provider Setup

CAID uses [litellm](https://docs.litellm.ai/) as a unified interface to LLM
providers. This means you can use any provider supported by litellm.

### Setting your API key

The recommended approach is to use an environment variable:

```bash
export CAID_LLM_API_KEY=sk-your-key-here
```

You can also set it in `caid.yaml`, but environment variables are preferred to
avoid accidentally committing secrets. The `.gitignore` excludes `caid.yaml`
by default.

### Model identifiers

CAID passes the `model` field directly to litellm. Use litellm's model
naming convention:

| Provider   | Model identifier example          |
|------------|-----------------------------------|
| Anthropic  | `claude-sonnet-4-5-20260120`          |
| OpenAI     | `openai/gpt-4o`                  |
| Azure      | `azure/gpt-4o`                   |
| Google     | `gemini/gemini-1.5-pro`          |
| Local      | `ollama/llama3`                  |

Set the model in your configuration:

```yaml
# caid.yaml
llm:
  provider: anthropic
  model: claude-sonnet-4-5-20260120
```

For OpenAI, also set:

```bash
export OPENAI_API_KEY=sk-...
```

For Anthropic:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
# Or use the unified key:
export CAID_LLM_API_KEY=sk-ant-...
```

---

## 4. Configuration Reference

CAID is configured via a YAML file (`caid.yaml`) and/or environment variables.
Copy the example to get started:

```bash
cp caid.yaml.example caid.yaml
```

### Full `caid.yaml` reference

```yaml
# Operating mode: "commit0" or "paperbench"
mode: commit0

# Path to the target git repository (must have an initial commit)
repo_path: ./my-repo

# Free-text description or path to paper file (PaperBench mode)
task_description: ""

# Number of concurrent engineer agents (paper default: 4 for Commit0, 2 for PaperBench)
max_engineers: 4

# Manager delegation loop iteration cap (paper default: 50)
manager_max_iterations: 50

# Per-engineer implement/test/fix iteration cap (paper default: 80)
engineer_max_iterations: 80

# Number of full delegation rounds (paper default: 2)
implementation_rounds: 2

# Files that engineers must not modify
restricted_files:
  - __init__.py

# Directory for isolated git worktrees
worktree_base_dir: /tmp/caid-worktrees

# LLM provider configuration
llm:
  provider: anthropic
  model: claude-sonnet-4-5-20260120
  # Prefer environment variable CAID_LLM_API_KEY over putting key here
  api_key: ""
  temperature: 0.0
  max_tokens: 4096
```

### Environment variable overrides

| Variable             | Overrides          |
|----------------------|--------------------|
| `CAID_LLM_API_KEY`  | `llm.api_key`     |
| `CAID_MODE`         | `mode`            |
| `CAID_MAX_ENGINEERS` | `max_engineers`   |
| `CAID_REPO_PATH`    | `repo_path`       |

---

## 5. Quick Start

Step-by-step from zero to running CAID on a real repository:

```bash
# 1. Install CAID (see Section 2)
pip install -e .

# 2. Set your API key
export CAID_LLM_API_KEY=sk-ant-your-key

# 3. Prepare a target repository with stub functions
cd /path/to/your/repo
git init && git add -A && git commit -m "Initial stubs"

# 4. Run CAID
caid run --repo /path/to/your/repo --mode commit0 --engineers 4

# 5. Check results
cd /path/to/your/repo
git log --oneline
python -m pytest
```

---

## 6. Commit0 Mode

Commit0 mode fills in stub functions (functions whose body is just `pass` or
`...`) in an existing Python repository.

### How it works

1. CAID scans the repository for Python files containing stub functions.
2. It analyzes imports to build a dependency graph (files that import from
   other files depend on them).
3. The Manager delegates tasks to Engineers in dependency order.
4. Each Engineer implements the stubs, runs pytest for verification, and
   iterates on failures.
5. Completed work is merged back to the main branch.

### Example

```bash
# Suppose you have a repo with stubs:
# src/base.py      -> setup(), validate()
# src/processor.py  -> process() (imports from src.base)
# src/main.py       -> run() (imports from src.processor, src.base)

caid run \
  --repo ./my-python-repo \
  --mode commit0 \
  --engineers 4 \
  --verbose
```

### Requirements for the target repository

- Must be a git repository with at least one commit.
- Stub functions must use `pass` or `...` as their body.
- A docstring followed by `pass` is also recognized as a stub.
- Test files in a `tests/` directory are automatically mapped to source files.

---

## 7. PaperBench Mode

PaperBench mode reproduces a research paper from scratch by decomposing it
into implementation tasks.

### How it works

1. CAID reads the paper description (text or `.md`/`.txt` file).
2. The LLM decomposes the paper into a structured task graph with
   categories (Code Development, Experiment Running, Results Analysis).
3. Tasks are delegated to Engineers based on dependencies.

### Example

```bash
# With a paper description file:
caid run \
  --repo ./paper-implementation \
  --mode paperbench \
  --task-description ./paper_summary.md \
  --engineers 2

# Or with inline description:
caid run \
  --repo ./paper-implementation \
  --mode paperbench \
  --task-description "Implement the Transformer architecture from Attention Is All You Need" \
  --engineers 2
```

Note: The paper recommends using 2 engineers for PaperBench mode.

---

## 8. Python API Usage

CAID can be used programmatically for integration into other tools:

### Async API

```python
import asyncio
from caid.api import run_caid

async def main():
    result = await run_caid(
        repo_path="./my-repo",
        mode="commit0",
        max_engineers=4,
        model="claude-sonnet-4-5-20260120",
        api_key="sk-ant-...",
    )
    print(f"Completed: {result['completed_tasks']}/{result['total_tasks']}")
    print(f"Tokens used: {result['token_usage']}")

asyncio.run(main())
```

### Sync API

```python
from caid.api import run_caid_sync

result = run_caid_sync(
    repo_path="./my-repo",
    mode="commit0",
    max_engineers=4,
)
print(f"Completed: {result['completed_tasks']}/{result['total_tasks']}")
```

### Custom configuration

```python
from caid.config import CAIDConfig, LLMConfig
from caid.manager import ManagerAgent

config = CAIDConfig(
    mode="commit0",
    repo_path="./my-repo",
    max_engineers=4,
    llm=LLMConfig(
        model="openai/gpt-4o",
        api_key="sk-...",
    ),
)

manager = ManagerAgent(config)
result = asyncio.run(manager.run())
```

### Using the dependency graph directly

```python
from pathlib import Path
from caid.commit0.extractor import extract_dependencies

graph = extract_dependencies(Path("./my-repo"))
print(f"Tasks: {graph.node_count}, Edges: {graph.edge_count}")
print(f"Topological order: {graph.topological_order()}")
print(f"Initially assignable: {graph.get_assignable_tasks()}")
```

---

## 9. CLI Reference

```
Usage: caid [OPTIONS] COMMAND [ARGS]...

Options:
  -v, --verbose    Enable verbose (DEBUG) logging
  --help           Show help message

Commands:
  run    Run the CAID framework on a repository
  graph  Display the dependency graph for a repository
```

### `caid run`

```
Usage: caid run [OPTIONS]

Options:
  --repo PATH              Path to the target git repository [required]
  --mode [commit0|paperbench]  Operating mode (default: commit0)
  --engineers INTEGER      Number of engineer agents (default: 4)
  --config PATH            Path to caid.yaml configuration file
  --task-description TEXT  Task description or path to paper file
  --help                   Show help message
```

### `caid graph`

```
Usage: caid graph [OPTIONS]

Options:
  --repo PATH    Path to the target git repository [required]
  --help         Show help message
```

---

## 10. Cost Warning

CAID uses multiple LLM agents working in parallel. The paper reports that
CAID uses approximately **4-5x more LLM tokens** than a single-agent
approach. This is by design -- the parallelism and specialization improve
code quality and completion rate, but at higher cost.

For example, with 4 engineers each running up to 80 iterations:
- A single Commit0 run may consume 500K-2M tokens depending on repository size.
- PaperBench runs can consume even more due to the decomposition step.

Monitor your LLM provider dashboard and set billing limits accordingly.
The result summary includes token usage:

```python
result = run_caid_sync(...)
print(f"Prompt tokens: {result['token_usage']['prompt']}")
print(f"Completion tokens: {result['token_usage']['completion']}")
```

---

## 11. Troubleshooting

### "git worktree" fails

Ensure git 2.5+ is installed. Check with `git --version`. On older systems,
update git:

```bash
# macOS
brew install git

# Ubuntu/Debian
sudo apt-get install git
```

### "No tasks found" in Commit0 mode

CAID only detects stub functions with bodies of `pass` or `...`. Verify your
repository has functions matching this pattern:

```python
def my_function():
    pass

def another_function():
    """Docstring."""
    ...
```

### API key errors

Verify your key is set correctly:

```bash
echo $CAID_LLM_API_KEY
# Should print your key (non-empty)
```

If using OpenAI, also check `OPENAI_API_KEY`. For Anthropic, check
`ANTHROPIC_API_KEY`.

### pytest not found in engineer worktrees

Engineers run pytest in their isolated worktrees. Ensure pytest is installed
in the Python environment that CAID uses:

```bash
pip install pytest
```

### Merge conflicts that do not resolve

If CAID's automatic conflict resolution fails repeatedly, try reducing the
number of engineers (`--engineers 2`) to minimize concurrent modifications
to overlapping files.

### High memory usage

For very large repositories, reduce `max_engineers` to limit the number of
concurrent git worktrees. Each worktree is a full copy of the repository's
working tree.

### Tests fail after CAID completes

CAID commits partial implementations when the engineer iteration limit is
reached. Run the tests manually and review the git log to identify incomplete
tasks:

```bash
git log --oneline --graph
python -m pytest -v
```

---

## 12. Architecture Overview

CAID follows a centralized asynchronous architecture:

```
Manager Agent
  |
  |-- Builds dependency graph (import analysis or LLM decomposition)
  |-- Creates isolated git worktrees for each engineer
  |-- Delegates tasks based on graph readiness (Eq. 3-4)
  |
  +-- Engineer 0 (worktree 0)
  |     |-- Receives task spec
  |     |-- Implements code via LLM
  |     |-- Runs pytest (self-verification)
  |     |-- Iterates on failures (up to 80 iterations)
  |     +-- Commits to branch
  |
  +-- Engineer 1 (worktree 1)
  |     +-- (same as above, different task)
  |
  +-- Engineer N ...
  |
  |-- Merges completed branches (handles conflicts via LLM)
  |-- Updates graph (marks completed, unlocks dependents)
  |-- Condenses context periodically
  +-- Final review pass
```

Key design principles from the CAID paper:

- **Isolation**: Each engineer works in a separate git worktree to prevent
  interference between concurrent implementations.
- **Dependency ordering**: Tasks are assigned based on a DAG. A task is only
  assigned when all its prerequisites are completed (Eq. 3).
- **Async first-completed**: The manager uses `asyncio.wait(FIRST_COMPLETED)`
  to process results as they arrive, maximizing parallelism.
- **Self-verification**: Engineers run pytest after each implementation
  attempt and iterate on failures before committing.

---

## Running Tests

```bash
# Run the full test suite
python -m pytest tests/ -v

# Run a specific test file
python -m pytest tests/test_graph.py -v

# Run with coverage
pip install pytest-cov
python -m pytest tests/ --cov=caid --cov-report=term-missing
```
