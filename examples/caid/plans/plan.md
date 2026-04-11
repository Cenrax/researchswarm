# CAID Framework -- Implementation Plan

## 1. Objective Restatement

Build a Python framework implementing the CAID (Centralized Asynchronous Isolated Delegation) multi-agent coordination system described in arXiv:2603.21489v1. The system consists of a centralized Manager agent that constructs a directed dependency graph from a target repository or research paper, delegates tasks as structured JSON to N concurrent Engineer agents, each operating in an isolated `git worktree`, and integrates their work via `git merge` on an `asyncio` event loop. The framework must support two modes: **Commit0** (implementing stub functions in existing Python repos, with import-based dependency analysis and test-to-file mapping) and **PaperBench** (implementing research papers from scratch, with LLM-inferred task decomposition). It must be configurable for LLM provider, model, API keys, engineer count, and iteration limits, and expose a CLI entry point and a Python API for integration with external agent systems.

---

## 2. Architecture Overview

### 2.1 High-Level Component Diagram

```
                            +-------------------+
                            |    CLI / API      |
                            | (click CLI +      |
                            |  Python callable) |
                            +--------+----------+
                                     |
                                     v
                            +-------------------+
                            |   Config Loader   |
                            | (YAML + env vars) |
                            +--------+----------+
                                     |
                                     v
+-------------------+       +-------------------+       +-------------------+
| Commit0 Dep.      |       |   MANAGER AGENT   |       | PaperBench Task   |
| Extractor         |<----->|                   |<----->| Decomposer        |
| (import analysis) |       | - explore repo    |       | (LLM paper read)  |
+-------------------+       | - build DAG       |       +-------------------+
                            | - delegate tasks  |
+-------------------+       | - await engineers |       +-------------------+
| Test-to-File      |<----->| - git merge       |<----->| LLM Summarizing   |
| Mapper            |       | - reassign        |       | Condenser         |
+-------------------+       | - final review    |       +-------------------+
                            +--------+----------+
                                     |
                      asyncio.wait(FIRST_COMPLETED)
                        /            |            \
                       v             v             v
              +-----------+  +-----------+  +-----------+
              | ENGINEER  |  | ENGINEER  |  | ENGINEER  |
              | Agent 1   |  | Agent 2   |  | Agent N   |
              |           |  |           |  |           |
              | worktree/ |  | worktree/ |  | worktree/ |
              | eng-1     |  | eng-2     |  | eng-N     |
              +-----------+  +-----------+  +-----------+
                    |              |              |
                    v              v              v
              +-----------------------------------------+
              |          GIT INFRASTRUCTURE             |
              | - worktree_add / worktree_remove        |
              | - commit / merge / reset                |
              | - conflict detection                    |
              +-----------------------------------------+
                                  |
                                  v
                         +----------------+
                         | Dependency     |
                         | Graph Engine   |
                         | (networkx)     |
                         +----------------+
```

### 2.2 Data Flow

```
Input (repo path + task description OR paper PDF)
  |
  v
[Config Loader] --> CAIDConfig (Pydantic model)
  |
  v
[Manager: explore_repository()] --> file listing, stub detection
  |
  v
[Dependency Extractor OR Paper Decomposer] --> nx.DiGraph G = (V, E)
  |
  v
[Manager: preparatory_commit()] --> stubs committed to main
  |
  v
[Manager: create_worktrees()] --> N worktree directories
  |
  v
[Manager: get_assignable_tasks()] --> list[TaskSpec] (JSON)
  |
  v
[asyncio: launch engineer coroutines] --> each receives TaskSpec
  |
  v
[Engineer: implement + test + commit] --> git commit in worktree branch
  |
  v
[Manager: await + git merge] --> merge into main, update C_t
  |
  v
[Manager: condense context] --> compressed history
  |
  v
[Loop until C_t == V or limits exhausted]
  |
  v
[Manager: final_review()] --> submission
```

### 2.3 Module Breakdown

| Module | File | Purpose |
|--------|------|---------|
| `config` | `caid/config.py` | Pydantic config models, YAML loader |
| `schemas` | `caid/schemas.py` | All JSON message schemas (Pydantic) |
| `graph` | `caid/graph.py` | Dependency graph, readiness checker, assignable set |
| `git_ops` | `caid/git_ops.py` | Git worktree, merge, commit, reset wrappers |
| `llm` | `caid/llm.py` | LLM call abstraction via litellm |
| `condenser` | `caid/condenser.py` | LLMSummarizingCondenser |
| `manager` | `caid/manager.py` | Manager agent asyncio loop |
| `engineer` | `caid/engineer.py` | Engineer agent coroutine |
| `commit0` | `caid/commit0/extractor.py` | Import-based dependency extractor |
| `commit0` | `caid/commit0/test_mapper.py` | Test-to-file mapper |
| `paperbench` | `caid/paperbench/decomposer.py` | Paper-reading task decomposer |
| `cli` | `caid/cli.py` | Click CLI entry point |
| `api` | `caid/api.py` | Programmatic Python API |

---

## 3. Dependencies

See [dependencies.md](./dependencies.md) for the full dependency table.

Summary of required packages:
- `networkx>=3.1` -- dependency graph
- `pydantic>=2.0` -- JSON schemas
- `litellm>=1.40` -- LLM calls (multi-provider)
- `click>=8.1` -- CLI
- `pyyaml>=6.0` -- config files
- `gitpython>=3.1.40` -- git operations (supplemented by subprocess for worktree commands)

---

## 4. Step-by-Step Implementation Plan

### Step 1: Project Scaffolding and Configuration

**What**: Create the project directory structure, `pyproject.toml`, and the configuration system.

**Why**: Every other component depends on a shared config (LLM API keys, model name, iteration limits, engineer count). The paper specifies exact default values (Section 3, Table in overview).

**Files**:
- `caid/__init__.py`
- `caid/config.py`
- `pyproject.toml`
- `caid.yaml.example`

**Acceptance Criteria**:
- `CAIDConfig.from_yaml("caid.yaml")` loads a config.
- `CAIDConfig()` with no args produces valid defaults matching the paper.
- Config can be overridden by environment variables (e.g., `CAID_LLM_API_KEY`).

**Code Skeleton**:
```python
# caid/config.py
from pydantic import BaseModel, Field
from pathlib import Path
import yaml, os

class LLMConfig(BaseModel):
    provider: str = "anthropic"          # or "openai", "minimax", etc.
    model: str = "claude-sonnet-4-5-20260120"
    api_key: str = Field(default_factory=lambda: os.environ.get("CAID_LLM_API_KEY", ""))
    temperature: float = 0.0
    max_tokens: int = 4096

class CAIDConfig(BaseModel):
    mode: str = "commit0"                # "commit0" or "paperbench"
    repo_path: Path = Path(".")
    task_description: str = ""
    max_engineers: int = 4               # 4 for Commit0, 2 for PaperBench
    manager_max_iterations: int = 50
    engineer_max_iterations: int = 80
    implementation_rounds: int = 2
    restricted_files: list[str] = Field(default_factory=lambda: ["__init__.py"])
    llm: LLMConfig = Field(default_factory=LLMConfig)
    worktree_base_dir: Path = Path("/tmp/caid-worktrees")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "CAIDConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
```

---

### Step 2: Pydantic JSON Schemas

**What**: Define all manager-to-engineer and engineer-to-manager message schemas as Pydantic models.

**Why**: The paper provides exact JSON schemas for Commit0 delegation, Commit0 reassignment, PaperBench delegation, and PaperBench reassignment (CAID Appendix, Section 4 of summary). Structured communication is a core CAID principle.

**Files**:
- `caid/schemas.py`

**Acceptance Criteria**:
- Each schema can be serialized to JSON and deserialized back losslessly.
- Invalid messages (missing required fields, wrong enum values) raise `ValidationError`.
- Round-trip test: `Model.model_validate_json(model.model_dump_json())` succeeds.

**Code Skeleton**:
```python
# caid/schemas.py
from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional

class Complexity(str, Enum):
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"

class TaskCategory(str, Enum):
    CODE_DEVELOPMENT = "Code Development"
    EXPERIMENT_RUNNING = "Experiment Running"
    RESULTS_ANALYSIS = "Results Analysis"
    OTHER = "Other"

# --- Commit0 Schemas ---

class Commit0Task(BaseModel):
    engineer_id: str
    task_id: str
    file_path: str
    functions_to_implement: list[str]
    complexity: Complexity
    instruction: str

class Commit0RemainingTask(BaseModel):
    task_id: str
    file_path: str
    functions_to_implement: list[str]
    complexity: Complexity
    depends_on: list[str] = Field(default_factory=list)

class Commit0DelegationPlan(BaseModel):
    first_round: "Commit0FirstRound"
    remaining_tasks: list[Commit0RemainingTask]

class Commit0FirstRound(BaseModel):
    num_agents: int
    reasoning: str
    tasks: list[Commit0Task]

class Commit0Delegation(BaseModel):
    delegation_plan: Commit0DelegationPlan

class Commit0Assignment(BaseModel):
    engineer_id: str
    task_id: str
    file_path: str
    functions_to_implement: list[str]
    instruction: str
    complexity: Complexity

class Commit0Reassignment(BaseModel):
    assign_task: "Commit0AssignTask"

class Commit0AssignTask(BaseModel):
    reasoning: str
    assignments: list[Commit0Assignment]

# --- PaperBench Schemas ---

class PaperBenchTask(BaseModel):
    engineer_id: str
    task_id: str
    task_node_id: Optional[str] = None
    requirements: str
    task_category: TaskCategory
    estimated_complexity: Complexity
    instruction: str

class PaperBenchRemainingTask(BaseModel):
    task_id: str
    task_node_id: Optional[str] = None
    requirements: str
    task_category: TaskCategory
    estimated_complexity: Complexity
    depends_on: list[str] = Field(default_factory=list)

class PaperBenchDelegationPlan(BaseModel):
    first_round: "PaperBenchFirstRound"
    remaining_tasks: list[PaperBenchRemainingTask]

class PaperBenchFirstRound(BaseModel):
    num_agents: int
    reasoning: str
    tasks: list[PaperBenchTask]

class PaperBenchDelegation(BaseModel):
    delegation_plan: PaperBenchDelegationPlan

class PaperBenchReassignment(BaseModel):
    assign_task: "PaperBenchAssignTaskBlock"

class PaperBenchAssignTaskBlock(BaseModel):
    reasoning: str
    tasks: list[PaperBenchTask]

# --- Engineer Result ---

class EngineerResult(BaseModel):
    engineer_id: str
    task_id: str
    status: str              # "completed", "partial", "conflict"
    branch_name: str
    commit_sha: Optional[str] = None
    test_passed: bool = False
    error_log: Optional[str] = None
```

---

### Step 3: Dependency Graph Engine

**What**: Build the dependency graph data structure with readiness checking and assignable task selection.

**Why**: Implements Equations 1-4 from the paper (Section 2.1). The graph is the central coordination structure -- every delegation decision depends on it.

**Files**:
- `caid/graph.py`
- `tests/test_graph.py`

**Acceptance Criteria**:
- Can construct a DiGraph from a list of nodes and edges.
- `is_ready(node)` returns True only when all predecessors are completed.
- `get_assignable_tasks()` returns the correct set (ready AND not completed AND not in progress).
- Topological sort produces a valid ordering.
- Handles nodes with no dependencies (immediately ready).
- Handles cycles gracefully (raises error at construction time).

**Code Skeleton**:
```python
# caid/graph.py
import networkx as nx
from dataclasses import dataclass, field

@dataclass
class TaskNode:
    task_id: str
    file_path: str = ""
    functions: list[str] = field(default_factory=list)
    complexity: str = "medium"
    metadata: dict = field(default_factory=dict)

class DependencyGraph:
    def __init__(self) -> None:
        self._graph: nx.DiGraph = nx.DiGraph()
        self._completed: set[str] = set()
        self._in_progress: set[str] = set()

    def add_task(self, node: TaskNode) -> None:
        self._graph.add_node(node.task_id, data=node)

    def add_dependency(self, prerequisite_id: str, dependent_id: str) -> None:
        """Add edge: dependent depends on prerequisite."""
        self._graph.add_edge(prerequisite_id, dependent_id)
        if not nx.is_directed_acyclic_graph(self._graph):
            self._graph.remove_edge(prerequisite_id, dependent_id)
            raise ValueError(f"Adding edge {prerequisite_id}->{dependent_id} creates a cycle")

    def is_ready(self, task_id: str) -> bool:
        """Eq. 3: Ready_t(v_j) iff all predecessors in completed set."""
        return all(
            pred in self._completed
            for pred in self._graph.predecessors(task_id)
        )

    def get_assignable_tasks(self) -> list[str]:
        """Eq. 4: tasks that are ready, not completed, not in progress."""
        candidates = set(self._graph.nodes) - self._completed - self._in_progress
        assignable = [v for v in candidates if self.is_ready(v)]
        # Prioritize upstream tasks (fewer successors depend on them completing)
        assignable.sort(key=lambda v: -len(list(self._graph.successors(v))))
        return assignable

    def mark_completed(self, task_id: str) -> None:
        self._in_progress.discard(task_id)
        self._completed.add(task_id)

    def mark_in_progress(self, task_id: str) -> None:
        self._in_progress.add(task_id)

    def mark_failed(self, task_id: str) -> None:
        """Return task to assignable pool."""
        self._in_progress.discard(task_id)

    def is_done(self) -> bool:
        """Eq. 5 (partial): C_t == V."""
        return self._completed == set(self._graph.nodes)

    def get_node_data(self, task_id: str) -> TaskNode:
        return self._graph.nodes[task_id]["data"]

    @property
    def completed(self) -> set[str]:
        return self._completed.copy()

    @property
    def all_tasks(self) -> set[str]:
        return set(self._graph.nodes)

    def topological_order(self) -> list[str]:
        return list(nx.topological_sort(self._graph))
```

---

### Step 4: Git Operations Module

**What**: Implement wrappers for all git primitives used by CAID: worktree add/remove, branch creation, commit, merge with conflict detection, reset, and pull.

**Why**: Git worktree isolation is one of the three core SWE primitives in CAID (Section 2.3). Every engineer operates in a separate worktree, and integration happens via git merge.

**Files**:
- `caid/git_ops.py`
- `tests/test_git_ops.py`

**Acceptance Criteria**:
- `worktree_add()` creates a new worktree directory with a new branch.
- `worktree_remove()` cleans up the worktree.
- `commit()` stages and commits all changes in a worktree.
- `merge()` returns a `MergeResult` with conflict status and details.
- `reset_hard()` syncs a worktree to latest main.
- All operations raise clear exceptions on failure.
- Tests use a temporary git repo (created in `tmp`).

**Code Skeleton**:
```python
# caid/git_ops.py
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)

@dataclass
class MergeResult:
    success: bool
    conflict_files: list[str]
    commit_sha: Optional[str] = None
    error_message: Optional[str] = None

class GitOps:
    def __init__(self, repo_path: Path) -> None:
        self.repo_path = Path(repo_path).resolve()

    def _run(self, args: list[str], cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
        cwd = cwd or self.repo_path
        result = subprocess.run(
            ["git"] + args,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            logger.warning(f"git {' '.join(args)} failed: {result.stderr}")
        return result

    def worktree_add(self, worktree_path: Path, branch_name: str) -> Path:
        """Create a new worktree with a new branch based on current HEAD."""
        worktree_path = Path(worktree_path).resolve()
        self._run(["worktree", "add", "-b", branch_name, str(worktree_path)])
        return worktree_path

    def worktree_remove(self, worktree_path: Path) -> None:
        """Remove a worktree and its branch."""
        self._run(["worktree", "remove", str(worktree_path), "--force"])

    def worktree_list(self) -> list[str]:
        result = self._run(["worktree", "list", "--porcelain"])
        paths = []
        for line in result.stdout.splitlines():
            if line.startswith("worktree "):
                paths.append(line.split(" ", 1)[1])
        return paths

    def commit_all(self, message: str, cwd: Optional[Path] = None) -> Optional[str]:
        """Stage all changes and commit. Returns commit SHA or None if nothing to commit."""
        work_dir = cwd or self.repo_path
        self._run(["add", "-A"], cwd=work_dir)
        result = self._run(["commit", "-m", message, "--allow-empty"], cwd=work_dir)
        if result.returncode == 0:
            sha = self._run(["rev-parse", "HEAD"], cwd=work_dir)
            return sha.stdout.strip()
        return None

    def merge_branch(self, branch_name: str, target_branch: str = "main") -> MergeResult:
        """Merge branch into target. Returns MergeResult with conflict info."""
        self._run(["checkout", target_branch])
        result = self._run(["merge", branch_name, "--no-ff", "-m", f"Merge {branch_name}"])
        if result.returncode == 0:
            sha = self._run(["rev-parse", "HEAD"]).stdout.strip()
            return MergeResult(success=True, conflict_files=[], commit_sha=sha)
        else:
            # Detect conflict files
            status = self._run(["diff", "--name-only", "--diff-filter=U"])
            conflicts = [f.strip() for f in status.stdout.splitlines() if f.strip()]
            return MergeResult(
                success=False,
                conflict_files=conflicts,
                error_message=result.stderr,
            )

    def abort_merge(self) -> None:
        self._run(["merge", "--abort"])

    def reset_hard(self, ref: str = "HEAD", cwd: Optional[Path] = None) -> None:
        self._run(["reset", "--hard", ref], cwd=cwd or self.repo_path)

    def pull_main(self, cwd: Path) -> subprocess.CompletedProcess:
        """Pull latest main into a worktree (for conflict resolution)."""
        return self._run(["pull", "origin", "main", "--rebase=false"], cwd=cwd)

    def get_current_branch(self, cwd: Optional[Path] = None) -> str:
        result = self._run(["branch", "--show-current"], cwd=cwd or self.repo_path)
        return result.stdout.strip()

    def checkout(self, branch: str, cwd: Optional[Path] = None) -> None:
        self._run(["checkout", branch], cwd=cwd or self.repo_path)
```

---

### Step 5: LLM Abstraction Layer

**What**: Build a thin wrapper around `litellm` for making LLM calls with structured JSON output, retries, and token tracking.

**Why**: Both the Manager and Engineer agents need to call LLMs. The paper tests multiple models (Claude Sonnet 4.5, GLM 4.7, MiniMax 2.5). litellm provides a unified interface. The wrapper adds JSON parsing, retry logic, and token usage tracking.

**Files**:
- `caid/llm.py`
- `tests/test_llm.py`

**Acceptance Criteria**:
- `llm_call()` returns parsed JSON when given a JSON schema.
- `llm_call()` returns plain text when no schema is given.
- Retries up to 3 times on transient errors (rate limit, timeout).
- Tracks cumulative token usage.
- Works with at least OpenAI and Anthropic providers.

**Code Skeleton**:
```python
# caid/llm.py
import json
import litellm
from pydantic import BaseModel
from typing import Optional, Type, TypeVar
from caid.config import LLMConfig
import logging

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

class LLMClient:
    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0

    async def call(
        self,
        messages: list[dict[str, str]],
        response_model: Optional[Type[T]] = None,
        max_retries: int = 3,
    ) -> str | T:
        """Make an LLM call. If response_model is provided, parse and validate JSON output."""
        for attempt in range(max_retries):
            try:
                response = await litellm.acompletion(
                    model=self.config.model,
                    messages=messages,
                    api_key=self.config.api_key,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                )
                content = response.choices[0].message.content
                usage = response.usage
                self.total_prompt_tokens += usage.prompt_tokens
                self.total_completion_tokens += usage.completion_tokens

                if response_model is not None:
                    # Extract JSON from response (handle markdown code blocks)
                    json_str = self._extract_json(content)
                    return response_model.model_validate_json(json_str)
                return content

            except (litellm.RateLimitError, litellm.Timeout) as e:
                if attempt == max_retries - 1:
                    raise
                logger.warning(f"LLM call attempt {attempt+1} failed: {e}")
                import asyncio
                await asyncio.sleep(2 ** attempt)

    @staticmethod
    def _extract_json(text: str) -> str:
        """Extract JSON from text that may contain markdown code blocks."""
        if "```json" in text:
            start = text.index("```json") + 7
            end = text.index("```", start)
            return text[start:end].strip()
        if "```" in text:
            start = text.index("```") + 3
            end = text.index("```", start)
            return text[start:end].strip()
        return text.strip()
```

---

### Step 6: LLM Summarizing Condenser

**What**: Implement context compression for the Manager agent to prevent context window overflow during long runs.

**Why**: The paper identifies context growth as a limitation (Section 7) and uses `LLMSummarizingCondenser` as mitigation. The manager's history of delegations, merge results, and error logs can grow beyond the context window over 50 iterations.

**Files**:
- `caid/condenser.py`
- `tests/test_condenser.py`

**Acceptance Criteria**:
- Given a list of messages, produces a shorter list preserving: dependency graph state, completed tasks, in-progress tasks, and unresolved errors.
- Output token count is at most 50% of input token count.
- Preserved information can be verified by checking that key task IDs and graph state appear in the summary.

**Code Skeleton**:
```python
# caid/condenser.py
from caid.llm import LLMClient
from caid.graph import DependencyGraph
from typing import Any

CONDENSER_SYSTEM_PROMPT = """You are a context condenser for a multi-agent software engineering system.
Given a conversation history, produce a concise summary that preserves:
1. The current dependency graph state (which tasks are completed, in-progress, and remaining)
2. Any unresolved errors or merge conflicts
3. Key decisions made and their rationale
4. The current implementation round number

Do NOT preserve: verbose code listings, redundant status updates, or resolved issues.
Output a structured summary in the same conversational format."""

class LLMSummarizingCondenser:
    def __init__(self, llm_client: LLMClient) -> None:
        self.llm = llm_client

    async def condense(
        self,
        messages: list[dict[str, str]],
        graph: DependencyGraph,
        preserve_keys: dict[str, Any] | None = None,
    ) -> list[dict[str, str]]:
        """Compress message history while preserving critical state."""
        if len(messages) <= 4:
            return messages  # Too short to compress

        # Build state summary header
        state_summary = self._build_state_header(graph, preserve_keys)

        # Ask LLM to condense the history
        condense_messages = [
            {"role": "system", "content": CONDENSER_SYSTEM_PROMPT},
            {"role": "user", "content": f"Current state:\n{state_summary}\n\nHistory to condense:\n{self._format_messages(messages)}"},
        ]
        summary = await self.llm.call(condense_messages)

        # Return condensed history: system prompt + state + summary
        return [
            messages[0],  # Keep original system prompt
            {"role": "assistant", "content": f"[Condensed History]\n{state_summary}\n\n{summary}"},
        ]

    def _build_state_header(self, graph: DependencyGraph, extras: dict | None) -> str:
        completed = graph.completed
        assignable = graph.get_assignable_tasks()
        remaining = graph.all_tasks - completed
        lines = [
            f"Completed tasks ({len(completed)}): {sorted(completed)}",
            f"Assignable tasks ({len(assignable)}): {assignable}",
            f"Remaining tasks ({len(remaining)}): {sorted(remaining)}",
        ]
        if extras:
            for k, v in extras.items():
                lines.append(f"{k}: {v}")
        return "\n".join(lines)

    @staticmethod
    def _format_messages(messages: list[dict]) -> str:
        parts = []
        for m in messages:
            parts.append(f"[{m['role']}]: {m['content'][:500]}")
        return "\n---\n".join(parts)
```

---

### Step 7: Engineer Agent Coroutine

**What**: Implement the Engineer agent as an async coroutine that receives a task spec, implements code in its worktree, runs tests, iterates on failures, and commits.

**Why**: Directly implements the engineer lifecycle from CAID Section 2.4-2.5 and the `engineer_loop` pseudocode from the equations file.

**Files**:
- `caid/engineer.py`
- `tests/test_engineer.py`

**Acceptance Criteria**:
- Receives a task spec (Pydantic model) and worktree path.
- Calls the LLM to generate implementation code.
- Writes generated code to the correct file(s) in the worktree.
- Runs the relevant test subset via `pytest` or similar.
- On test failure, feeds error output back to LLM and retries (up to `max_iterations`).
- On success (or iteration exhaustion), commits and returns `EngineerResult`.
- Does NOT modify restricted files.

**Code Skeleton**:
```python
# caid/engineer.py
import asyncio
import subprocess
from pathlib import Path
from caid.llm import LLMClient
from caid.schemas import EngineerResult, Commit0Task, PaperBenchTask
from caid.git_ops import GitOps
from caid.config import CAIDConfig
from typing import Union
import logging

logger = logging.getLogger(__name__)

ENGINEER_SYSTEM_PROMPT = """You are a software engineer agent. You receive a task specification
and must implement the required code changes. You operate in an isolated git worktree.

Rules:
- Only modify the files specified in your task.
- Do NOT modify any restricted files ({restricted_files}).
- Write clean, well-documented code.
- After implementation, tests will be run. If they fail, you will receive the error
  output and must fix the issues.
- When done, your changes will be committed automatically."""

class EngineerAgent:
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

    async def run(
        self,
        task: Union[Commit0Task, PaperBenchTask],
        worktree_path: Path,
        relevant_tests: list[str] | None = None,
    ) -> EngineerResult:
        """Main engineer coroutine. Implements, tests, fixes, commits."""
        branch_name = f"engineer-{self.engineer_id}-{task.task_id}"
        messages = [
            {"role": "system", "content": ENGINEER_SYSTEM_PROMPT.format(
                restricted_files=self.config.restricted_files
            )},
            {"role": "user", "content": self._format_task_prompt(task, worktree_path)},
        ]

        for iteration in range(self.config.engineer_max_iterations):
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
                return EngineerResult(
                    engineer_id=self.engineer_id,
                    task_id=task.task_id,
                    status="completed",
                    branch_name=branch_name,
                    commit_sha=sha,
                    test_passed=True,
                )

            # Step 4: Feed errors back to LLM
            error_msg = f"Tests failed. Output:\n{test_result['output']}\nFix the issues."
            messages.append({"role": "user", "content": error_msg})
            logger.info(f"Engineer {self.engineer_id}: iteration {iteration+1}, tests failed")

        # Iteration limit -- partial commit
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

    def _format_task_prompt(self, task: Union[Commit0Task, PaperBenchTask], worktree_path: Path) -> str:
        """Build the implementation prompt from the task spec."""
        if isinstance(task, Commit0Task):
            # Read the target file content
            target = worktree_path / task.file_path
            existing = target.read_text() if target.exists() else "(file does not exist)"
            return (
                f"Task: {task.task_id}\n"
                f"File: {task.file_path}\n"
                f"Functions to implement: {task.functions_to_implement}\n"
                f"Complexity: {task.complexity}\n"
                f"Instruction: {task.instruction}\n\n"
                f"Current file content:\n```python\n{existing}\n```\n\n"
                f"Implement the specified functions. Return the COMPLETE updated file content "
                f"wrapped in ```python ... ``` code blocks."
            )
        else:
            # PaperBench task
            return (
                f"Task: {task.task_id}\n"
                f"Category: {task.task_category}\n"
                f"Requirements: {task.requirements}\n"
                f"Complexity: {task.estimated_complexity}\n"
                f"Instruction: {task.instruction}\n\n"
                f"Implement this task. Return any code wrapped in ```python ... ``` blocks "
                f"with the target filename indicated."
            )

    def _apply_code_changes(
        self,
        llm_response: str,
        worktree_path: Path,
        task: Union[Commit0Task, PaperBenchTask],
    ) -> None:
        """Extract code blocks from LLM response and write to files."""
        # Simple extraction: find ```python ... ``` blocks
        import re
        blocks = re.findall(r"```python\n(.*?)```", llm_response, re.DOTALL)
        if blocks and isinstance(task, Commit0Task):
            target = worktree_path / task.file_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(blocks[0])
        elif blocks:
            # PaperBench: look for filename hints or write to default location
            for block in blocks:
                # Try to find # filename: ... at the top of the block
                lines = block.split("\n")
                if lines[0].startswith("# filename:"):
                    fname = lines[0].split(":", 1)[1].strip()
                    target = worktree_path / fname
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text("\n".join(lines[1:]))

    async def _run_tests(
        self,
        worktree_path: Path,
        relevant_tests: list[str] | None,
    ) -> dict:
        """Run pytest on relevant tests in the worktree."""
        cmd = ["python", "-m", "pytest", "-x", "--tb=short", "-q"]
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
            return {"passed": passed, "output": output[-3000:]}  # Truncate
        except subprocess.TimeoutExpired:
            return {"passed": False, "output": "Test execution timed out (300s)"}

    async def resolve_conflict(self, worktree_path: Path, conflict_files: list[str]) -> EngineerResult:
        """Handle merge conflict re-routing from manager."""
        # Pull latest main
        self.git.pull_main(worktree_path)

        # Read conflict markers
        conflict_contents = {}
        for f in conflict_files:
            fpath = worktree_path / f
            if fpath.exists():
                conflict_contents[f] = fpath.read_text()

        # Ask LLM to resolve
        messages = [
            {"role": "system", "content": "You are resolving a git merge conflict. "
             "For each file, output the resolved content (no conflict markers)."},
            {"role": "user", "content": f"Conflict files:\n{conflict_contents}"},
        ]
        response = await self.llm.call(messages)

        # Apply resolutions and commit
        # (parse response for file contents, write them)
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
```

---

### Step 8: Manager Agent Event Loop

**What**: Implement the central Manager agent that builds the dependency graph, delegates tasks, runs the asyncio event loop, handles merges, and coordinates the full lifecycle.

**Why**: This is the core orchestrator described in CAID Section 2.2 and the pseudocode in the equations file. It implements the full manager event loop.

**Files**:
- `caid/manager.py`
- `tests/test_manager.py`

**Acceptance Criteria**:
- Constructs the dependency graph (via Commit0 extractor or PaperBench decomposer).
- Creates worktrees for N engineers.
- Delegates initial batch of ready tasks as JSON.
- Launches engineer coroutines with `asyncio.create_task()`.
- Uses `asyncio.wait(FIRST_COMPLETED)` to process results as they arrive.
- Performs `git merge` after each engineer completes.
- Routes merge conflicts back to the engineer.
- Updates dependency graph state and reassigns tasks.
- Calls condenser periodically.
- Terminates when `C_t == V` or limits exhausted.
- Performs final review pass.

**Code Skeleton**:
```python
# caid/manager.py
import asyncio
from pathlib import Path
from caid.config import CAIDConfig
from caid.graph import DependencyGraph, TaskNode
from caid.git_ops import GitOps, MergeResult
from caid.llm import LLMClient
from caid.engineer import EngineerAgent
from caid.condenser import LLMSummarizingCondenser
from caid.schemas import (
    Commit0Delegation, Commit0Reassignment,
    PaperBenchDelegation, PaperBenchReassignment,
    EngineerResult,
)
from typing import Optional
import logging

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
    def __init__(self, config: CAIDConfig) -> None:
        self.config = config
        self.llm = LLMClient(config.llm)
        self.git = GitOps(config.repo_path)
        self.graph: Optional[DependencyGraph] = None
        self.condenser = LLMSummarizingCondenser(self.llm)
        self.engineers: dict[str, EngineerAgent] = {}
        self.worktree_paths: dict[str, Path] = {}
        self.messages: list[dict[str, str]] = []

    async def run(self) -> dict:
        """Main manager event loop."""
        # Phase 1: Explore and build dependency graph
        self.graph = await self._build_dependency_graph()
        logger.info(f"Dependency graph built: {len(self.graph.all_tasks)} tasks")

        # Phase 2: Preparatory commit (stubs, structure)
        await self._preparatory_commit()

        # Phase 3: Setup engineers and worktrees
        self._setup_engineers()

        # Phase 4: Main delegation loop
        for impl_round in range(self.config.implementation_rounds):
            logger.info(f"=== Implementation round {impl_round + 1} ===")
            await self._delegation_loop()

        # Phase 5: Final review
        await self._final_review()

        # Phase 6: Cleanup
        self._cleanup_worktrees()

        return {
            "completed_tasks": len(self.graph.completed),
            "total_tasks": len(self.graph.all_tasks),
            "token_usage": {
                "prompt": self.llm.total_prompt_tokens,
                "completion": self.llm.total_completion_tokens,
            },
        }

    async def _build_dependency_graph(self) -> DependencyGraph:
        """Build the dependency graph based on mode."""
        if self.config.mode == "commit0":
            from caid.commit0.extractor import extract_dependencies
            return extract_dependencies(self.config.repo_path)
        else:
            from caid.paperbench.decomposer import decompose_paper
            return await decompose_paper(self.config, self.llm)

    async def _preparatory_commit(self) -> None:
        """Commit any stubs or structural changes to main before delegation."""
        self.git.commit_all("CAID: preparatory commit (stubs)")

    def _setup_engineers(self) -> None:
        """Create worktrees and engineer agent instances."""
        for i in range(self.config.max_engineers):
            eng_id = f"eng-{i}"
            worktree_path = self.config.worktree_base_dir / eng_id
            branch_name = f"caid/{eng_id}"
            self.git.worktree_add(worktree_path, branch_name)
            self.worktree_paths[eng_id] = worktree_path
            self.engineers[eng_id] = EngineerAgent(
                engineer_id=eng_id,
                config=self.config,
                llm_client=LLMClient(self.config.llm),  # Each engineer gets own client
                git_ops=GitOps(worktree_path),
            )

    async def _delegation_loop(self) -> None:
        """Core asyncio event loop for task delegation and integration."""
        running_tasks: dict[str, asyncio.Task] = {}  # engineer_id -> asyncio.Task
        task_assignments: dict[str, str] = {}         # engineer_id -> task_id

        iteration = 0
        while iteration < self.config.manager_max_iterations:
            # Get assignable tasks
            assignable = self.graph.get_assignable_tasks()

            if not assignable and not running_tasks:
                logger.info("All tasks completed or no more assignable tasks.")
                break

            # Assign to idle engineers
            idle_engineers = [
                eid for eid in self.engineers
                if eid not in running_tasks
            ]
            for eng_id, task_id in zip(idle_engineers, assignable):
                task_node = self.graph.get_node_data(task_id)
                task_spec = self._build_task_spec(eng_id, task_node)
                self.graph.mark_in_progress(task_id)
                task_assignments[eng_id] = task_id

                # Resolve relevant tests for Commit0
                relevant_tests = None
                if self.config.mode == "commit0":
                    from caid.commit0.test_mapper import get_relevant_tests
                    relevant_tests = get_relevant_tests(
                        self.config.repo_path, task_node.file_path
                    )

                coro = self.engineers[eng_id].run(
                    task=task_spec,
                    worktree_path=self.worktree_paths[eng_id],
                    relevant_tests=relevant_tests,
                )
                running_tasks[eng_id] = asyncio.create_task(coro)
                logger.info(f"Assigned {task_id} to {eng_id}")

            if not running_tasks:
                break

            # Await first completion
            done, _ = await asyncio.wait(
                running_tasks.values(),
                return_when=asyncio.FIRST_COMPLETED,
            )

            for finished in done:
                result: EngineerResult = finished.result()
                eng_id = result.engineer_id
                task_id = task_assignments.pop(eng_id, result.task_id)

                # Merge engineer branch into main
                merge_result = self.git.merge_branch(result.branch_name)

                if merge_result.success:
                    self.graph.mark_completed(task_id)
                    logger.info(f"Merged {task_id} from {eng_id}")
                else:
                    # Conflict: route back to engineer
                    logger.warning(f"Merge conflict for {task_id}: {merge_result.conflict_files}")
                    self.git.abort_merge()
                    conflict_result = await self.engineers[eng_id].resolve_conflict(
                        self.worktree_paths[eng_id],
                        merge_result.conflict_files,
                    )
                    # Retry merge after resolution
                    retry_merge = self.git.merge_branch(result.branch_name)
                    if retry_merge.success:
                        self.graph.mark_completed(task_id)
                    else:
                        self.graph.mark_failed(task_id)
                        logger.error(f"Failed to merge {task_id} after conflict resolution")

                # Sync all active worktrees to latest main
                for wt_eng_id, wt_path in self.worktree_paths.items():
                    if wt_eng_id in running_tasks and wt_eng_id != eng_id:
                        self.git.reset_hard("main", cwd=wt_path)

                del running_tasks[eng_id]

            # Condense context periodically
            if iteration % 5 == 0 and iteration > 0:
                self.messages = await self.condenser.condense(
                    self.messages, self.graph
                )

            iteration += 1

            # Check termination
            if self.graph.is_done():
                logger.info("All tasks completed!")
                break

    def _build_task_spec(self, engineer_id: str, task_node: TaskNode):
        """Build the appropriate task spec based on mode."""
        from caid.schemas import Commit0Task, PaperBenchTask
        if self.config.mode == "commit0":
            return Commit0Task(
                engineer_id=engineer_id,
                task_id=task_node.task_id,
                file_path=task_node.file_path,
                functions_to_implement=task_node.functions,
                complexity=task_node.complexity,
                instruction=task_node.metadata.get("instruction", "Implement the functions."),
            )
        else:
            return PaperBenchTask(
                engineer_id=engineer_id,
                task_id=task_node.task_id,
                task_node_id=task_node.metadata.get("task_node_id"),
                requirements=task_node.metadata.get("requirements", ""),
                task_category=task_node.metadata.get("task_category", "Code Development"),
                estimated_complexity=task_node.complexity,
                instruction=task_node.metadata.get("instruction", "Implement the task."),
            )

    async def _final_review(self) -> None:
        """Manager final review pass before submission."""
        messages = [
            {"role": "system", "content": "You are reviewing the final state of the repository after "
             "all engineer agents have completed their tasks."},
            {"role": "user", "content": f"Completed tasks: {sorted(self.graph.completed)}\n"
             f"Total tasks: {len(self.graph.all_tasks)}\n"
             f"Review the repository and make any final adjustments."},
        ]
        await self.llm.call(messages)

    def _cleanup_worktrees(self) -> None:
        """Remove all worktrees."""
        for eng_id, path in self.worktree_paths.items():
            try:
                self.git.worktree_remove(path)
            except Exception as e:
                logger.warning(f"Failed to remove worktree {path}: {e}")
```

---

### Step 9: Commit0 Import-Based Dependency Extractor

**What**: Parse Python source files in a repository to extract import-level dependencies and build the dependency graph automatically.

**Why**: CAID Section 2.1 specifies that for Commit0 tasks, the dependency graph is built from Python import analysis. The Manager explores the repo, identifies stub files (functions with `pass` bodies), and determines which files depend on which.

**Files**:
- `caid/commit0/__init__.py`
- `caid/commit0/extractor.py`
- `tests/test_commit0_extractor.py`

**Acceptance Criteria**:
- Given a repo path, finds all `.py` files with stub functions (body is just `pass` or `...`).
- Extracts `import` and `from ... import` statements to build file-level dependency edges.
- Returns a `DependencyGraph` with file-level nodes and import-based edges.
- Ignores stdlib and third-party imports (only tracks intra-repo imports).
- Splits large files into function-level nodes when they contain more than 5 stubs.

**Code Skeleton**:
```python
# caid/commit0/extractor.py
import ast
from pathlib import Path
from caid.graph import DependencyGraph, TaskNode
import logging

logger = logging.getLogger(__name__)

def extract_dependencies(repo_path: Path) -> DependencyGraph:
    """Build dependency graph from Python import analysis."""
    repo_path = Path(repo_path).resolve()
    graph = DependencyGraph()

    # Step 1: Find all Python files with stubs
    stub_files = _find_stub_files(repo_path)
    logger.info(f"Found {len(stub_files)} files with stubs")

    # Step 2: Extract imports for each stub file
    import_map = {}  # file_path -> list of imported repo files
    module_to_file = _build_module_map(repo_path)

    for file_path, stubs in stub_files.items():
        rel_path = str(file_path.relative_to(repo_path))
        imports = _extract_imports(file_path, module_to_file, repo_path)
        import_map[rel_path] = imports

        # Decide granularity: file-level or function-level
        if len(stubs) > 5:
            # Split into function-level nodes
            for func_name in stubs:
                task_id = f"{rel_path}::{func_name}"
                graph.add_task(TaskNode(
                    task_id=task_id,
                    file_path=rel_path,
                    functions=[func_name],
                    complexity=_estimate_complexity(func_name, file_path),
                ))
        else:
            graph.add_task(TaskNode(
                task_id=rel_path,
                file_path=rel_path,
                functions=stubs,
                complexity="medium",
            ))

    # Step 3: Add dependency edges based on imports
    all_task_files = {
        node_id.split("::")[0] if "::" in node_id else node_id
        for node_id in graph.all_tasks
    }
    for file_path, imports in import_map.items():
        for imported_file in imports:
            if imported_file in all_task_files:
                # imported_file must be completed before file_path
                source_nodes = [
                    t for t in graph.all_tasks
                    if t == imported_file or t.startswith(imported_file + "::")
                ]
                target_nodes = [
                    t for t in graph.all_tasks
                    if t == file_path or t.startswith(file_path + "::")
                ]
                for src in source_nodes:
                    for tgt in target_nodes:
                        if src != tgt:
                            try:
                                graph.add_dependency(src, tgt)
                            except ValueError:
                                pass  # Skip if it would create a cycle

    return graph


def _find_stub_files(repo_path: Path) -> dict[Path, list[str]]:
    """Find Python files containing stub functions (body is pass or ...)."""
    stubs = {}
    for py_file in repo_path.rglob("*.py"):
        if ".git" in py_file.parts or "venv" in py_file.parts:
            continue
        try:
            tree = ast.parse(py_file.read_text())
        except SyntaxError:
            continue
        file_stubs = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if _is_stub(node):
                    file_stubs.append(node.name)
        if file_stubs:
            stubs[py_file] = file_stubs
    return stubs


def _is_stub(func_node: ast.FunctionDef) -> bool:
    """Check if a function body is just `pass` or `...` (Ellipsis)."""
    if len(func_node.body) == 1:
        stmt = func_node.body[0]
        if isinstance(stmt, ast.Pass):
            return True
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
            if stmt.value.value is ...:
                return True
    return False


def _build_module_map(repo_path: Path) -> dict[str, str]:
    """Map module names to relative file paths."""
    module_map = {}
    for py_file in repo_path.rglob("*.py"):
        if ".git" in py_file.parts or "venv" in py_file.parts:
            continue
        rel = py_file.relative_to(repo_path)
        # Convert path to module name: foo/bar/baz.py -> foo.bar.baz
        parts = list(rel.parts)
        if parts[-1] == "__init__.py":
            parts = parts[:-1]
        else:
            parts[-1] = parts[-1].replace(".py", "")
        module_name = ".".join(parts)
        module_map[module_name] = str(rel)
    return module_map


def _extract_imports(
    file_path: Path, module_map: dict[str, str], repo_path: Path
) -> list[str]:
    """Extract intra-repo imports from a Python file."""
    try:
        tree = ast.parse(file_path.read_text())
    except SyntaxError:
        return []

    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in module_map:
                    imports.append(module_map[alias.name])
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module in module_map:
                imports.append(module_map[node.module])
            elif node.module:
                # Try partial match (parent package)
                parts = node.module.split(".")
                for i in range(len(parts), 0, -1):
                    candidate = ".".join(parts[:i])
                    if candidate in module_map:
                        imports.append(module_map[candidate])
                        break
    return list(set(imports))


def _estimate_complexity(func_name: str, file_path: Path) -> str:
    """Simple heuristic for function complexity."""
    # Could be enhanced with AST analysis of function signature
    return "medium"
```

---

### Step 10: Test-to-File Mapper (Commit0)

**What**: Map test files to source files so each engineer knows which tests are relevant to their assigned task.

**Why**: CAID Section 2.4 specifies that engineers run the "relevant test subset" within their worktree. The test mapper determines which test files exercise which source files by analyzing test file imports.

**Files**:
- `caid/commit0/test_mapper.py`
- `tests/test_test_mapper.py`

**Acceptance Criteria**:
- Given a repo path and a source file, returns the list of test files that import from that source.
- Handles both `tests/test_foo.py` naming conventions and import-based detection.
- Returns empty list (not error) if no tests found.

**Code Skeleton**:
```python
# caid/commit0/test_mapper.py
import ast
from pathlib import Path

def get_relevant_tests(repo_path: Path, source_file: str) -> list[str]:
    """Find test files that import from the given source file."""
    repo_path = Path(repo_path).resolve()
    source_module = _file_to_module(source_file)
    relevant = []

    # Strategy 1: Convention-based (test_<name>.py for <name>.py)
    source_name = Path(source_file).stem
    for test_file in repo_path.rglob("test_*.py"):
        rel = str(test_file.relative_to(repo_path))
        if source_name in test_file.stem:
            relevant.append(rel)

    # Strategy 2: Import-based
    for test_file in repo_path.rglob("test_*.py"):
        rel = str(test_file.relative_to(repo_path))
        if rel in relevant:
            continue
        try:
            tree = ast.parse(test_file.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if source_module in alias.name:
                        relevant.append(rel)
                        break
            elif isinstance(node, ast.ImportFrom):
                if node.module and source_module in node.module:
                    relevant.append(rel)
                    break

    return list(set(relevant))


def build_test_map(repo_path: Path) -> dict[str, list[str]]:
    """Build a complete mapping of source files to their test files."""
    repo_path = Path(repo_path).resolve()
    test_map = {}
    source_files = [
        str(f.relative_to(repo_path))
        for f in repo_path.rglob("*.py")
        if "test" not in f.name and ".git" not in f.parts and "venv" not in f.parts
    ]
    for src in source_files:
        tests = get_relevant_tests(repo_path, src)
        if tests:
            test_map[src] = tests
    return test_map


def _file_to_module(file_path: str) -> str:
    """Convert file path to module name."""
    parts = Path(file_path).parts
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        parts = list(parts)
        parts[-1] = parts[-1].replace(".py", "")
    return ".".join(parts)
```

---

### Step 11: PaperBench Task Decomposer

**What**: Use the Manager LLM to read a paper description and decompose it into a dependency graph of implementation tasks.

**Why**: CAID Section 2.1 states that for PaperBench, the dependency graph is inferred by the manager reading the paper. Unlike Commit0, there is no import structure to parse -- the LLM must infer the task breakdown.

**Files**:
- `caid/paperbench/__init__.py`
- `caid/paperbench/decomposer.py`
- `tests/test_paperbench_decomposer.py`

**Acceptance Criteria**:
- Given a paper description (text or path to PDF/markdown), produces a `DependencyGraph`.
- The LLM is prompted to output a PaperBenchDelegation JSON.
- The delegation JSON is parsed into graph nodes and edges.
- Handles the case where the LLM produces an invalid graph (cycles, missing dependencies).

**Code Skeleton**:
```python
# caid/paperbench/decomposer.py
from caid.config import CAIDConfig
from caid.llm import LLMClient
from caid.graph import DependencyGraph, TaskNode
from caid.schemas import PaperBenchDelegation
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

DECOMPOSE_PROMPT = """You are analyzing a research paper to create an implementation plan.
Break the paper's contributions into discrete, implementable tasks.
For each task, identify:
- A unique task_id
- Requirements (what needs to be built)
- Category (Code Development, Experiment Running, Results Analysis, Other)
- Complexity (simple, medium, complex)
- Dependencies on other tasks (by task_id)

Output as JSON matching this schema:
{{
  "delegation_plan": {{
    "first_round": {{
      "num_agents": <int>,
      "reasoning": "<string>",
      "tasks": [...]
    }},
    "remaining_tasks": [...]
  }}
}}

Paper content:
{paper_content}
"""

async def decompose_paper(
    config: CAIDConfig,
    llm: LLMClient,
) -> DependencyGraph:
    """Decompose a paper into a dependency graph of tasks."""
    # Read paper content
    paper_content = _read_paper(config)

    messages = [
        {"role": "system", "content": "You are a research paper analyst."},
        {"role": "user", "content": DECOMPOSE_PROMPT.format(paper_content=paper_content)},
    ]

    delegation = await llm.call(messages, response_model=PaperBenchDelegation)

    # Convert delegation plan to DependencyGraph
    graph = DependencyGraph()

    # Add first round tasks
    for task in delegation.delegation_plan.first_round.tasks:
        graph.add_task(TaskNode(
            task_id=task.task_id,
            metadata={
                "task_node_id": task.task_node_id,
                "requirements": task.requirements,
                "task_category": task.task_category,
                "instruction": task.instruction,
                "complexity": task.estimated_complexity,
            },
        ))

    # Add remaining tasks
    for task in delegation.delegation_plan.remaining_tasks:
        graph.add_task(TaskNode(
            task_id=task.task_id,
            metadata={
                "task_node_id": task.task_node_id,
                "requirements": task.requirements,
                "task_category": task.task_category,
                "complexity": task.estimated_complexity,
            },
        ))
        # Add dependency edges
        for dep in task.depends_on:
            if dep in graph.all_tasks:
                try:
                    graph.add_dependency(dep, task.task_id)
                except ValueError:
                    logger.warning(f"Skipping cyclic dependency: {dep} -> {task.task_id}")

    return graph


def _read_paper(config: CAIDConfig) -> str:
    """Read paper content from task description or file."""
    # Try to read from a file path in the task description
    if config.task_description.endswith((".md", ".txt")):
        paper_path = Path(config.task_description)
        if paper_path.exists():
            return paper_path.read_text()
    return config.task_description
```

---

### Step 12: CLI Entry Point

**What**: Build a Click-based CLI that ties everything together and provides the user-facing interface.

**Why**: User requirement. The CLI must accept a config file or command-line arguments, run the CAID framework, and report results.

**Files**:
- `caid/cli.py`
- `caid/api.py`

**Acceptance Criteria**:
- `caid run --config caid.yaml` runs the full framework.
- `caid run --repo ./my-repo --mode commit0 --engineers 4` works with CLI args.
- `caid graph --repo ./my-repo` prints the dependency graph without running engineers (for debugging).
- Exit code 0 on success, non-zero on failure.
- Progress output shows task assignments, completions, and final summary.

**Code Skeleton**:
```python
# caid/cli.py
import click
import asyncio
from pathlib import Path
from caid.config import CAIDConfig
from caid.manager import ManagerAgent
import logging

@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
def cli(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

@cli.command()
@click.option("--config", "-c", type=click.Path(exists=True), help="Path to YAML config file")
@click.option("--repo", type=click.Path(exists=True), help="Repository path")
@click.option("--mode", type=click.Choice(["commit0", "paperbench"]), default="commit0")
@click.option("--engineers", type=int, default=None, help="Number of engineer agents")
@click.option("--task", type=str, default="", help="Task description")
def run(config, repo, mode, engineers, task) -> None:
    """Run the CAID multi-agent framework."""
    if config:
        cfg = CAIDConfig.from_yaml(config)
    else:
        kwargs = {"mode": mode, "task_description": task}
        if repo:
            kwargs["repo_path"] = Path(repo)
        if engineers:
            kwargs["max_engineers"] = engineers
        cfg = CAIDConfig(**kwargs)

    manager = ManagerAgent(cfg)
    result = asyncio.run(manager.run())

    click.echo(f"\n=== CAID Run Complete ===")
    click.echo(f"Tasks completed: {result['completed_tasks']}/{result['total_tasks']}")
    click.echo(f"Token usage: {result['token_usage']}")

@cli.command()
@click.option("--repo", type=click.Path(exists=True), required=True, help="Repository path")
def graph(repo) -> None:
    """Display the dependency graph for a repository (Commit0 mode)."""
    from caid.commit0.extractor import extract_dependencies
    g = extract_dependencies(Path(repo))
    click.echo(f"Tasks: {len(g.all_tasks)}")
    for task_id in g.topological_order():
        node = g.get_node_data(task_id)
        click.echo(f"  {task_id}: {node.functions} (complexity={node.complexity})")

def main() -> None:
    cli()

if __name__ == "__main__":
    main()
```

```python
# caid/api.py
"""Programmatic API for calling CAID from external agents."""
import asyncio
from pathlib import Path
from caid.config import CAIDConfig
from caid.manager import ManagerAgent

async def run_caid(
    repo_path: str | Path,
    mode: str = "commit0",
    task_description: str = "",
    max_engineers: int = 4,
    model: str = "claude-sonnet-4-5-20260120",
    api_key: str = "",
    **kwargs,
) -> dict:
    """Run CAID and return results. Callable from any Python agent."""
    from caid.config import LLMConfig
    config = CAIDConfig(
        repo_path=Path(repo_path),
        mode=mode,
        task_description=task_description,
        max_engineers=max_engineers,
        llm=LLMConfig(model=model, api_key=api_key),
        **kwargs,
    )
    manager = ManagerAgent(config)
    return await manager.run()

def run_caid_sync(**kwargs) -> dict:
    """Synchronous wrapper for run_caid."""
    return asyncio.run(run_caid(**kwargs))
```

---

## 5. Data and Preprocessing

### Expected Input Formats

**Commit0 Mode**:
- A git repository path containing Python source files.
- Source files should contain stub functions (functions with `pass` or `...` as body).
- Test files following `test_*.py` naming convention.
- The repository must be a valid git repo (has `.git` directory).

**PaperBench Mode**:
- A paper description as plain text, markdown file path, or inline string.
- Optionally, a rubric JSON file with task node IDs and weights.
- A workspace directory for the submission output.

### Preprocessing Pipeline

1. **Repository Validation**: Check that the path is a git repo, has a `main` branch, and is in a clean state.
2. **Stub Detection** (Commit0): AST-parse all `.py` files to find functions with stub bodies.
3. **Import Extraction** (Commit0): AST-parse imports to build the dependency map.
4. **Test Discovery** (Commit0): Find all `test_*.py` files and map them to source files via imports.
5. **Paper Parsing** (PaperBench): Read paper content and pass to the LLM for decomposition.
6. **Graph Construction**: Build `networkx.DiGraph`, validate it is acyclic, compute topological order.

### Dataset Requirements

- No specific datasets required. The framework operates on arbitrary git repos (Commit0) or paper descriptions (PaperBench).
- For evaluation against the paper's benchmarks, the user would need access to the Commit0-Lite benchmark (16 repos) or PaperBench (20 papers), which are external.

---

## 6. Testing Strategy

### Unit Tests

| Component | Test File | Key Tests |
|-----------|-----------|-----------|
| Config | `tests/test_config.py` | Load from YAML, defaults match paper values, env var override |
| Schemas | `tests/test_schemas.py` | Round-trip serialization, validation errors on bad input |
| Graph | `tests/test_graph.py` | Readiness check, assignable tasks, cycle detection, completion tracking |
| Git Ops | `tests/test_git_ops.py` | Worktree create/remove, commit, merge, conflict detection (uses temp repo) |
| LLM | `tests/test_llm.py` | JSON extraction, mock LLM calls, retry logic |
| Condenser | `tests/test_condenser.py` | Output is shorter than input, preserves key state |
| Engineer | `tests/test_engineer.py` | Mock LLM + mock git, test iteration loop, restricted file enforcement |
| Extractor | `tests/test_commit0_extractor.py` | Parse stubs, extract imports, build graph from sample repo |
| Test Mapper | `tests/test_test_mapper.py` | Convention-based and import-based mapping |
| Decomposer | `tests/test_paperbench_decomposer.py` | Mock LLM returns valid delegation, graph is acyclic |

### Integration Tests

| Test | Description |
|------|-------------|
| `tests/integration/test_single_task.py` | One engineer, one task, verify commit and merge (mock LLM) |
| `tests/integration/test_two_engineers.py` | Two engineers with dependency, verify ordering (mock LLM) |
| `tests/integration/test_merge_conflict.py` | Force a merge conflict, verify conflict resolution flow |
| `tests/integration/test_full_commit0.py` | Full run on a small test repo with 3 stub files (mock LLM) |
| `tests/integration/test_full_paperbench.py` | Full run with a mock paper description (mock LLM) |

### Evaluation Metrics (from Paper)

- **Pass Rate** (Commit0): `passing_tests / total_tests * 100%` (Eq. 6)
- **PaperBench Score**: Weighted average of rubric node scores (Eq. 7) -- requires external judge
- **Token Usage**: Total prompt + completion tokens across all agents
- **Wall-Clock Time**: End-to-end runtime in seconds

### Running Tests

```bash
# Unit tests
pytest tests/ -x --tb=short -q

# Integration tests (require git)
pytest tests/integration/ -x --tb=short -q

# All tests with coverage
pytest tests/ --cov=caid --cov-report=term-missing
```

---

## 7. Integration with Existing Agents

### Python API

Any OpenAI or Claude agent can call CAID programmatically:

```python
from caid.api import run_caid

# Async usage (from an async agent)
result = await run_caid(
    repo_path="/path/to/repo",
    mode="commit0",
    max_engineers=4,
    model="claude-sonnet-4-5-20260120",
    api_key="sk-...",
)

# Sync usage (from a synchronous agent)
from caid.api import run_caid_sync
result = run_caid_sync(repo_path="/path/to/repo", mode="commit0")
```

### CLI Usage (from agent subprocess)

```bash
caid run --repo /path/to/repo --mode commit0 --engineers 4
```

### API Surface

```python
# Top-level functions
async def run_caid(repo_path, mode, task_description, max_engineers, model, api_key, **kwargs) -> dict
def run_caid_sync(**kwargs) -> dict

# Key classes (for advanced integration)
class CAIDConfig          # Configuration
class ManagerAgent        # Orchestrator (call .run())
class EngineerAgent       # Individual engineer (call .run())
class DependencyGraph     # Task graph (call .get_assignable_tasks(), .is_done())
class GitOps              # Git operations

# Result format (dict)
{
    "completed_tasks": int,
    "total_tasks": int,
    "token_usage": {"prompt": int, "completion": int},
}
```

### Serialization and I/O

- **Input**: Filesystem paths (git repos), plain text strings (paper descriptions), YAML config files.
- **Output**: Python dict (from API), JSON-printable (from CLI), git repository state (modified files on main branch).
- **Inter-agent messages**: Pydantic models serialized as JSON strings.

---

## 8. Risks and Open Questions

### Ambiguities in the Paper

1. **Engineer completion signal mechanism**: The paper says engineers signal completion via `git commit`, but does not specify the inter-process communication mechanism. This plan uses `asyncio.Task` completion as the signal (the coroutine returns). This works because we control the engineer loop directly rather than spawning external processes.

2. **LLMSummarizingCondenser specifics**: The paper references this as an OpenHands SDK component without detailing its implementation. The plan implements a simple version (LLM-based summary with state preservation). The real OpenHands implementation may differ.

3. **"Implementation rounds" definition**: The paper mentions 2 implementation rounds but does not clearly define what constitutes a round boundary. This plan interprets a round as one full pass through the delegation loop (assign all assignable tasks, wait for all to complete, then start the next round).

4. **Worktree sync timing**: The paper says worktrees sync via `git reset --hard HEAD` after merges, but syncing a worktree while an engineer is actively working in it could corrupt state. The plan only syncs idle worktrees. This may differ from the paper's intent.

5. **Restricted file enforcement**: The paper only enforces restricted files via prompt instructions, not tooling. A more robust approach would be a git pre-commit hook, but the plan follows the paper's approach for fidelity.

### Scalability Concerns

- **LLM cost**: 4 engineers at 80 iterations each, plus 50 manager iterations, can consume significant tokens. At approximately 4-5x the cost of a single agent (per the paper).
- **Merge conflicts increase with engineer count**: The paper reports degradation at 8 engineers. The framework should log conflict frequency to help users tune engineer count.
- **Context window limits**: Even with the condenser, very large repos may exceed context limits. Consider chunking file contents.
- **Git worktree limits**: Some git configurations limit worktree count. Ensure cleanup is robust.

### Things That May Need User Clarification

1. **Which LLM to use**: The paper tests Claude Sonnet 4.5, GLM 4.7, and MiniMax 2.5. The user should specify their preferred model and API key.
2. **Test runner**: The plan assumes `pytest`. If the target repo uses a different test framework (unittest, nose), the engineer test runner needs adaptation.
3. **Commit0 benchmark access**: If the user wants to evaluate against the exact Commit0-Lite benchmark, they need access to those 16 repositories.
4. **PaperBench rubric format**: If the user wants PaperBench scoring, they need a rubric JSON file in the PaperBench format. Without it, the framework can still decompose and implement, but cannot self-score.
5. **OpenHands integration**: The paper builds on OpenHands v1.11.0. This plan builds a standalone implementation. If the user wants OpenHands compatibility, the engineer agent would need to be adapted to run as an OpenHands agent rather than a direct LLM-calling coroutine.

---

## File Structure Summary

```
caid/
  __init__.py
  config.py              # Step 1: Configuration models
  schemas.py             # Step 2: JSON message schemas
  graph.py               # Step 3: Dependency graph engine
  git_ops.py             # Step 4: Git operations
  llm.py                 # Step 5: LLM abstraction
  condenser.py           # Step 6: Context condenser
  engineer.py            # Step 7: Engineer agent
  manager.py             # Step 8: Manager agent
  cli.py                 # Step 12: CLI entry point
  api.py                 # Step 12: Python API
  commit0/
    __init__.py
    extractor.py         # Step 9: Import dependency extractor
    test_mapper.py       # Step 10: Test-to-file mapper
  paperbench/
    __init__.py
    decomposer.py        # Step 11: Paper task decomposer
tests/
  test_config.py
  test_schemas.py
  test_graph.py
  test_git_ops.py
  test_llm.py
  test_condenser.py
  test_engineer.py
  test_commit0_extractor.py
  test_test_mapper.py
  test_paperbench_decomposer.py
  integration/
    test_single_task.py
    test_two_engineers.py
    test_merge_conflict.py
    test_full_commit0.py
    test_full_paperbench.py
pyproject.toml
caid.yaml.example
```

Build order follows the numbered steps: 1 through 12. Each step produces independently testable code. Steps 1-6 have no inter-dependencies beyond configuration. Steps 7-8 depend on all prior steps. Steps 9-11 are mode-specific and can be built in parallel. Step 12 ties everything together.
