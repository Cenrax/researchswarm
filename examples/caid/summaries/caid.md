# Effective Strategies for Asynchronous Software Engineering Agents (CAID)

**Source**: /Users/subham/Desktop/codes/agentsclaude/input/week_16/caid.pdf
**Date Read**: 2026-04-12
**Relevance**: HIGH to the stated objective (build the CAID framework in Python)

---

## 1. Core Idea

CAID (Centralized Asynchronous Isolated Delegation) is a multi-agent coordination framework for long-horizon software engineering tasks. It solves the problem of concurrent agent interference on shared codebases by grounding coordination in three SWE primitives: centralized task delegation via a dependency graph, asynchronous execution of engineer agents in isolated `git worktree` workspaces, and structured integration through `git merge`. The main contribution is proving that branch-and-merge discipline — borrowed directly from human software team practices — is the key mechanism enabling reliable parallel multi-agent development.

---

## 2. Relevance to Objective

This paper IS the system to build. Every section describes a concrete component of the Python implementation:

- The manager agent loop, dependency graph construction, and JSON-based task delegation protocol are fully specified.
- The engineer agent coroutine lifecycle (implement → self-verify → commit) is described precisely.
- The Python `asyncio` event loop with `await` for coordination is explicitly named.
- All prompts for both task types (Commit0 and PaperBench) are provided verbatim in the appendix.
- The JSON schemas for manager-to-engineer communication are given in full.
- Configuration numbers (iterations, engineer counts) are documented.

---

## 3. Key Techniques

- **Dependency Graph Construction**: Manager builds a directed acyclic graph $G = (V, E)$ of work units. For Commit0, nodes are files/functions; edges come from Python `import` analysis. For PaperBench, nodes are paper contribution sub-tasks; edges are inferred from reading the paper.
- **Dependency-Aware Task Delegation**: Manager selects only "ready" nodes (all predecessors completed) for assignment. Prioritizes tasks that enable earlier test execution and upstream dependencies.
- **git worktree Isolation**: Each engineer agent operates in a physically separate worktree (`git worktree add`). Edits cannot interfere until an explicit `git merge`.
- **Asynchronous Event Loop**: Python `asyncio` runs engineers as independent coroutines concurrently. Manager uses an event loop with `await` to listen for engineer completion signals.
- **Structured JSON Communication**: All manager-to-engineer messages are machine-parsable JSON (not free-form text). Schemas are provided in the appendix.
- **Engineer Self-Verification**: Before committing, each engineer runs the relevant test subset within its worktree. Failed tests must be fixed before submission.
- **git merge Integration**: Manager merges completed engineer branches into `main`. Merge conflicts are handed back to the originating engineer to resolve locally (pull → resolve → resubmit).
- **LLMSummarizingCondenser**: Manager uses a summarizing condenser to compress execution history and prevent context explosion while preserving the dependency graph state, completed tasks, and unresolved errors.
- **State Synchronization**: After merging, worktrees sync to the latest integrated state via `git reset --hard HEAD`.
- **Termination Control**: Execution ends when all dependency graph nodes are completed and integrated, or when predefined limits (max rounds for manager, max iterations per engineer) are exhausted.

---

## 4. Implementation-Critical Details

### Agent Configuration (exact values from paper)

| Parameter | Single-Agent | Manager (CAID) | Engineer (CAID) |
|---|---|---|---|
| `max_iterations` | 100 | 50 | 80 |
| Engineers on PaperBench | — | 2 | 2 |
| Engineers on Commit0 | — | 4 | 4 |
| Implementation rounds | 1 | 2 | 2 |

**Optimal engineer counts**: 2 for PaperBench, 4 for Commit0-Lite. Performance degrades at 8 engineers due to delegation quality breakdown.

### Agent SDK

- Built on **OpenHands agent SDK v1.11.0** (`openhands`)
- Manager is instantiated as a centralized OpenHands agent
- Engineers are instantiated as software-engineer OpenHands agents
- Models tested: Claude Sonnet 4.5 (best), GLM 4.7, MiniMax 2.5

### Task Lifecycle (per round)

1. Manager explores repository structure.
2. Manager builds dependency graph (import analysis for Commit0; paper reading for PaperBench).
3. Manager commits any stubs it adds to `main` branch (preparatory commit).
4. Manager creates `git worktree` for each engineer.
5. Engineers run concurrently as `asyncio` coroutines.
6. Each engineer: implement assigned functions → run tests → resolve failures → `git commit`.
7. Manager awaits first engineer completion.
8. Manager runs `git merge` of engineer branch into `main`.
9. If merge conflict: manager instructs engineer to `git pull main` → resolve → resubmit.
10. Manager updates dependency state, assigns next ready task.
11. Idle engineers receive next tasks or remain idle.
12. Loop continues until no ready tasks remain or limits are exhausted.
13. Manager does a final review pass before submission.

### Restricted Files

- Shared files (e.g., `__init__.py`) are marked as restricted.
- Engineers are explicitly instructed not to commit changes to restricted files.

### Worktree Lifecycle

- Worktree created: `git worktree add <path> <branch>`
- Worktree deleted: after all tasks for that engineer are done or iteration limit reached.

### Manager JSON Output Schema (Commit0 — task delegation)

```json
{
  "delegation_plan": {
    "first_round": {
      "num_agents": "<int 1..N>",
      "reasoning": "<string>",
      "tasks": [
        {
          "engineer_id": "<string>",
          "task_id": "<string>",
          "file_path": "<path/to/file.py>",
          "functions_to_implement": ["func1", "func2"],
          "complexity": "simple|medium|complex",
          "instruction": "<string>"
        }
      ]
    },
    "remaining_tasks": [
      {
        "task_id": "<string>",
        "file_path": "<string>",
        "functions_to_implement": ["<string>"],
        "complexity": "simple|medium|complex",
        "depends_on": ["<file_path_1>"]
      }
    ]
  }
}
```

### Manager JSON Output Schema (Commit0 — reassignment)

```json
{
  "assign_task": {
    "reasoning": "<string>",
    "assignments": [
      {
        "engineer_id": "<string>",
        "task_id": "<string or fix-<original-id>>",
        "file_path": "<string>",
        "functions_to_implement": ["<string>"],
        "instruction": "<string>",
        "complexity": "simple|medium|complex"
      }
    ]
  }
}
```

### Manager JSON Output Schema (PaperBench — task delegation)

```json
{
  "delegation_plan": {
    "first_round": {
      "num_agents": "<int>",
      "reasoning": "<string>",
      "tasks": [
        {
          "engineer_id": "<string>",
          "task_id": "<string>",
          "task_node_id": "<rubric node id if available>",
          "requirements": "<string>",
          "task_category": "Code Development|Experiment Running|Results Analysis|Other",
          "estimated_complexity": "simple|medium|complex",
          "instruction": "<string>"
        }
      ]
    },
    "remaining_tasks": [
      {
        "task_id": "<string>",
        "task_node_id": "<string>",
        "requirements": "<string>",
        "task_category": "<string>",
        "estimated_complexity": "<string>",
        "depends_on": ["<task_id>"]
      }
    ]
  }
}
```

### Manager JSON Output Schema (PaperBench — reassignment)

```json
{
  "assign_task": {
    "reasoning": "<string>",
    "tasks": [
      {
        "engineer_id": "<string>",
        "task_id": "<string>",
        "task_node_id": "<string>",
        "requirements": "<string>",
        "task_category": "<string>",
        "estimated_complexity": "<string>",
        "instruction": "<string>"
      }
    ]
  }
}
```

### Dependency Graph Rule (exact eligibility formula)

A task node $v_j$ is eligible for delegation if and only if all its predecessor nodes are in the completed set:

$$\text{Ready}_t(v_j) \iff \forall (v_i, v_j) \in E,\ v_i \in C_t$$

where $C_t$ is the set of completed-and-integrated nodes at round $t$.

### Coordination Prompt Variants and Trade-offs

| Mode | Pass Rate (Commit0 subset) | Runtime |
|---|---|---|
| Round-Manager Review | 60.2% | 3689.1s |
| Engineer Self-Verification (default) | 55.1% | 2243.9s |
| Efficiency-Prioritized | 54.0% | 1908.6s |

### Commit0 Manager Exploration Steps

1. Check import statements to identify file-level dependencies.
2. Collect all executable test cases.
3. Map which tests exercise which files.
4. Identify which files must be implemented first for dependent tests to pass.
5. Delegate at file level by default; split to function level if a file has too many `pass` stubs.

### PaperBench Submission Requirements

- Submission at `/workspace/submission/`
- Must include `/workspace/submission/reproduce.sh`
- Must not exceed 1GB committed size
- Run in Ubuntu 24.04 LTS Docker with NVIDIA A10 GPU
- Max runtime: 7 days
- Final `README.md` required after all tasks complete

---

## 5. Equations

### Dependency Graph Representation

$$G = (V, E)$$

**Variables**:
- $V$ — set of work unit nodes (files, functions, or paper contribution sub-tasks)
- $E$ — set of directed edges where $(v_i, v_j) \in E$ means $v_j$ depends on $v_i$

**Used in**: task specification and dependency graph construction (Section 2.1)

---

### Task Readiness Condition

$$\text{Ready}_t(v_j) \iff \forall (v_i, v_j) \in E,\ v_i \in C_t$$

**Variables**:
- $\text{Ready}_t(v_j)$ — boolean: whether task $v_j$ is eligible for delegation at round $t$
- $C_t \subseteq V$ — set of completed-and-integrated nodes at round $t$
- $E$ — dependency edges

**Plain English**: A task can only be assigned when every task it depends on has already been merged into `main`.

**Used in**: dependency-aware task delegation (Section 2.2), the manager's assignment loop

**Implementation note**: Implement as a set membership check. After each `git merge`, add the completed node to $C_t$ and re-evaluate which nodes in $V \setminus C_t$ are now ready.

---

### Ready Set Selection

$$\text{AssignableAt}_t = \{v \in V \mid \text{Ready}_t(v)\}$$

**Used in**: the manager selects the next batch of tasks from this set, up to $N$ engineers.

---

## 6. Architecture Diagram

```
INPUT: Repository + Task Description
         |
         v
+------------------+
|    MANAGER       |  (max_iterations=50, 1 instance)
|                  |
|  1. Explore repo |
|  2. Build DAG    |
|  3. Preparatory  |
|     commit       |
|  4. Create N     |
|     worktrees    |
+--------+---------+
         |
         | JSON delegation (first_round tasks)
         |
    asyncio.gather()
    /       |       \
   v        v        v
+------+ +------+ +------+
| Eng1 | | Eng2 | | Eng3 |  (max_iterations=80 each)
|      | |      | |      |
| impl | | impl | | impl |
| test | | test | | test |
| fix  | | fix  | | fix  |
| git  | | git  | | git  |
|commit| |commit| |commit|
+--+---+ +--+---+ +--+---+
   |         |         |
   | signal  | signal  | signal
   v         v         v
+------------------+
|    MANAGER       |
|                  |
| await completion |
| git merge ->main |
|  (conflict?      |
|   -> engineer    |
|   resolves)      |
| update C_t       |
| reassign next    |
| ready tasks      |
+--------+---------+
         |
         | (loop until V == C_t or limits hit)
         v
+------------------+
|  FINAL REVIEW    |
|  (manager)       |
|  Submit result   |
+------------------+
```

**SWE Primitive Mapping**:

| Component | Git/Python Primitive |
|---|---|
| Isolated workspace | `git worktree add` |
| Engineer completion signal | `git commit` |
| Output integration | `git merge` |
| Conflict handling | Engineer: `git pull` + resolve + resubmit |
| State sync | `git reset --hard HEAD` |
| Concurrent execution | `asyncio` coroutines |
| Coordination cycle | `asyncio` event loop + `await` |
| Context management | `LLMSummarizingCondenser` |

---

## 7. Limitations and Caveats

- **Higher cost**: Multi-agent execution consistently costs more than single-agent (roughly 4-5x API cost for 4 engineers).
- **Wall-clock time not reduced**: Integration is sequential and test-gated; parallel execution gains are offset by merge coordination overhead.
- **Manager delegation quality is the bottleneck**: Weaker LLMs produce unreliable task decomposition, especially for open-ended tasks (PaperBench). MiniMax 2.5 shows non-significant gains on PaperBench ($p = 0.408$).
- **Non-monotonic scaling**: More engineers does not monotonically improve performance. At 8 engineers on Commit0, pass rate drops from 59.1% (4 engineers) to a lower value because delegation becomes too fine-grained and file-level conflicts increase.
- **Prompt-engineering heuristics only**: Task delegation relies on prompts, not learned policies. Coarse-grained or misaligned decomposition produces locally correct but globally incompatible outputs.
- **SWE-specific**: The framework assumes `git` infrastructure, explicit test suites, and versioned artifacts. Extension to non-SWE domains (e.g., document synthesis) requires re-designing isolation and verification mechanisms.
- **Context growth**: Manager context can grow during long runs; the `LLMSummarizingCondenser` is a mitigation but not a complete solution.
- **`__init__.py` and shared files**: Engineers must not touch restricted shared files; enforcement is instruction-level only, not enforced by tooling.

---

## 8. Implementation Priority

### Build First (critical path)

1. **Git worktree manager**: Functions to create/delete worktrees per engineer (`git worktree add/remove`).
2. **Dependency graph data structure**: Directed graph with nodes as work units, edges as dependencies, and a `completed_set` tracker.
3. **Readiness checker**: Function implementing $\text{Ready}_t(v_j)$ — checks if all predecessors of a node are in the completed set.
4. **Manager agent loop**: The central `asyncio` event loop — delegates tasks, awaits completion signals, triggers merges, updates state, reassigns.
5. **Engineer agent coroutine**: The per-engineer async function — receives JSON task spec, implements, runs tests, commits.
6. **JSON schemas**: Pydantic models or TypedDicts for all manager/engineer message formats.
7. **git merge handler with conflict resolution protocol**: After merge attempt, detect conflicts and route back to the originating engineer.

### Build Second

8. **`LLMSummarizingCondenser`**: Context compression for the manager — periodically summarize history while preserving dependency graph and task state.
9. **Commit0-specific dependency extractor**: Parse Python `import` statements to build the file-level dependency graph automatically.
10. **PaperBench-specific task decomposer**: Manager reads paper PDF/markdown and infers implementation order from contribution structure.
11. **Test-to-file mapper**: For Commit0, map which test files import which source files to determine the relevant test subset per engineer.

### Defer or Use Off-the-Shelf

12. **OpenHands SDK integration**: Use OpenHands v1.11.0 directly as the agent substrate; do not reimplement the underlying LLM call machinery.
13. **Round-manager review variant**: The default mode is engineer self-verification; the stricter review mode can be a configuration flag added later.

### Requires Clarification

- The exact implementation of `LLMSummarizingCondenser` is not specified in the paper (it is an OpenHands SDK component — check OpenHands v1.11.0 source).
- The paper does not specify how the manager detects that an engineer has submitted a commit (polling vs. callback vs. watching git refs) — this is an implementation choice.
- The exact format of the "completion signal" from engineer to manager is described as a `git commit`, but the inter-process communication mechanism (e.g., shared filesystem, asyncio queue, file watcher) must be designed.
- "2 implementation rounds" is mentioned but the definition of a round and how the manager knows a round is complete needs to be clarified from the OpenHands SDK or inferred from context.
