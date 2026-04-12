# Equations from: Effective Strategies for Asynchronous Software Engineering Agents (CAID)

**Source**: /Users/subham/Desktop/codes/agentsclaude/input/week_16/caid.pdf
**Paper arXiv ID**: arXiv:2603.21489v1

---

> **Note**: CAID is a system/framework paper, not a machine learning paper. It contains no loss functions, neural network layers, or gradient-based equations. Its mathematical content is confined to formal definitions of graph-theoretic scheduling constraints. All equations are from Section 2.1.

---

## 1. Forward Pass / Core Mechanism

### Eq. 1: Repository Dependency Graph

$$G = (V, E)$$

**Variables**:
- $G$ — the dependency graph of the repository
- $V$ — set of work unit nodes. Each $v \in V$ corresponds to one unit of implementation work (a file, a set of functions, or a paper contribution sub-task depending on the benchmark)
- $E$ — set of directed edges. A directed edge $(v_i, v_j) \in E$ indicates that $v_j$ depends on $v_i$ (i.e., $v_i$ must be completed before $v_j$ can begin)

**Used in**: Task Specification and Dependency Graph (Section 2.1), task delegation (Section 2.2)

**Implementation notes**:
```python
import networkx as nx

# Construct the graph
G = nx.DiGraph()
G.add_nodes_from(work_units)          # each node is a task identifier
G.add_edges_from(dependency_pairs)    # (v_i, v_j) means v_j depends on v_i
```

No numerical stability concerns — this is a discrete graph structure.

---

### Eq. 2: Completed Set at Round t

$$C_t \subseteq V$$

**Variables**:
- $C_t$ — the set of work unit nodes that have been completed **and** successfully integrated into the `main` branch by the end of round $t$
- $V$ — the full set of all work unit nodes

**Used in**: determining task readiness (Section 2.1), termination condition

**Implementation notes**:
```python
completed_set: set[str] = set()   # starts empty; grows as engineers merge

# After each successful git merge:
completed_set.add(task_id)
```

The manager updates $C_t$ immediately after each `git merge` succeeds.

---

### Eq. 3: Task Readiness Condition

$$\text{Ready}_t(v_j) \iff \forall (v_i, v_j) \in E,\ v_i \in C_t$$

**Variables**:
- $\text{Ready}_t(v_j)$ — boolean predicate: True if task $v_j$ is eligible for delegation at round $t$
- $(v_i, v_j) \in E$ — all predecessor edges pointing into $v_j$
- $C_t$ — set of completed-and-integrated nodes at round $t$

**Plain English**: A task $v_j$ is ready to be assigned to an engineer only when every task that $v_j$ depends on has already been merged into the main branch. If even one predecessor is unfinished, $v_j$ must wait.

**Used in**: Dependency-Aware Task Delegation (Section 2.2), the manager's reassignment step after each merge

**Implementation notes**:
```python
def is_ready(graph: nx.DiGraph, node: str, completed: set[str]) -> bool:
    """
    Returns True iff all predecessors of `node` are in `completed`.
    graph.predecessors(node) gives all v_i where edge (v_i, node) exists.
    """
    return all(pred in completed for pred in graph.predecessors(node))
```

This is equivalent to checking that the in-degree of $v_j$ in the subgraph induced by $V \setminus C_t$ is zero.

---

### Eq. 4: Assignable Task Set

$$\text{AssignableAt}_t = \{v \in V \mid \text{Ready}_t(v) \text{ and } v \notin C_t\}$$

**Variables**:
- $\text{AssignableAt}_t$ — all tasks that can be delegated at round $t$ (ready but not yet completed)

**Plain English**: The manager picks the next batch of tasks from this set. Up to $N$ tasks are selected (one per engineer), prioritizing tasks that enable earlier test execution and tasks closer to the upstream end of the dependency chain.

**Used in**: manager's selection loop (Section 2.2)

**Implementation notes**:
```python
def get_assignable_tasks(
    graph: nx.DiGraph,
    completed: set[str],
    in_progress: set[str],
) -> list[str]:
    """
    Returns tasks that are ready and not yet completed or in progress.
    """
    candidates = set(graph.nodes) - completed - in_progress
    return [v for v in candidates if is_ready(graph, v, completed)]
```

Sort the result by priority (upstream tasks first, tasks enabling more tests first) before selecting the top-N for assignment.

---

### Eq. 5: Termination Condition

$$\text{Done} \iff C_t = V \quad \text{or} \quad \text{rounds} \geq \text{max\_rounds} \quad \text{or} \quad \text{iterations} \geq \text{max\_iterations}$$

**Variables**:
- $C_t = V$ — all work units completed and integrated
- `max_rounds` — predefined round limit for the manager (not explicitly given a number in the paper; tuned per deployment)
- `max_iterations = 50` for manager, `= 80` for each engineer

**Used in**: Self-Verification and Termination (Section 2.5)

**Implementation notes**: If termination is triggered by limit exhaustion while $C_t \subsetneq V$, the task is considered **incomplete** (partial credit in evaluation).

---

## 2. Metrics (Evaluation)

### Eq. 6: Pass Rate (Commit0)

$$\text{PassRate} = \frac{\text{number of passing unit tests}}{\text{total unit tests}} \times 100\%$$

**Used in**: Commit0 evaluation. The task is considered fully successful only if all tests pass.

---

### Eq. 7: Score (PaperBench)

The PaperBench score is computed by a judge model (GPT-5-mini) against a hierarchical rubric. It is a weighted average of correctness scores per rubric node, weighted by importance to the paper's main contributions.

$$\text{Score}_{\text{PaperBench}} = \frac{\sum_{i} w_i \cdot s_i}{\sum_{i} w_i}$$

**Variables**:
- $w_i$ — weight of rubric node $i$ (determined by importance to the paper's main contributions)
- $s_i \in [0, 1]$ — correctness score assigned by the judge model for rubric node $i$

**Note**: The exact weighting formula is defined by the PaperBench benchmark, not by CAID. CAID uses Code-Dev evaluation protocol.

---

### Eq. 8: Statistical Test

One-sided paired t-test used to assess significance of CAID gains:

$$H_1: \mu_{\text{CAID}} > \mu_{\text{SingleAgent}}$$

$$t = \frac{\bar{d}}{s_d / \sqrt{n}}$$

**Variables**:
- $\bar{d}$ — mean per-repository score difference (CAID minus single-agent)
- $s_d$ — standard deviation of differences
- $n$ — number of repositories (16 for Commit0-Lite, 20 for PaperBench)

**Key results**:
- Commit0 Claude 4.5: $t = 2.87$, $p = 0.006$ (significant)
- Commit0 MiniMax 2.5: $t = 2.81$, $p = 0.007$ (significant)
- PaperBench Claude 4.5: $t = 1.78$, $p = 0.046$ (significant)
- PaperBench GLM 4.7: $t = 1.93$, $p = 0.034$ (significant)
- PaperBench MiniMax 2.5: $t = 0.23$, $p = 0.408$ (not significant)

---

## Pseudocode: Manager Event Loop

```python
# Python asyncio pseudocode for the CAID manager loop
import asyncio

async def caid_manager(
    graph: nx.DiGraph,        # dependency graph G = (V, E)
    max_engineers: int,       # N
    max_iterations_manager: int = 50,
    max_iterations_engineer: int = 80,
):
    completed: set[str] = set()     # C_t
    in_progress: set[str] = set()
    engineer_tasks: dict[str, asyncio.Task] = {}

    # Setup phase
    await manager_explore_repository()
    await manager_preparatory_commit()  # commit stubs to main

    iteration = 0
    while iteration < max_iterations_manager:
        # Select assignable tasks
        assignable = get_assignable_tasks(graph, completed, in_progress)
        if not assignable and not in_progress:
            break  # Done: C_t == V (or no progress possible)

        # Assign to idle engineers (up to max_engineers)
        idle_slots = max_engineers - len(in_progress)
        to_assign = assignable[:idle_slots]

        for task_id in to_assign:
            engineer_id = allocate_engineer()
            worktree_path = git_worktree_add(engineer_id, task_id)
            task_spec = build_json_task_spec(task_id, engineer_id)

            # Launch engineer coroutine
            coro = engineer_loop(
                engineer_id, task_spec, worktree_path,
                max_iterations=max_iterations_engineer
            )
            engineer_tasks[engineer_id] = asyncio.create_task(coro)
            in_progress.add(task_id)

        # Await any completion
        done, _ = await asyncio.wait(
            engineer_tasks.values(),
            return_when=asyncio.FIRST_COMPLETED
        )

        for finished_task in done:
            engineer_id, committed_task_id = finished_task.result()

            # Integration
            conflict = git_merge(committed_task_id, target="main")
            if conflict:
                # Hand conflict back to engineer
                await resolve_conflict(engineer_id, committed_task_id)
            else:
                completed.add(committed_task_id)
                in_progress.discard(committed_task_id)
                git_worktree_remove(engineer_id)

            del engineer_tasks[engineer_id]

        # Sync all worktrees to latest main
        for worktree in active_worktrees():
            git_reset_hard_head(worktree)

        # Compress manager context
        manager_context = llm_summarizing_condenser(
            history=manager_context,
            preserve={"dependency_graph": graph, "completed": completed}
        )

        iteration += 1

    await manager_final_review()


async def engineer_loop(
    engineer_id: str,
    task_spec: dict,
    worktree_path: str,
    max_iterations: int = 80,
) -> tuple[str, str]:
    for _ in range(max_iterations):
        # Implement
        await llm_implement(task_spec, worktree_path)

        # Self-verify
        test_result = await run_relevant_tests(worktree_path, task_spec)
        if test_result.passed:
            git_commit(worktree_path, task_spec["task_id"])
            return engineer_id, task_spec["task_id"]
        else:
            # Iterative fix using error logs
            task_spec["error_context"] = test_result.error_log

    # Iteration limit hit: partial commit
    git_commit(worktree_path, task_spec["task_id"] + "-partial")
    return engineer_id, task_spec["task_id"]
```

---

## Categorization Summary

| Category | Equations Present |
|---|---|
| Loss Functions | None (not an ML training paper) |
| Forward Pass / Core Mechanism | Eq. 1 (graph), Eq. 2 (completed set), Eq. 3 (readiness), Eq. 4 (assignable set) |
| Attention / Novel Mechanism | None |
| Normalization / Regularization | None |
| Initialization | None |
| Metrics | Eq. 5 (termination), Eq. 6 (pass rate), Eq. 7 (PaperBench score), Eq. 8 (t-test) |
