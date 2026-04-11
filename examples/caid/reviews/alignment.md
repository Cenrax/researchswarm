# Paper Alignment Report — CAID Framework (Post-Fix Re-Review)

**Paper**: arXiv:2603.21489v1 — "Effective Strategies for Asynchronous Software Engineering Agents"
**Code**: `output/20260412_004603_build-the-caid-framework-in-python/code/`
**Date**: 2026-04-12
**Reviewer**: Code Review Specialist (automated)
**Review type**: Post-fix re-review

---

## 1. Algorithm Fidelity

| Algorithm Step | In Paper | In Code | Notes |
|---|---|---|---|
| Manager explores repository | Yes | Yes — `_build_dependency_graph()` dispatches to extractor/decomposer | Match |
| Manager makes preparatory commit | Yes | Yes — `_preparatory_commit()` commits stubs to main | Match |
| Manager creates N worktrees | Yes | Yes — `_setup_engineers()` calls `git.worktree_add()` | Match |
| asyncio event loop with `FIRST_COMPLETED` | Yes | Yes — `asyncio.wait(..., return_when=asyncio.FIRST_COMPLETED)` in `_delegation_loop()` | Exact match |
| Merge completed branch into main | Yes | Yes — `_integrate_result()` calls `git.merge_branch()` | Match |
| Conflict routed back to engineer | Yes | Yes — `engineer.resolve_conflict()` called on merge failure | Match |
| Sync active worktrees after merge | Yes | Yes — `_sync_worktrees()` calls `git.reset_hard("main")` | Match |
| LLMSummarizingCondenser every N iterations | Yes | Yes — `condense_sync()` called every 5 iterations | Partial: uses sync fallback, not async LLM condense |
| Final review pass | Yes | Yes — `_final_review()` makes an LLM call with completion summary | Weak: no actual repo inspection |
| Engineer: implement → test → fix loop | Yes | Yes — `EngineerAgent.run()` iterates up to `engineer_max_iterations` | Match |
| Engineer: partial commit on iteration limit | Yes | Yes — falls through to `git.commit_all("Partial ...")` | Match |
| Multiple implementation rounds | Yes | Yes — outer `for impl_round in range(config.implementation_rounds)` | Match |
| Import-based dependency extraction (Commit0) | Yes | Yes — `commit0/extractor.py` with AST import analysis | Match |
| Test-to-file mapping (Commit0) | Yes | Yes — `commit0/test_mapper.py` | Match |
| LLM paper decomposition (PaperBench) | Yes | Yes — `paperbench/decomposer.py` | Match |
| Restricted files instruction to engineers | Yes | Yes — `ENGINEER_SYSTEM_PROMPT` mentions `{restricted_files}` | Instruction-level only (paper notes same limitation) |

**Deviations found:**

- **D1 (Minor)**: `condenser.condense_sync()` is a simple truncation, not LLM summarization. The async `condenser.condense()` exists but is never called in the delegation loop. This is a pragmatic fallback but deviates from paper intent.
- **D2 (Nit)**: `_final_review()` sends a simple message to the LLM but does not inspect actual repository files. The paper describes a substantive manager review pass.
- **D3 (Nit)**: `_sync_worktrees()` correctly skips engineers that are running, which is the right behavior but differs from the literal pseudocode that says "sync all worktrees."

---

## 2. Equation Implementation

### Eq. 1 — Dependency Graph G = (V, E)

- **Paper**: `nx.DiGraph()` with nodes as work units and directed edges for dependencies.
- **Code**: `DependencyGraph._graph: nx.DiGraph` (graph.py:47)
- **Status**: PASS

### Eq. 2 — Completed Set C_t

- **Paper**: `completed_set: set[str]`, grows after each successful `git merge`.
- **Code**: `DependencyGraph._completed: set[str]` (graph.py:48), updated via `mark_completed()` called in `_integrate_result()` after successful merge.
- **Status**: PASS

### Eq. 3 — Task Readiness Condition Ready_t(v_j) ⟺ ∀(v_i, v_j) ∈ E, v_i ∈ C_t

- **Paper**: All predecessors in completed set.
- **Code** (graph.py:81-84):
  ```python
  return all(
      pred in self._completed
      for pred in self._graph.predecessors(task_id)
  )
  ```
- **Status**: PASS — exact translation. No-predecessor case returns `True` (correct).

### Eq. 4 — Assignable Task Set AssignableAt_t = {v ∈ V | Ready_t(v) and v ∉ C_t}

- **Paper**: Ready tasks not yet completed.
- **Code** (graph.py:94-95):
  ```python
  candidates = set(self._graph.nodes) - self._completed - self._in_progress
  assignable = [v for v in candidates if self.is_ready(v)]
  ```
- **Status**: PASS — matches exactly. Priority sort (upstream-first) is an additional improvement over base paper formula.

### Eq. 5 — Termination Done ⟺ C_t = V or rounds ≥ max_rounds or iterations ≥ max_iterations

- **Paper**: Three-way termination.
- **Code**: All three checks present:
  - `self.graph.is_done()` checks C_t == V (manager.py:230)
  - `iteration < self.config.manager_max_iterations` loop bound (manager.py:159)
  - `for impl_round in range(self.config.implementation_rounds)` outer loop (manager.py:94)
- **Status**: PASS

---

## 3. Hyperparameter Defaults

| Parameter | Paper Value | Code Default (config.py) | Match? |
|---|---|---|---|
| `manager_max_iterations` | 50 | 50 | YES |
| `engineer_max_iterations` | 80 | 80 | YES |
| `max_engineers` (Commit0) | 4 | 4 | YES |
| `max_engineers` (PaperBench) | 2 | 4 (same default) | PARTIAL |
| `implementation_rounds` | 2 | 2 | YES |
| `temperature` | 0 (implied) | 0.0 | YES |
| Model | Claude Sonnet 4.5 | `claude-sonnet-4-5-20260120` | YES |

**Note on PaperBench engineers**: The paper specifies 2 optimal engineers for PaperBench vs 4 for Commit0. The code uses a single `max_engineers=4` default regardless of mode. USAGE.md documents this and recommends `--engineers 2` for PaperBench.

---

## 4. Architecture Match

| Component | Paper Description | Code Component | Match? |
|---|---|---|---|
| Centralized Manager | Single manager, orchestrates all | `ManagerAgent` singleton | YES |
| Engineer Agents | N concurrent coroutines | `EngineerAgent` with `asyncio.create_task()` | YES |
| git worktree isolation | `git worktree add` per engineer | `GitOps.worktree_add()` | YES |
| asyncio event loop | `asyncio.wait(FIRST_COMPLETED)` | `manager.py:196-199` | YES |
| JSON communication | Pydantic schemas matching paper appendix | `schemas.py` — all 8 schema types | YES |
| LLMSummarizingCondenser | Context compression preserving graph state | `condenser.py` | YES (partial, see D1) |
| git merge integration | `git merge --no-ff` | `git_ops.py:154` | YES |
| Conflict resolution | Engineer pulls main, resolves, resubmits | `engineer.py:166-216` | YES |
| State sync after merge | `git reset --hard HEAD` on worktrees | `git_ops.py:179-181` | YES |

---

## 5. JSON Schema Match (Paper Appendix)

| Schema | Paper Specified | Code Class | Match? |
|---|---|---|---|
| Commit0 delegation (`delegation_plan`) | YES | `Commit0Delegation` | YES |
| Commit0 reassignment (`assign_task`) | YES | `Commit0Reassignment` | YES |
| PaperBench delegation (`delegation_plan`) | YES | `PaperBenchDelegation` | YES |
| PaperBench reassignment (`assign_task`) | YES | `PaperBenchReassignment` | YES |

Both field names match: Commit0 reassignment uses `assignments`, PaperBench reassignment uses `tasks` — both match the paper appendix schemas.

---

## Alignment Score

**9 / 10**

All core equations (1–5), the asyncio event loop pattern, git worktree isolation, engineer lifecycle, and JSON schemas are faithfully implemented. The two-pass dependency fix in `decomposer.py` ensures correct multi-level dependency chains. Deductions:

- (-1) The LLM condenser is called in sync/truncation mode (not the LLM-based async mode the paper describes), and the final review pass is a thin LLM call without actual repository inspection.
