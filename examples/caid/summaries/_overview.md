# Research Paper Overview

**Objective**: Build the CAID framework in Python
**Date**: 2026-04-12
**Papers Processed**: 1

---

## Relevance Ranking

| Rank | Paper | File | Relevance | Reason |
|---|---|---|---|---|
| 1 | Effective Strategies for Asynchronous Software Engineering Agents (CAID) | `caid.pdf` | HIGH | This paper IS the system to build. Every architectural component, configuration value, JSON schema, and prompt is specified. |

---

## Summary

Only one paper was provided. It is directly and fully relevant to the objective.

---

## Paper 1: CAID

**Title**: Effective Strategies for Asynchronous Software Engineering Agents
**Authors**: Jiayi Geng, Graham Neubig (Carnegie Mellon University, LTI)
**arXiv**: 2603.21489v1 (March 23, 2026)
**Code**: https://github.com/JiayiGeng/CAID

### What to Build

CAID is a Python multi-agent system with these concrete components:

1. **Manager Agent** — an LLM agent that:
   - Explores a repository or paper
   - Constructs a directed dependency graph $G = (V, E)$
   - Delegates tasks via structured JSON to engineer agents
   - Runs an `asyncio` event loop awaiting completion signals
   - Performs `git merge` integration after each engineer commits
   - Re-evaluates readiness ($\text{Ready}_t(v_j)$) and reassigns dynamically
   - Compresses context with `LLMSummarizingCondenser`
   - Performs a final review pass before submission

2. **Engineer Agent(s)** — N concurrent LLM agent coroutines, each:
   - Receives a JSON task specification
   - Operates in an isolated `git worktree`
   - Implements assigned functions
   - Runs relevant tests and iteratively fixes failures
   - Submits a `git commit` as the completion signal
   - Resolves merge conflicts if routed back by the manager

3. **Git Infrastructure** — wrappers for:
   - `git worktree add/remove`
   - `git commit`
   - `git merge` with conflict detection
   - `git reset --hard HEAD` for state sync
   - `git pull` for conflict resolution

4. **JSON Communication Protocol** — Pydantic schemas for all message types (see full schemas in `caid.md` Section 4)

5. **Dependency Graph Engine** — built on `networkx.DiGraph`:
   - For Commit0: auto-built from Python `import` statement analysis
   - For PaperBench: inferred by manager LLM reading the paper

### Key Numbers

| Parameter | Value |
|---|---|
| Manager max_iterations | 50 |
| Engineer max_iterations | 80 |
| Engineers for Commit0 | 4 (optimal) |
| Engineers for PaperBench | 2 (optimal) |
| Implementation rounds | 2 |
| Agent SDK | OpenHands v1.11.0 |
| Best model tested | Claude Sonnet 4.5 |

### Performance Gains Over Single-Agent Baseline

| Benchmark | Best gain | Statistical significance |
|---|---|---|
| PaperBench | +26.6 pp (MiniMax 2.5) | p=0.046 (Claude), p=0.034 (GLM) |
| Commit0-Lite | +14.7 pp (MiniMax 2.5) | p=0.006 (Claude), p=0.007 (MiniMax) |

### Build Order

1. Git worktree manager utilities
2. Dependency graph data structure + readiness checker
3. JSON schema models (Pydantic)
4. Engineer agent coroutine
5. Manager `asyncio` event loop
6. `git merge` handler with conflict re-routing
7. `LLMSummarizingCondenser` integration
8. Commit0 import-based dependency extractor
9. PaperBench paper-reading task decomposer
10. Test-to-file mapper (Commit0)

### Files Produced

- `/Users/subham/Desktop/codes/agentsclaude/output/20260412_004603_build-the-caid-framework-in-python/summaries/caid.md` — full structured summary
- `/Users/subham/Desktop/codes/agentsclaude/output/20260412_004603_build-the-caid-framework-in-python/summaries/caid_equations.md` — all equations with implementation notes and Python pseudocode
- `/Users/subham/Desktop/codes/agentsclaude/output/20260412_004603_build-the-caid-framework-in-python/summaries/_overview.md` — this file
