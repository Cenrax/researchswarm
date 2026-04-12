# Dependencies

## Required

| Package | Version | Purpose | Paper Reference |
|---------|---------|---------|-----------------|
| networkx | >=3.1 | Directed acyclic graph for dependency tracking (DiGraph, predecessors, topological_sort) | CAID Section 2.1, Eq. 1-4 |
| pydantic | >=2.0 | JSON schema validation for all manager-engineer message types | CAID Section 4 (Appendix schemas) |
| litellm | >=1.40 | Unified LLM API abstraction -- supports OpenAI, Anthropic, and other providers through a single interface | CAID Section 3 (multi-model support) |
| click | >=8.1 | CLI entry point for running the framework | User requirement (CLI) |
| pyyaml | >=6.0 | Configuration file parsing (YAML config for repos, API keys, model selection) | User requirement (configurable) |
| gitpython | >=3.1.40 | Programmatic git operations (worktree, merge, commit, reset) | CAID Section 2.3-2.4 (git primitives) |

## Optional

| Package | Version | Purpose |
|---------|---------|---------|
| pytest | >=7.0 | Running test suites within worktrees (Commit0 self-verification) |
| rich | >=13.0 | Terminal output formatting for manager status display |
| tiktoken | >=0.5 | Token counting for context window management in LLMSummarizingCondenser |

## Development

| Package | Version | Purpose |
|---------|---------|---------|
| pytest | >=7.0 | Unit and integration tests for CAID itself |
| pytest-asyncio | >=0.23 | Testing async coroutines (manager loop, engineer loop) |
| mypy | >=1.8 | Static type checking |

## Conflicts

- None found. All packages are compatible with Python >=3.11.

## Notes

- **gitpython vs subprocess**: The paper uses raw git CLI commands (`git worktree add`, `git merge`). GitPython wraps these but for worktree operations specifically, we may need to fall back to `subprocess.run(["git", ...])` since GitPython's worktree support is limited. The implementation should use `subprocess` for worktree commands and GitPython for higher-level operations (commit, merge, diff).
- **litellm vs direct API clients**: Using litellm allows a single codebase to work with Claude Sonnet 4.5, GLM 4.7, MiniMax 2.5, and any other model. This matches the paper's multi-model evaluation.
- **No OpenHands dependency**: The paper uses OpenHands v1.11.0 as the agent substrate, but we are building a standalone implementation. The CAID coordination logic (dependency graph, worktree isolation, asyncio loop) is independent of OpenHands. LLM calls go through litellm directly.

## Install Command

```bash
pip install networkx>=3.1 pydantic>=2.0 litellm>=1.40 click>=8.1 pyyaml>=6.0 gitpython>=3.1.40
pip install pytest>=7.0 pytest-asyncio>=0.23 rich>=13.0 tiktoken>=0.5  # optional
```
