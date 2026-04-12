"""Programmatic Python API for the CAID framework.

Provides async and sync entry points for running CAID from
external agents or application code.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any


async def run_caid(
    repo_path: str | Path,
    mode: str = "commit0",
    task_description: str = "",
    max_engineers: int = 4,
    model: str = "claude-sonnet-4-5-20260120",
    api_key: str = "",
    **kwargs: Any,
) -> dict[str, Any]:
    """Run the CAID framework asynchronously.

    This is the primary programmatic entry point. It creates a
    ManagerAgent with the given configuration and executes the
    full CAID lifecycle.

    Args:
        repo_path: Path to the target git repository.
        mode: Operating mode ('commit0' or 'paperbench').
        task_description: Free-text task description or paper path.
        max_engineers: Number of concurrent engineer agents.
        model: LLM model identifier (litellm format).
        api_key: LLM API key.
        **kwargs: Additional CAIDConfig fields.

    Returns:
        Summary dict with completed_tasks, total_tasks, and token_usage.

    Example:
        result = await run_caid(
            repo_path="./my-repo",
            mode="commit0",
            max_engineers=4,
            model="claude-sonnet-4-5-20260120",
            api_key="sk-...",
        )
    """
    from caid.config import CAIDConfig, LLMConfig
    from caid.manager import ManagerAgent

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


def run_caid_sync(
    repo_path: str | Path,
    mode: str = "commit0",
    task_description: str = "",
    max_engineers: int = 4,
    model: str = "claude-sonnet-4-5-20260120",
    api_key: str = "",
    **kwargs: Any,
) -> dict[str, Any]:
    """Run the CAID framework synchronously.

    Convenience wrapper around run_caid() for non-async contexts.
    Creates a new event loop if none is running.

    Args:
        Same as run_caid().

    Returns:
        Same as run_caid().
    """
    return asyncio.run(
        run_caid(
            repo_path=repo_path,
            mode=mode,
            task_description=task_description,
            max_engineers=max_engineers,
            model=model,
            api_key=api_key,
            **kwargs,
        )
    )


if __name__ == "__main__":
    # Verify API imports work
    print("run_caid: async entry point")
    print("run_caid_sync: synchronous entry point")
    print("API module OK")
