"""Click CLI entry point for the CAID framework.

Provides commands:
  - caid run: Execute the full CAID multi-agent framework
  - caid graph: Display the dependency graph (for debugging)
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

import click

from caid.config import CAIDConfig


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
def cli(verbose: bool) -> None:
    """CAID - Centralized Asynchronous Isolated Delegation framework."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        stream=sys.stderr,
    )


@cli.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    help="Path to YAML config file",
)
@click.option("--repo", type=click.Path(exists=True), help="Repository path")
@click.option(
    "--mode",
    type=click.Choice(["commit0", "paperbench"]),
    default="commit0",
    help="Operating mode",
)
@click.option(
    "--engineers", type=int, default=None, help="Number of engineer agents"
)
@click.option("--task", type=str, default="", help="Task description")
def run(
    config: str | None,
    repo: str | None,
    mode: str,
    engineers: int | None,
    task: str,
) -> None:
    """Run the CAID multi-agent framework."""
    from caid.manager import ManagerAgent

    if config:
        cfg = CAIDConfig.from_yaml(config)
    else:
        kwargs: dict = {"mode": mode, "task_description": task}
        if repo:
            kwargs["repo_path"] = Path(repo)
        if engineers is not None:
            kwargs["max_engineers"] = engineers
        cfg = CAIDConfig(**kwargs)

    click.echo(f"CAID starting: mode={cfg.mode}, engineers={cfg.max_engineers}")

    manager = ManagerAgent(cfg)
    result = asyncio.run(manager.run())

    click.echo("\n=== CAID Run Complete ===")
    click.echo(
        f"Tasks completed: {result['completed_tasks']}/{result['total_tasks']}"
    )
    click.echo(f"Token usage: {result['token_usage']}")


@cli.command()
@click.option(
    "--repo",
    type=click.Path(exists=True),
    required=True,
    help="Repository path",
)
def graph(repo: str) -> None:
    """Display the dependency graph for a repository (Commit0 mode)."""
    from caid.commit0.extractor import extract_dependencies

    g = extract_dependencies(Path(repo))

    click.echo(f"Tasks: {len(g.all_tasks)}")
    click.echo(f"Edges: {g.edge_count}")
    click.echo(f"Topological order:")

    for task_id in g.topological_order():
        node = g.get_node_data(task_id)
        deps = g.predecessors(task_id)
        dep_str = f" (depends on: {deps})" if deps else ""
        click.echo(
            f"  {task_id}: "
            f"functions={node.functions}, "
            f"complexity={node.complexity}"
            f"{dep_str}"
        )


def main() -> None:
    """CLI entry point."""
    cli()


if __name__ == "__main__":
    main()
