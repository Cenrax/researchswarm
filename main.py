#!/usr/bin/env python3
"""Entry point for the Research Paper Implementation Swarm.

Usage:
    # Auto-detect papers from latest week folder (PDF/MD files or paper_ids.json)
    python main.py --objective "Build a transformer text classifier"

    # Specify arXiv paper IDs explicitly
    python main.py --objective "Build a transformer text classifier" \
                   --papers 2301.07041 1706.03762

    # Interactive mode — prompts for objective
    python main.py
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import get_latest_week_dir, discover_papers
from agents.director import run_director


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Research Paper Implementation Swarm"
    )
    parser.add_argument(
        "--objective",
        type=str,
        default=None,
        help="What you want to build from the papers.",
    )
    parser.add_argument(
        "--papers",
        nargs="*",
        default=None,
        help="arXiv paper IDs (e.g. 2301.07041 1706.03762). "
        "If omitted, auto-detects from input/<latest_week>/.",
    )
    args = parser.parse_args()

    # Get objective
    objective = args.objective
    if not objective:
        objective = input("Enter your objective (what to build from the papers): ").strip()
        if not objective:
            print("Error: objective is required.")
            sys.exit(1)

    # Get papers — explicit IDs or auto-discover from week folder
    if args.papers:
        # User passed arXiv IDs on the command line
        mode = "arxiv"
        papers = args.papers
        print(f"Using {len(papers)} arXiv paper ID(s) from command line")
    else:
        # Auto-discover from latest week folder
        week_dir = get_latest_week_dir()
        discovery = discover_papers(week_dir)
        mode = discovery["mode"]
        papers = discovery["papers"]

        if mode == "local":
            print(f"Found {len(papers)} paper file(s) in {week_dir.name}/:")
            for p in papers:
                print(f"  - {Path(p).name}")
        else:
            print(f"Loaded {len(papers)} arXiv paper ID(s) from {week_dir.name}/paper_ids.json")

    # Run the director
    asyncio.run(run_director(objective, papers, mode=mode))


if __name__ == "__main__":
    main()
