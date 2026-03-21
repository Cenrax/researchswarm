"""Shared configuration for the research swarm."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"
SUMMARIES_DIR = OUTPUT_DIR / "summaries"
PLANS_DIR = OUTPUT_DIR / "plans"
CODE_DIR = OUTPUT_DIR / "code"
REVIEWS_DIR = OUTPUT_DIR / "reviews"

# ── ArXiv MCP Server ────────────────────────────────────────────────────────
ARXIV_STORAGE_PATH = os.getenv(
    "ARXIV_STORAGE_PATH", "/Users/local-test/papers"
)

ARXIV_MCP_CONFIG = {
    "command": "docker",
    "args": [
        "run",
        "-i",
        "--rm",
        "-e",
        "ARXIV_STORAGE_PATH",
        "-v",
        f"{ARXIV_STORAGE_PATH}:{ARXIV_STORAGE_PATH}",
        "mcp/arxiv-mcp-server",
    ],
    "env": {
        "ARXIV_STORAGE_PATH": ARXIV_STORAGE_PATH,
    },
}

# ── Model choices ────────────────────────────────────────────────────────────
DIRECTOR_MODEL = "claude-opus-4-6"
READER_MODEL = "sonnet"
PLANNER_MODEL = "opus"
CODER_MODEL = "opus"
REVIEWER_MODEL = "sonnet"


PAPER_EXTENSIONS = {".pdf", ".md", ".txt", ".tex"}


def get_latest_week_dir() -> Path:
    """Return the input sub-folder with the highest week_id."""
    week_dirs = sorted(
        [d for d in INPUT_DIR.iterdir() if d.is_dir() and d.name.startswith("week_")],
        key=lambda d: int(d.name.split("_")[1]),
    )
    if not week_dirs:
        raise FileNotFoundError("No week_* folders found in input/")
    return week_dirs[-1]


def discover_papers(week_dir: Path) -> dict:
    """Discover papers in the week folder.

    Returns a dict with:
        mode: "local" if paper files found, "arxiv" if paper_ids.json found
        papers: list of file paths (local mode) or paper IDs (arxiv mode)
    """
    # Check for local paper files first
    local_papers = sorted(
        f for f in week_dir.iterdir()
        if f.is_file() and f.suffix.lower() in PAPER_EXTENSIONS
    )
    if local_papers:
        return {
            "mode": "local",
            "papers": [str(p) for p in local_papers],
        }

    # Fall back to paper_ids.json for arXiv MCP mode
    ids_file = week_dir / "paper_ids.json"
    if ids_file.exists():
        import json
        with open(ids_file) as f:
            paper_ids = json.load(f)
        return {
            "mode": "arxiv",
            "papers": paper_ids,
        }

    raise FileNotFoundError(
        f"No papers found in {week_dir}. "
        f"Place PDF/MD/TXT files or a paper_ids.json in the folder."
    )


def ensure_output_dirs() -> None:
    """Create output directories if they don't exist."""
    for d in (SUMMARIES_DIR, PLANS_DIR, CODE_DIR, REVIEWS_DIR):
        d.mkdir(parents=True, exist_ok=True)
