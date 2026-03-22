"""Shared configuration for the research swarm."""

import os
import re
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"

# These are set dynamically per project via create_project_output()
SUMMARIES_DIR: Path = OUTPUT_DIR / "summaries"
PLANS_DIR: Path = OUTPUT_DIR / "plans"
CODE_DIR: Path = OUTPUT_DIR / "code"
REVIEWS_DIR: Path = OUTPUT_DIR / "reviews"

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


def _slugify(text: str) -> str:
    """Convert text to a filesystem-safe slug."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "-", text)
    return text[:60].rstrip("-")


def create_project_output(objective: str) -> dict[str, Path]:
    """Create a timestamped project subfolder under output/.

    Folder name: <timestamp>_<objective-slug>
    e.g. output/20260322_143021_build-a-transformer-text-classifier/

    Returns a dict with the project paths and updates the module-level
    SUMMARIES_DIR, PLANS_DIR, CODE_DIR, REVIEWS_DIR.
    """
    global SUMMARIES_DIR, PLANS_DIR, CODE_DIR, REVIEWS_DIR

    slug = _slugify(objective)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    project_dir = OUTPUT_DIR / f"{timestamp}_{slug}"

    SUMMARIES_DIR = project_dir / "summaries"
    PLANS_DIR = project_dir / "plans"
    CODE_DIR = project_dir / "code"
    REVIEWS_DIR = project_dir / "reviews"

    for d in (SUMMARIES_DIR, PLANS_DIR, CODE_DIR, REVIEWS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    return {
        "project_dir": project_dir,
        "summaries": SUMMARIES_DIR,
        "plans": PLANS_DIR,
        "code": CODE_DIR,
        "reviews": REVIEWS_DIR,
    }


def list_projects() -> list[Path]:
    """List all existing project folders under output/, newest first."""
    if not OUTPUT_DIR.exists():
        return []
    return sorted(
        [d for d in OUTPUT_DIR.iterdir() if d.is_dir()],
        reverse=True,
    )
