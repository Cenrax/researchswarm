"""FastAPI backend for the Research Swarm UI.

Run:
    uvicorn server:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import (
    INPUT_DIR,
    OUTPUT_DIR,
    get_latest_week_dir,
    discover_papers,
    create_project_output,
    list_projects,
    PAPER_EXTENSIONS,
)
from events import EventEmitter, set_emitter, get_emitter

app = FastAPI(title="Research Swarm")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Track running pipelines: project_dir_name → task
_running_pipelines: dict[str, asyncio.Task] = {}


# ── REST Endpoints ───────────────────────────────────────────────────────────

@app.post("/api/run")
async def start_pipeline(body: dict):
    """Start the research pipeline with a given objective."""
    objective = body.get("objective", "").strip()
    if not objective:
        return JSONResponse({"error": "objective is required"}, status_code=400)

    # Discover papers
    try:
        week_dir = get_latest_week_dir()
        discovery = discover_papers(week_dir)
    except FileNotFoundError as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    # Create project output
    paths = create_project_output(objective)
    project_id = paths["project_dir"].name

    # Set up emitter
    emitter = EventEmitter()
    set_emitter(emitter)

    # Run pipeline in background
    task = asyncio.create_task(
        _run_pipeline(objective, discovery["papers"], discovery["mode"], emitter)
    )
    _running_pipelines[project_id] = task

    return {
        "project_id": project_id,
        "mode": discovery["mode"],
        "papers": discovery["papers"],
        "paths": {k: str(v) for k, v in paths.items()},
    }


async def _run_pipeline(
    objective: str, papers: list[str], mode: str, emitter: EventEmitter
) -> None:
    """Run the director pipeline with the UI event emitter."""
    from agents.director import run_director_ui

    try:
        await run_director_ui(objective, papers, mode, emitter)
    except Exception as e:
        await emitter.emit({"type": "log", "level": "error", "message": str(e)})


@app.post("/api/upload")
async def upload_papers(files: list[UploadFile] = File(...)):
    """Upload PDF/MD/TXT files to the latest week folder."""
    # Auto-create next week folder
    try:
        week_dir = get_latest_week_dir()
    except FileNotFoundError:
        week_dir = INPUT_DIR / "week_1"
        week_dir.mkdir(parents=True, exist_ok=True)

    uploaded = []
    for f in files:
        suffix = Path(f.filename or "").suffix.lower()
        if suffix not in PAPER_EXTENSIONS:
            continue
        dest = week_dir / f.filename
        with open(dest, "wb") as out:
            content = await f.read()
            out.write(content)
        uploaded.append({"name": f.filename, "path": str(dest), "size": len(content)})

    return {"uploaded": uploaded, "week": week_dir.name}


@app.get("/api/projects")
async def get_projects():
    """List all past project runs."""
    projects = []
    for d in list_projects():
        # Parse timestamp and slug from folder name
        parts = d.name.split("_", 2)
        projects.append({
            "id": d.name,
            "path": str(d),
            "created": f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else d.name,
            "objective": parts[2].replace("-", " ") if len(parts) > 2 else d.name,
            "stages": {
                "summaries": len(list((d / "summaries").glob("*"))) if (d / "summaries").exists() else 0,
                "plans": len(list((d / "plans").glob("*"))) if (d / "plans").exists() else 0,
                "code": len(list((d / "code").glob("*"))) if (d / "code").exists() else 0,
                "reviews": len(list((d / "reviews").glob("*"))) if (d / "reviews").exists() else 0,
            },
        })
    return {"projects": projects}


@app.get("/api/projects/{project_id}/files")
async def get_file_tree(project_id: str):
    """Get the file tree for a project."""
    project_dir = OUTPUT_DIR / project_id
    if not project_dir.exists():
        return JSONResponse({"error": "Project not found"}, status_code=404)

    def build_tree(directory: Path, prefix: str = "") -> list[dict]:
        items = []
        for entry in sorted(directory.iterdir()):
            relative = str(entry.relative_to(project_dir))
            if entry.is_dir():
                items.append({
                    "name": entry.name,
                    "path": relative,
                    "type": "directory",
                    "children": build_tree(entry, relative),
                })
            else:
                items.append({
                    "name": entry.name,
                    "path": relative,
                    "type": "file",
                    "size": entry.stat().st_size,
                })
        return items

    return {"tree": build_tree(project_dir)}


@app.get("/api/projects/{project_id}/file")
async def get_file_content(project_id: str, path: str):
    """Read file content from a project."""
    project_dir = OUTPUT_DIR / project_id
    file_path = project_dir / path

    # Prevent path traversal
    try:
        file_path.resolve().relative_to(project_dir.resolve())
    except ValueError:
        return JSONResponse({"error": "Invalid path"}, status_code=400)

    if not file_path.exists():
        return JSONResponse({"error": "File not found"}, status_code=404)

    try:
        content = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        content = f"[Binary file: {file_path.name}]"

    return {"path": path, "name": file_path.name, "content": content}


@app.get("/api/papers")
async def get_papers():
    """List papers in the latest week folder."""
    try:
        week_dir = get_latest_week_dir()
        discovery = discover_papers(week_dir)
        papers = []
        if discovery["mode"] == "local":
            for p in discovery["papers"]:
                fp = Path(p)
                papers.append({"name": fp.name, "path": str(fp), "size": fp.stat().st_size})
        else:
            for pid in discovery["papers"]:
                papers.append({"name": pid, "path": pid, "size": 0})
        return {"week": week_dir.name, "mode": discovery["mode"], "papers": papers}
    except FileNotFoundError:
        return {"week": None, "mode": None, "papers": []}


# ── WebSocket ────────────────────────────────────────────────────────────────

@app.websocket("/ws/{project_id}")
async def websocket_endpoint(websocket: WebSocket, project_id: str):
    """WebSocket for real-time pipeline events."""
    await websocket.accept()

    emitter = get_emitter()
    if not emitter:
        await websocket.send_json({"type": "error", "message": "No active pipeline"})
        await websocket.close()
        return

    queue = await emitter.subscribe()

    # Two concurrent tasks: send events to UI, receive responses from UI
    async def send_events():
        try:
            while True:
                event = await queue.get()
                await websocket.send_json(event)
                if event.get("type") == "complete":
                    break
        except Exception:
            pass

    async def receive_responses():
        try:
            while True:
                data = await websocket.receive_json()
                if data.get("type") == "approval_response":
                    emitter.resolve_approval(
                        data.get("id", ""),
                        data.get("answer", ""),
                    )
        except WebSocketDisconnect:
            pass
        except Exception:
            pass

    try:
        await asyncio.gather(send_events(), receive_responses())
    finally:
        await emitter.unsubscribe(queue)
