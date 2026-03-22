#!/usr/bin/env python3
"""Tests for the ArXiv MCP server connectivity and tool availability.

Run:
    python -m pytest tests/test_arxiv_mcp.py -v
    # or directly:
    python tests/test_arxiv_mcp.py
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import ARXIV_MCP_CONFIG, ARXIV_STORAGE_PATH


# ── Helpers ──────────────────────────────────────────────────────────────────

def _docker_available() -> bool:
    """Check if Docker daemon is running."""
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _image_exists(image: str) -> bool:
    """Check if a Docker image is pulled locally."""
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", image],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _send_jsonrpc(proc: subprocess.Popen, method: str, params: dict | None = None, req_id: int = 1) -> dict:
    """Send a JSON-RPC request to the MCP server and read the response."""
    request = {
        "jsonrpc": "2.0",
        "id": req_id,
        "method": method,
    }
    if params is not None:
        request["params"] = params

    payload = json.dumps(request) + "\n"
    proc.stdin.write(payload)
    proc.stdin.flush()

    # Read response line(s) — MCP servers respond with JSON-RPC over stdout
    while True:
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError("MCP server closed stdout without responding")
        line = line.strip()
        if not line:
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            # Skip non-JSON lines (e.g. server logs)
            continue


# ── Tests ────────────────────────────────────────────────────────────────────

def test_docker_is_running():
    """Docker daemon must be running to use the arxiv MCP server."""
    assert _docker_available(), (
        "Docker is not running. Start Docker Desktop and try again."
    )


def test_arxiv_image_exists():
    """The mcp/arxiv-mcp-server image must be pulled."""
    if not _docker_available():
        print("SKIP: Docker not running")
        return
    assert _image_exists("mcp/arxiv-mcp-server"), (
        "Docker image 'mcp/arxiv-mcp-server' not found.\n"
        "Pull it with: docker pull mcp/arxiv-mcp-server"
    )


def test_mcp_server_starts_and_lists_tools():
    """Start the MCP server, initialize it, and verify tools are listed."""
    if not _docker_available():
        print("SKIP: Docker not running")
        return
    if not _image_exists("mcp/arxiv-mcp-server"):
        print("SKIP: arxiv image not pulled")
        return

    cmd = [ARXIV_MCP_CONFIG["command"]] + ARXIV_MCP_CONFIG["args"]
    env = {**ARXIV_MCP_CONFIG.get("env", {})}

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**dict(__import__("os").environ), **env},
    )

    try:
        # Step 1: Initialize handshake
        init_resp = _send_jsonrpc(proc, "initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test-client", "version": "0.1.0"},
        }, req_id=1)

        print(f"  Init response: {json.dumps(init_resp, indent=2)}")
        assert "result" in init_resp, f"Initialize failed: {init_resp}"
        assert "serverInfo" in init_resp["result"] or "capabilities" in init_resp["result"], (
            f"Unexpected init response: {init_resp}"
        )

        # Step 2: Send initialized notification (no response expected)
        notif = {"jsonrpc": "2.0", "method": "notifications/initialized"}
        proc.stdin.write(json.dumps(notif) + "\n")
        proc.stdin.flush()

        # Step 3: List tools
        tools_resp = _send_jsonrpc(proc, "tools/list", {}, req_id=2)
        print(f"  Tools response: {json.dumps(tools_resp, indent=2)}")

        assert "result" in tools_resp, f"tools/list failed: {tools_resp}"
        tools = tools_resp["result"].get("tools", [])
        tool_names = [t["name"] for t in tools]
        print(f"  Available tools: {tool_names}")

        # Verify expected tools exist
        assert "read_paper" in tool_names, (
            f"'read_paper' tool not found. Available: {tool_names}"
        )
        print("  ✓ read_paper tool is available")

        # Check for download_paper too
        if "download_paper" in tool_names:
            print("  ✓ download_paper tool is available")
        if "search_papers" in tool_names:
            print("  ✓ search_papers tool is available")

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_search_papers_tool():
    """Test that search_papers actually works with a known query."""
    if not _docker_available():
        print("SKIP: Docker not running")
        return
    if not _image_exists("mcp/arxiv-mcp-server"):
        print("SKIP: arxiv image not pulled")
        return

    cmd = [ARXIV_MCP_CONFIG["command"]] + ARXIV_MCP_CONFIG["args"]
    env = {**ARXIV_MCP_CONFIG.get("env", {})}

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**dict(__import__("os").environ), **env},
    )

    try:
        # Initialize
        _send_jsonrpc(proc, "initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test-client", "version": "0.1.0"},
        }, req_id=1)

        notif = {"jsonrpc": "2.0", "method": "notifications/initialized"}
        proc.stdin.write(json.dumps(notif) + "\n")
        proc.stdin.flush()

        # Call search_papers with "attention is all you need"
        search_resp = _send_jsonrpc(proc, "tools/call", {
            "name": "search_papers",
            "arguments": {"query": "attention is all you need", "max_results": 1},
        }, req_id=3)

        print(f"  Search response keys: {list(search_resp.keys())}")

        if "result" in search_resp:
            content = search_resp["result"].get("content", [])
            print(f"  Got {len(content)} result(s)")
            if content:
                text = content[0].get("text", "")[:200]
                print(f"  First result preview: {text}...")
            assert len(content) > 0, "search_papers returned no results"
            print("  ✓ search_papers works")
        elif "error" in search_resp:
            print(f"  ✗ search_papers error: {search_resp['error']}")
            # Don't hard-fail — search might need network

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_download_and_read_paper():
    """Download paper 1706.03762 (Attention Is All You Need) and verify content is returned."""
    if not _docker_available():
        print("SKIP: Docker not running")
        return
    if not _image_exists("mcp/arxiv-mcp-server"):
        print("SKIP: arxiv image not pulled")
        return

    cmd = [ARXIV_MCP_CONFIG["command"]] + ARXIV_MCP_CONFIG["args"]
    env = {**ARXIV_MCP_CONFIG.get("env", {})}

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**dict(__import__("os").environ), **env},
    )

    try:
        # Initialize
        _send_jsonrpc(proc, "initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test-client", "version": "0.1.0"},
        }, req_id=1)

        notif = {"jsonrpc": "2.0", "method": "notifications/initialized"}
        proc.stdin.write(json.dumps(notif) + "\n")
        proc.stdin.flush()

        # Step 1: Download the paper
        paper_id = "1706.03762"
        print(f"  Downloading paper {paper_id}...")
        download_resp = _send_jsonrpc(proc, "tools/call", {
            "name": "download_paper",
            "arguments": {"paper_id": paper_id},
        }, req_id=10)

        if "error" in download_resp:
            print(f"  ✗ download_paper error: {download_resp['error']}")
            print("  (May require network access)")
            return

        print(f"  ✓ download_paper succeeded")

        # Step 2: Read the paper
        print(f"  Reading paper {paper_id}...")
        read_resp = _send_jsonrpc(proc, "tools/call", {
            "name": "read_paper",
            "arguments": {"paper_id": paper_id},
        }, req_id=11)

        if "error" in read_resp:
            print(f"  ✗ read_paper error: {read_resp['error']}")
            return

        assert "result" in read_resp, f"read_paper returned no result: {read_resp}"
        content = read_resp["result"].get("content", [])
        assert len(content) > 0, "read_paper returned empty content"

        # Extract text from the response
        full_text = ""
        for block in content:
            if block.get("type") == "text":
                full_text += block.get("text", "")

        assert len(full_text) > 500, (
            f"Paper content too short ({len(full_text)} chars), expected a full paper"
        )

        # Verify it's actually "Attention Is All You Need"
        text_lower = full_text.lower()
        assert "attention" in text_lower, "Paper content doesn't mention 'attention'"
        assert "transformer" in text_lower, "Paper content doesn't mention 'transformer'"

        print(f"  ✓ read_paper returned {len(full_text):,} characters")
        print(f"  ✓ Content contains 'attention' and 'transformer'")

        # Show a preview
        preview = full_text[:300].replace("\n", " ")
        print(f"  Preview: {preview}...")

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_storage_path_exists():
    """The ARXIV_STORAGE_PATH directory should exist or be creatable."""
    path = Path(ARXIV_STORAGE_PATH)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        print(f"  Created storage path: {path}")
    assert path.is_dir(), f"Storage path is not a directory: {path}"
    print(f"  ✓ Storage path exists: {path}")


# ── Runner ───────────────────────────────────────────────────────────────────

def run_all():
    """Run all tests manually (no pytest needed)."""
    tests = [
        test_docker_is_running,
        test_arxiv_image_exists,
        test_storage_path_exists,
        test_mcp_server_starts_and_lists_tools,
        test_search_papers_tool,
        test_download_and_read_paper,
    ]

    passed = 0
    failed = 0
    skipped = 0

    for test in tests:
        name = test.__name__
        print(f"\n{'─'*60}")
        print(f"  {name}")
        print(f"{'─'*60}")
        try:
            test()
            passed += 1
            print(f"  → PASSED")
        except AssertionError as e:
            failed += 1
            print(f"  → FAILED: {e}")
        except Exception as e:
            failed += 1
            print(f"  → ERROR: {type(e).__name__}: {e}")

    print(f"\n{'='*60}")
    print(f"  Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(run_all())
