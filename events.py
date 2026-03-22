"""Event emitter bridge between the agent pipeline and the UI.

Converts pipeline messages into structured JSON events that can be
pushed over WebSocket to the frontend.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any


class EventEmitter:
    """Thread-safe event emitter for pipeline → UI communication."""

    def __init__(self) -> None:
        self._subscribers: list[asyncio.Queue[dict[str, Any]]] = []
        self._lock = asyncio.Lock()
        # For AskUserQuestion: approval_id → asyncio.Event + answer storage
        self._pending_approvals: dict[str, dict] = {}
        self._approval_counter = 0

    async def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        """Create a new subscriber queue."""
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        async with self._lock:
            self._subscribers.append(queue)
        return queue

    async def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        """Remove a subscriber queue."""
        async with self._lock:
            self._subscribers = [q for q in self._subscribers if q is not queue]

    async def emit(self, event: dict[str, Any]) -> None:
        """Broadcast an event to all subscribers."""
        event["timestamp"] = datetime.now().strftime("%H:%M:%S")
        async with self._lock:
            for queue in self._subscribers:
                await queue.put(event)

    async def request_approval(
        self, question: str, options: list[dict] | None = None, header: str = ""
    ) -> str:
        """Send an approval request to the UI and wait for response."""
        self._approval_counter += 1
        approval_id = f"approval_{self._approval_counter}"

        event_obj = asyncio.Event()
        self._pending_approvals[approval_id] = {
            "event": event_obj,
            "answer": "",
        }

        await self.emit({
            "type": "approval_needed",
            "id": approval_id,
            "question": question,
            "header": header,
            "options": options or [],
        })

        # Wait for the UI to respond
        await event_obj.wait()

        answer = self._pending_approvals[approval_id]["answer"]
        del self._pending_approvals[approval_id]
        return answer

    def resolve_approval(self, approval_id: str, answer: str) -> bool:
        """Resolve a pending approval (called when UI sends response)."""
        if approval_id not in self._pending_approvals:
            return False
        self._pending_approvals[approval_id]["answer"] = answer
        self._pending_approvals[approval_id]["event"].set()
        return True


# Global emitter instance — None when running in terminal mode
_emitter: EventEmitter | None = None


def get_emitter() -> EventEmitter | None:
    return _emitter


def set_emitter(emitter: EventEmitter) -> None:
    global _emitter
    _emitter = emitter
