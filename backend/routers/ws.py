"""
backend/routers/ws.py — WebSocket broadcast hub for live event streaming.

Architecture
------------
A single in-process broadcast bus (`EventBus`) maintains a set of active
WebSocket connections.  Any backend code can call::

    from backend.routers.ws import bus
    await bus.publish({"type": "siren", "msg": "...", ...})

Each connected dashboard client receives every published message as JSON.

WebSocket endpoint
------------------
  GET /ws/logs  — persistent connection; server pushes events as they happen.

The client sends no data upward (read-only feed).  The server sends a
``{"type": "ping"}`` heartbeat every 20 s to keep the connection alive
through proxies and load balancers.

Thread-safety note
------------------
The handlers for /dispatch, /audio, /anpr are synchronous ``def`` functions
running in FastAPI's thread-pool executor.  They cannot call ``await``
directly.  Instead they use::

    asyncio.run_coroutine_threadsafe(bus.publish(...), bus.loop)

which safely schedules the coroutine on the event loop from the worker thread.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

log = logging.getLogger(__name__)
router = APIRouter(tags=["WebSocket"])

_HEARTBEAT_INTERVAL = 20   # seconds


# ── Broadcast bus ─────────────────────────────────────────────────────────────

class EventBus:
    """Thread-safe in-process broadcast hub for WebSocket clients."""

    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()
        self.loop: Optional[asyncio.AbstractEventLoop] = None

    def _register_loop(self) -> None:
        """Capture the running event loop (called from an async context)."""
        if self.loop is None:
            try:
                self.loop = asyncio.get_running_loop()
            except RuntimeError:
                pass

    async def connect(self, ws: WebSocket) -> None:
        self._register_loop()
        await ws.accept()
        self._clients.add(ws)
        log.info("WS client connected  (total=%d)", len(self._clients))

    def disconnect(self, ws: WebSocket) -> None:
        self._clients.discard(ws)
        log.info("WS client disconnected (total=%d)", len(self._clients))

    async def publish(self, payload: dict) -> None:
        """Broadcast *payload* to all connected clients (fire-and-forget)."""
        if not self._clients:
            return
        text = json.dumps(payload, ensure_ascii=False, default=str)
        dead: list[WebSocket] = []
        for ws in list(self._clients):
            try:
                await ws.send_text(text)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._clients.discard(ws)

    def publish_sync(self, payload: dict) -> None:
        """
        Thread-safe publish called from synchronous (thread-pool) handlers.
        Schedules the coroutine on the event loop captured at startup.
        """
        if self.loop is None or not self.loop.is_running():
            return
        asyncio.run_coroutine_threadsafe(self.publish(payload), self.loop)


# Module-level singleton — import this everywhere
bus = EventBus()


# ── WebSocket endpoint ────────────────────────────────────────────────────────

@router.websocket("/ws/logs")
async def ws_logs(websocket: WebSocket) -> None:
    """
    Persistent WebSocket feed.  The server pushes live JSON events; the
    client never needs to send anything.

    Message schema (all fields optional except ``type``):
    {
        "type":       "siren" | "anpr" | "dispatch" | "security" | "sim" | "ping",
        "msg":        "human-readable summary",
        "payload":    { ...full detail... },
        "ts":         "ISO-8601 timestamp"
    }
    """
    await bus.connect(websocket)
    # Announce connection to the new client
    await bus.publish({
        "type": "system",
        "msg":  "Connected to Siren AI live event feed.",
    })
    try:
        while True:
            # Send heartbeat; also detects dead connections
            await asyncio.sleep(_HEARTBEAT_INTERVAL)
            await websocket.send_text('{"type":"ping"}')
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        log.debug("WS connection closed: %s", exc)
    finally:
        bus.disconnect(websocket)
