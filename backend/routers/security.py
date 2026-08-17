"""
backend/routers/security.py — Security & guard endpoints.

Endpoints
---------
POST /security/test-injection
    Accepts {"text": "..."}, runs it through the 3-layer PromptGuard,
    logs the result to the security_events table, and returns:
        {"blocked": bool, "reason": str, "layer": int, "llm_verdict": str | null}

GET  /security/events
    Returns the last N rows from security_events (most-recent-first).
    Useful for the dashboard feed.

Threading note
--------------
Both handlers are synchronous (plain ``def``). FastAPI runs them in a
thread pool, keeping the event loop free during the Gemini network call
inside Layer 3.
"""
from __future__ import annotations

import logging
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import SecurityEvent
from security.guard import guard   # module-level singleton
from backend.routers.ws import bus as _bus

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/security", tags=["Security"])


# ── Request / response schemas ────────────────────────────────────────────────

class InjectionTestRequest(BaseModel):
    text: str = Field(..., description="Free-text string to run through the guard")


class InjectionTestResponse(BaseModel):
    blocked:     bool
    reason:      str
    layer:       int                   # 0=safe, 1/2/3=blocking layer
    llm_verdict: str | None = None     # 'SAFE', 'BLOCKED', or 'UNAVAILABLE'
    event_id:    int | None = None     # security_events row id


class SecurityEventOut(BaseModel):
    id:             int
    payload:        str | None
    verdict:        str
    layer_blocked:  int | None
    blocked_reason: str | None
    endpoint:       str | None
    created_at:     datetime

    class Config:
        from_attributes = True


# ── Helpers ───────────────────────────────────────────────────────────────────

def _log_security_event(
    db:       Session,
    payload:  str,
    verdict:  str,
    layer:    int | None,
    reason:   str | None,
    endpoint: str,
) -> int:
    """Persist a SecurityEvent row and return its id."""
    row = SecurityEvent(
        payload        = payload[:2048],            # cap stored text
        verdict        = verdict,
        layer_blocked  = layer if layer else None,
        blocked_reason = reason if verdict == "BLOCKED" else None,
        endpoint       = endpoint,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row.id


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post(
    "/test-injection",
    response_model=InjectionTestResponse,
    summary="Test a text string against the 3-layer prompt-injection guard",
    description=(
        "Runs the supplied text through the full PromptGuard pipeline:\n\n"
        "1. **Layer 1** — schema/length validation\n"
        "2. **Layer 2** — regex heuristic (15+ known injection patterns)\n"
        "3. **Layer 3** — isolated Gemini call: 'SAFE' or 'BLOCKED'\n\n"
        "The result is logged to the `security_events` table.\n\n"
        "**Intended for demo and testing only** — use the guard implicitly "
        "via `POST /dispatch` in production."
    ),
)
def test_injection(
    body: InjectionTestRequest,
    db:   Session = Depends(get_db),
) -> InjectionTestResponse:
    """Run the full 3-layer guard against a user-supplied string."""
    text = body.text

    logger.info("[/security/test-injection] Checking: %.80r", text)

    result = guard.check(text)

    verdict = "BLOCKED" if result.blocked else "SAFE"
    event_id = _log_security_event(
        db       = db,
        payload  = text,
        verdict  = verdict,
        layer    = result.layer if result.blocked else None,
        reason   = result.reason if result.blocked else None,
        endpoint = "/security/test-injection",
    )

    logger.info(
        "[/security/test-injection] verdict=%s layer=%d reason=%r event_id=%d",
        verdict, result.layer, result.reason, event_id,
    )

    # Broadcast to live dashboard
    _bus.publish_sync({
        "type":    "security",
        "msg":     f"[Guard] {'BLOCKED' if result.blocked else 'SAFE'} — "
                   f"layer {result.layer}: {result.reason[:80] if result.reason else 'ok'}",
        "payload": {
            "blocked":     result.blocked,
            "layer":       result.layer,
            "reason":      result.reason,
            "llm_verdict": result.llm_verdict,
            "event_id":    event_id,
        },
    })

    return InjectionTestResponse(
        blocked     = result.blocked,
        reason      = result.reason,
        layer       = result.layer,
        llm_verdict = result.llm_verdict,
        event_id    = event_id,
    )


@router.get(
    "/events",
    response_model=list[SecurityEventOut],
    summary="Fetch recent security guard events",
)
def get_security_events(
    limit: int = 50,
    db:    Session = Depends(get_db),
) -> list[SecurityEventOut]:
    """Return the most-recent security guard events (default: last 50)."""
    rows = (
        db.query(SecurityEvent)
        .order_by(SecurityEvent.id.desc())
        .limit(min(limit, 200))
        .all()
    )
    return rows
