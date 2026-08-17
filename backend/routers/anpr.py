"""
backend/routers/anpr.py — POST /anpr/verify  (ANPR plate verification).

Flow
----
1. Accept image upload (multipart) + optional dispatch_id (Form field).
2. Save to temp file, run anpr.detect_plate.detect_plate().
3. Build response:
   a. Plate detected + IS in allowlist  → authorized, no action.
   b. Plate detected + NOT in allowlist → unauthorized:
        • Log a SecurityEvent with reason
          "unauthorized vehicle attempted green corridor".
        • If dispatch_id provided, call POST /dispatch/{id}/cancel
          (in-process DB update — no HTTP round-trip).
   c. No plate detected → return result with is_detected=False, no crash.

Threading note
--------------
Plain ``def`` — runs in FastAPI's thread pool.  EasyOCR inference and the
cancel DB write are blocking but safe here.
"""
from __future__ import annotations

import json
import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import SecurityEvent, SirenEvent
from backend.routers.ws import bus as _bus

log = logging.getLogger(__name__)
router = APIRouter(prefix="/anpr", tags=["ANPR / Plate Verification"])

UNAUTHORIZED_REASON = "unauthorized vehicle attempted green corridor"


# ── Internal: cancel a dispatch event in-process ──────────────────────────────

def _cancel_dispatch_event(dispatch_id: int, plate_text: str, db: Session) -> bool:
    """
    Mark a SirenEvent row as cancelled by appending a JSON note.
    Returns True if the row was found and updated.
    """
    event = db.query(SirenEvent).filter(SirenEvent.id == dispatch_id).first()
    if event is None:
        log.warning("cancel_dispatch: SirenEvent id=%d not found", dispatch_id)
        return False

    try:
        existing = json.loads(event.notes or "{}")
    except Exception:
        existing = {"original_notes": event.notes}

    existing["cancelled"] = True
    existing["cancelled_at"] = datetime.now(tz=timezone.utc).isoformat()
    existing["cancelled_reason"] = UNAUTHORIZED_REASON
    existing["unauthorized_plate"] = plate_text

    event.notes = json.dumps(existing, ensure_ascii=False)
    db.commit()
    log.warning(
        "Dispatch event id=%d CANCELLED — unauthorized plate: %s",
        dispatch_id, plate_text,
    )
    return True


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.post(
    "/verify",
    summary="Verify a vehicle's licence plate against the ambulance allowlist",
    description=(
        "Accepts an image upload (multipart/form-data) plus an optional "
        "`dispatch_id`.\n\n"
        "1. Runs OpenCV contour detection to find the plate region.\n"
        "2. Runs EasyOCR to read the plate text.\n"
        "3. Checks the normalized plate against the `AMBULANCE_ALLOWLIST`.\n\n"
        "If the plate is **not** in the allowlist:\n"
        "- Logs a `SecurityEvent` with reason "
        f'`"{UNAUTHORIZED_REASON}"`.\n'
        "- If `dispatch_id` is provided, cancels that dispatch in the DB.\n\n"
        "If **no plate is detected** in the image, returns gracefully with "
        "`is_detected=false` — no crash."
    ),
)
def verify_plate(
    file:        UploadFile    = File(..., description="Vehicle image (jpg/png/bmp)"),
    dispatch_id: int | None    = Form(default=None, description="SirenEvent id to cancel if unauthorized"),
    db:          Session       = Depends(get_db),
) -> dict:
    """Synchronous handler — safe for blocking EasyOCR inference."""

    # ── 1. Save upload to temp file ───────────────────────────────────────────
    suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file.file.read())
            tmp_path = tmp.name
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Failed to save upload: {exc}")

    # ── 2. Run plate detection ────────────────────────────────────────────────
    try:
        from anpr.detect_plate import detect_plate
        plate = detect_plate(tmp_path)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        log.error("Plate detection error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Detection error: {exc}")
    finally:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass

    log.info(
        "[/anpr/verify] file=%r  plate=%r  authorized=%s  vehicle=%s",
        file.filename, plate["plate_text"],
        plate["is_authorized"], plate["vehicle_type"],
    )

    # ── 3. Security enforcement for unauthorized plates ───────────────────────
    dispatch_cancelled = False
    security_event_id: int | None = None

    if plate["is_detected"] and plate["is_authorized"] is False:
        # Log security event
        reason_detail = (
            f"{UNAUTHORIZED_REASON} | "
            f"plate='{plate['plate_text']}' "
            f"confidence={plate['confidence']:.3f}"
        )
        try:
            sec = SecurityEvent(
                payload        = json.dumps({
                    "plate_text":   plate["plate_text"],
                    "raw_ocr":      plate["raw_ocr_text"],
                    "dispatch_id":  dispatch_id,
                    "vehicle_type": plate["vehicle_type"],
                }),
                verdict        = "BLOCKED",
                layer_blocked  = 0,            # ANPR layer (outside the 3-layer guard)
                blocked_reason = reason_detail[:512],
                endpoint       = "/anpr/verify",
            )
            db.add(sec)
            db.commit()
            db.refresh(sec)
            security_event_id = sec.id
            log.warning("[/anpr/verify] SecurityEvent id=%d logged: %s", sec.id, reason_detail)
        except Exception as exc:
            log.error("Failed to log security event: %s", exc)

        # Cancel dispatch if id provided
        if dispatch_id is not None:
            try:
                dispatch_cancelled = _cancel_dispatch_event(
                    dispatch_id, plate["plate_text"] or "", db
                )
            except Exception as exc:
                log.error("Dispatch cancel failed: %s", exc)

    # ── 4. Broadcast to live dashboard ───────────────────────────────────────
    if plate["is_detected"]:
        _bus.publish_sync({
            "type":    "anpr",
            "msg":     f"[ANPR] Plate {plate['plate_text']} — "
                       f"{'AUTHORIZED ' + plate['vehicle_type'] if plate['is_authorized'] else 'CIVILIAN — BLOCKED'}",
            "payload": {
                "plate_text":   plate["plate_text"],
                "authorized":   plate["is_authorized"],
                "vehicle_type": plate["vehicle_type"],
                "confidence":   plate["confidence"],
                "cancelled":    dispatch_cancelled,
            },
        })

    # ── 5. Build response ─────────────────────────────────────────────────────
    if not plate["is_detected"]:
        message = "No licence plate detected in the image — no action taken."
    elif plate["is_authorized"]:
        message = (
            f"Plate {plate['plate_text']} is AUTHORIZED "
            f"({plate['vehicle_type']}). Green corridor confirmed."
        )
    else:
        message = (
            f"Plate {plate['plate_text']} is NOT in the allowlist "
            f"(classified as '{plate['vehicle_type']}'). "
            + ("Dispatch cancelled. " if dispatch_cancelled else "")
            + "Security event logged."
        )

    return {
        "plate_text":          plate["plate_text"],
        "raw_ocr_text":        plate["raw_ocr_text"],
        "is_detected":         plate["is_detected"],
        "is_authorized":       plate["is_authorized"],
        "vehicle_type":        plate["vehicle_type"],
        "confidence":          plate["confidence"],
        "plate_region":        plate["plate_region"],
        "dispatch_cancelled":  dispatch_cancelled,
        "security_event_id":   security_event_id,
        "message":             message,
    }
