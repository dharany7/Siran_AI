"""
backend/routers/corridor.py
────────────────────────────
POST /corridor/sensor-trigger

End-to-end multi-modal emergency corridor pipeline.
1. Acoustic check (WAV/MP3) via MFCC + RandomForest (confidence > 0.8)
2. Vision check (JPG/PNG) via OpenCV + EasyOCR
3. If both pass -> TraCI preemption to ALL GREEN
4. Audit & Broadcast -> DB SecurityEvent + WebSocket SIGNAL_PREEMPTED
"""
from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import SecurityEvent
from backend.routers.ws import bus as _bus

log = logging.getLogger(__name__)

router = APIRouter(prefix="/corridor", tags=["Corridor"])

@router.post("/sensor-trigger")
def sensor_trigger(
    request: Request,
    junction_id: str = Form(...),
    audio_file: UploadFile = File(...),
    camera_frame: UploadFile = File(...),
    db: Session = Depends(get_db)
) -> dict:
    
    # Check Simulation
    sim = getattr(request.app.state, "sim", None)
    if sim is None or not getattr(sim, "is_connected", False):
        log.warning("corridor/sensor-trigger: simulation not running. Preemption will be skipped.")
    
    tmp_audio: Optional[Path] = None
    tmp_image: Optional[Path] = None

    try:
        # Save temp files
        a_suffix = Path(audio_file.filename or "audio.wav").suffix or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=a_suffix) as tmp_a:
            shutil.copyfileobj(audio_file.file, tmp_a)
            tmp_audio = Path(tmp_a.name)

        c_suffix = Path(camera_frame.filename or "frame.jpg").suffix or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=c_suffix) as tmp_c:
            shutil.copyfileobj(camera_frame.file, tmp_c)
            tmp_image = Path(tmp_c.name)

        # ── Step A: Acoustic Check ─────────────────────────────────────────────
        from audio_ml.infer import predict as _predict
        try:
            clf_result = _predict(tmp_audio)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Audio inference failed: {exc}")

        is_siren = clf_result.get("is_siren", False)
        confidence = clf_result.get("confidence", 0.0)

        if not is_siren or confidence < 0.80:
            msg = f"Acoustic check failed: is_siren={is_siren}, confidence={confidence:.2f} < 0.80"
            _log_event(db, "BLOCKED", 1, msg, "/corridor/sensor-trigger")
            return {"status": "BLOCKED", "reason": msg}

        # ── Step B: Vision Check (ANPR) ────────────────────────────────────────
        try:
            from anpr.detect_plate import detect_plate
            plate_res = detect_plate(str(tmp_image))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"ANPR inference failed: {exc}")

        is_authorized = plate_res.get("is_authorized", False)
        plate_text = plate_res.get("plate_text", "UNKNOWN")

        if not is_authorized:
            msg = f"Vision check failed: plate {plate_text} not authorized."
            _log_event(db, "BLOCKED", 2, msg, "/corridor/sensor-trigger")
            return {"status": "BLOCKED", "reason": msg, "plate": plate_text}

        # ── Step C: TraCI Preemption ───────────────────────────────────────────
        new_phase = None
        if sim and getattr(sim, "is_connected", False):
            try:
                current_phase = sim.get_traffic_light_state(junction_id)
                new_phase = "G" * len(current_phase)
                sim.set_traffic_light_state(junction_id, new_phase)
                log.info(f"Preempted junction {junction_id} to GREEN (was {current_phase})")
            except Exception as exc:
                log.error(f"TraCI preemption failed for junction {junction_id}: {exc}")
                msg = f"TraCI failed: {exc}"
                _log_event(db, "BLOCKED", 3, msg, "/corridor/sensor-trigger")
                return {"status": "ERROR", "reason": msg, "plate": plate_text}
        else:
            log.warning(f"Sim offline. Pretending to preempt junction {junction_id}")

        # ── Step D: Audit & Broadcast ──────────────────────────────────────────
        _log_event(db, "SAFE", None, None, "/corridor/sensor-trigger")
        
        _bus.publish_sync({
            "event": "SIGNAL_PREEMPTED",
            "junction_id": junction_id,
            "state": "GREEN",
            "plate": plate_text
        })

        return {
            "status": "PREEMPTED",
            "junction_id": junction_id,
            "plate": plate_text,
            "confidence": confidence,
            "new_phase": new_phase
        }

    finally:
        if tmp_audio and tmp_audio.exists():
            try: tmp_audio.unlink()
            except: pass
        if tmp_image and tmp_image.exists():
            try: tmp_image.unlink()
            except: pass


def _log_event(db: Session, verdict: str, layer: Optional[int], reason: Optional[str], endpoint: str):
    """Helper to write SecurityEvent to PostgreSQL"""
    try:
        ev = SecurityEvent(
            payload="Corridor multi-modal trigger",
            verdict=verdict,
            layer_blocked=layer,
            blocked_reason=reason,
            endpoint=endpoint
        )
        db.add(ev)
        db.commit()
    except Exception as exc:
        log.error(f"Failed to log SecurityEvent: {exc}")
        db.rollback()
