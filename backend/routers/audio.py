"""
backend/routers/audio.py — POST /audio/detect  (siren detection endpoint).

Flow
----
1. Accept a WAV file upload (multipart/form-data).
2. Save to a temp file and run audio_ml.infer.predict().
3. Return the detection result immediately.
4. If is_siren=True AND confidence > 0.8 AND the SUMO simulation is
   running (app.state.sim), trigger the full ambulance dispatch pipeline
   (Navigator + GridController) with a hardcoded demo route start="1",
   destination="2" — the same route used in manual /dispatch calls.

Response JSON
-------------
{
  "is_siren":        bool,
  "confidence":      float,
  "label":           str,           // 'siren' | 'traffic' | 'other'
  "all_proba":       dict,          // per-class probabilities
  "synthetic_model": bool,
  "dispatch_triggered": bool,
  "dispatch_result": dict | null,   // full dispatch output if triggered
  "message":         str            // human-readable summary for demo
}

Threading note
--------------
Plain ``def`` handler — FastAPI runs it in a thread pool so the
blocking librosa load + random forest inference + Gemini HTTP call
(if dispatch triggers) never stall the async event loop.
"""
from __future__ import annotations

import json
import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import SirenEvent
from backend.routers.ws import bus as _bus

log = logging.getLogger(__name__)
router = APIRouter(prefix="/audio", tags=["Audio / Siren Detection"])

# Confidence threshold above which an automatic dispatch is triggered
DISPATCH_CONFIDENCE_THRESHOLD = 0.80
# Hardcoded demo dispatch route
DEMO_START       = "1"
DEMO_DESTINATION = "2"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_infer():
    """Lazy-import predict() so missing model gives a clean 503, not a 500."""
    try:
        from audio_ml.infer import predict
        return predict
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc) + " — Run 'python -m audio_ml.train' first.",
        )


def _run_dispatch_pipeline(sim, db: Session) -> dict:
    """
    Run the MAS dispatch pipeline (Navigator + GridController) against
    the live SumoEnv.  Mirrors the logic in backend/routers/dispatch.py
    but called programmatically rather than via HTTP.
    """
    from agents.ambulance_navigator import AmbulanceNavigatorAgent
    from agents.grid_controller import GridControllerAgent

    try:
        sim_state = sim.get_state()
    except Exception as exc:
        return {"error": f"Could not read sim state: {exc}"}

    # ── Navigator (Gemini) ────────────────────────────────────────────────────
    navigator  = AmbulanceNavigatorAgent()
    nav_result = navigator.run_sync({
        "start":       DEMO_START,
        "destination": DEMO_DESTINATION,
        "sim_state":   sim_state,
    })
    route     = nav_result.get("route", [DEMO_START, "0", DEMO_DESTINATION])
    reasoning = nav_result.get("reasoning", "")
    model_used = nav_result.get("model", "unknown")

    # ── Grid Controller (ETA + TraCI) ─────────────────────────────────────────
    grid_ctrl   = GridControllerAgent()
    grid_result = grid_ctrl.run_sync({
        "route":       route,
        "start":       DEMO_START,
        "destination": DEMO_DESTINATION,
        "sim":         sim,
    })

    timing_log   = grid_result.get("timing_log", [])
    steps_taken  = grid_result.get("steps_taken", 0)
    completed    = grid_result.get("completed", False)

    # ── DB log ────────────────────────────────────────────────────────────────
    try:
        notes = json.dumps({
            "trigger":    "audio_detect",
            "start":      DEMO_START,
            "destination": DEMO_DESTINATION,
            "timestamp":  datetime.now(tz=timezone.utc).isoformat(),
            "navigator":  {"model": model_used, "route": route, "reasoning": reasoning},
            "grid":       {"timing_log": timing_log, "steps_taken": steps_taken,
                           "completed": completed},
        }, ensure_ascii=False)
        db.add(SirenEvent(
            siren_type="audio_detect_dispatch",
            confidence=f"{DEMO_START}->{DEMO_DESTINATION}",
            notes=notes,
        ))
        db.commit()
    except Exception as exc:
        log.error("DB log failed: %s", exc)

    return {
        "route":           route,
        "model_used":      model_used,
        "navigator_reasoning": reasoning,
        "timing_log":      timing_log,
        "steps_taken":     steps_taken,
        "completed":       completed,
    }


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.post(
    "/detect",
    summary="Upload a WAV file — detect siren and optionally auto-dispatch",
    description=(
        "Accepts a WAV audio file (multipart upload). Runs the siren classifier "
        "(RandomForest on 80-dim MFCCs). If `is_siren=true` and "
        f"`confidence > {DISPATCH_CONFIDENCE_THRESHOLD}` and the SUMO simulation "
        "is running, automatically triggers the full MAS ambulance dispatch "
        f"pipeline (demo route: junction {DEMO_START} → {DEMO_DESTINATION}).\n\n"
        "**Requires `python -m audio_ml.train` to have been run at least once.**"
    ),
)
def audio_detect(
    request: Request,
    file:    UploadFile = File(..., description="WAV audio file to classify"),
    db:      Session    = Depends(get_db),
) -> dict:
    """
    Synchronous handler — FastAPI runs this in a thread pool.
    All blocking I/O (librosa load, sklearn predict) is safe here.
    """
    predict = _load_infer()

    # ── 1. Save upload to temp file ───────────────────────────────────────────
    suffix = Path(file.filename or "upload.wav").suffix or ".wav"
    try:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=suffix,
            dir=tempfile.gettempdir(),
        ) as tmp:
            tmp.write(file.file.read())
            tmp_path = tmp.name
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Failed to save upload: {exc}")

    # ── 2. Run inference ──────────────────────────────────────────────────────
    try:
        result = predict(tmp_path)
    except Exception as exc:
        log.error("Inference error: %s", exc)
        raise HTTPException(status_code=422, detail=f"Audio inference failed: {exc}")
    finally:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass

    log.info(
        "[/audio/detect] file=%r is_siren=%s confidence=%.3f label=%r",
        file.filename, result["is_siren"], result["confidence"], result["label"],
    )

    # ── 3. Decide on dispatch ─────────────────────────────────────────────────
    dispatch_triggered = False
    dispatch_result    = None
    sim = getattr(request.app.state, "sim", None)
    sim_connected = sim is not None and getattr(sim, "is_connected", False)

    if result["is_siren"] and result["confidence"] > DISPATCH_CONFIDENCE_THRESHOLD:
        if sim_connected:
            log.info(
                "[/audio/detect] SIREN CONFIRMED (conf=%.3f) — triggering dispatch %s->%s",
                result["confidence"], DEMO_START, DEMO_DESTINATION,
            )
            dispatch_triggered = True
            dispatch_result = _run_dispatch_pipeline(sim, db)
        else:
            log.info(
                "[/audio/detect] SIREN CONFIRMED but sim not running — "
                "skipping auto-dispatch (call POST /sim/start first)"
            )

    # ── 4. Build summary message ──────────────────────────────────────────────
    if result["is_siren"] and result["confidence"] > DISPATCH_CONFIDENCE_THRESHOLD:
        if dispatch_triggered:
            msg = (
                f"Siren detected (confidence {result['confidence']:.1%}). "
                f"Green corridor dispatched: junction {DEMO_START} -> {DEMO_DESTINATION}."
            )
        else:
            msg = (
                f"Siren detected (confidence {result['confidence']:.1%}) "
                "but simulation is not running — start it with POST /sim/start "
                "to enable auto-dispatch."
            )
    elif result["is_siren"]:
        msg = (
            f"Possible siren detected but confidence {result['confidence']:.1%} "
            f"is below threshold {DISPATCH_CONFIDENCE_THRESHOLD:.0%} — no dispatch."
        )
    else:
        msg = f"No siren detected. Classified as '{result['label']}' ({result['confidence']:.1%})."

    if result.get("synthetic_model"):
        msg += "  [NOTE: model trained on synthetic data — not production-ready]"

    response = {
        "is_siren":           result["is_siren"],
        "confidence":         result["confidence"],
        "label":              result["label"],
        "all_proba":          result["all_proba"],
        "synthetic_model":    result["synthetic_model"],
        "dispatch_triggered": dispatch_triggered,
        "dispatch_result":    dispatch_result,
        "message":            msg,
    }

    # ── Broadcast to live dashboard ─────────────────────────────────────────
    _bus.publish_sync({
        "type":    "siren",
        "msg":     f"[Audio] {result['label'].title()} — {result['confidence']:.1%} confidence"
                   + ("  ✓ dispatch triggered" if dispatch_triggered else ""),
        "payload": {
            "is_siren":    result["is_siren"],
            "confidence":  result["confidence"],
            "label":       result["label"],
            "dispatched":  dispatch_triggered,
        },
    })

    return response
