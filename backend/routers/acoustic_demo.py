"""
backend/routers/acoustic_demo.py
─────────────────────────────────
POST /junction/{junction_id}/acoustic-event

Simulated acoustic-detection pipeline that uses *real inference* at every
stage but pre-recorded clip files instead of live microphone input.

Pipeline stages
───────────────
1. mic_heard          — clip file resolved and read into a temp file
2. siren_classified   — RandomForest (audio_ml.infer.predict) on 80-dim MFCCs
3. camera_triggered   — picks demo-assets/plates/plate_demo.jpg
   anpr_confirmed     — OpenCV + EasyOCR (anpr.detect_plate.detect_plate)
4. signal_flipped     — sim.set_traffic_light_state(junction_id, all-green)
                        only if SUMO is running AND is_siren AND conf>=0.75
                        AND is_authorized

Each stage is broadcast to /ws/logs as an "acoustic_demo" event *before*
the HTTP response is returned, so the ops dashboard updates live.

WS event shape (matches the shape produced by /dispatch — same dashboard)
─────────────────────────────────────────────────────────────────────────
{
  "type":    "acoustic_demo",
  "msg":     "<human-readable one-liner>",
  "payload": {
    "stage":        "mic_heard|siren_classified|camera_triggered|
                     anpr_confirmed|signal_flipped",
    "junction_id":  "<str>",
    "is_siren":     <bool|null>,
    "confidence":   <float|null>,
    "label":        "<str|null>",
    "plate_text":   "<str|null>",
    "is_authorized":<bool|null>,
    "signal_green": <bool|null>
  }
}

Helper endpoints
────────────────
GET /junction/demo-assets/list      — available sample filenames
GET /junction/demo-assets/junctions — TLS junction IDs from running SUMO
                                      (falls back to hardcoded list)

Threading note
──────────────
Plain ``def`` handler — FastAPI runs it in a thread pool.  All blocking
calls (librosa load, sklearn predict, OpenCV, EasyOCR, TraCI) are safe.
"""
from __future__ import annotations

import logging
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from backend.routers.ws import bus as _bus

log = logging.getLogger(__name__)

router = APIRouter(prefix="/junction", tags=["Acoustic Demo"])

# ── Demo-asset paths ──────────────────────────────────────────────────────────
_ROOT        = Path(__file__).parent.parent.parent   # project root
DEMO_SIRENS  = _ROOT / "demo-assets" / "sirens"
DEMO_PLATES  = _ROOT / "demo-assets" / "plates"

# Siren detection threshold for the acoustic-demo path
SIREN_CONFIDENCE_THRESHOLD = 0.75

# Fallback junction list when SUMO is not running
_FALLBACK_JUNCTIONS = [
    "cluster_245880890_296583389_52050268_52050270",
    "cluster_296583414_52050264_52050266",
    "252373097",
    "296583462",
    "52050258",
]


# ── WebSocket helpers ─────────────────────────────────────────────────────────

def _emit(stage: str, junction_id: str, msg: str, **extra) -> None:
    """Broadcast one acoustic-demo stage event to all WS clients."""
    payload = {"stage": junction_id, **extra}
    payload["stage"] = stage
    payload["junction_id"] = junction_id
    _bus.publish_sync({
        "type":    "acoustic_demo",
        "msg":     msg,
        "payload": payload,
        "ts":      datetime.now(tz=timezone.utc).isoformat(),
    })


# ── Helper endpoints ──────────────────────────────────────────────────────────

@router.get(
    "/demo-assets/list",
    summary="List pre-canned demo audio sample names",
    description=(
        "Returns the filenames available in ``demo-assets/sirens/``.  "
        "Pass one of these as ``sample_name`` to the acoustic-event endpoint."
    ),
)
def list_demo_samples() -> dict:
    if not DEMO_SIRENS.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                "demo-assets/sirens/ not found. "
                "Run 'python setup_demo_assets.py' first."
            ),
        )
    files = sorted(p.name for p in DEMO_SIRENS.glob("*.wav"))
    return {
        "sirens":  [f for f in files if f.startswith("siren_")],
        "ambient": [f for f in files if f.startswith("ambient_")],
        "all":     files,
    }


@router.get(
    "/demo-assets/junctions",
    summary="List TLS junction IDs available for the acoustic demo",
    description=(
        "Returns TLS junction IDs from the live SUMO simulation if running, "
        "otherwise returns a hardcoded fallback list from the Kilpauk network."
    ),
)
def list_demo_junctions(request: Request) -> dict:
    sim = getattr(request.app.state, "sim", None)
    if sim is not None and getattr(sim, "is_connected", False):
        try:
            import traci
            junctions = list(traci.trafficlight.getIDList())
            return {"source": "live_sumo", "junctions": junctions}
        except Exception as exc:
            log.warning("Could not fetch TLS IDs from TraCI: %s", exc)

    return {"source": "fallback", "junctions": _FALLBACK_JUNCTIONS}


@router.get(
    "/demo-assets/plates",
    summary="List pre-canned demo plate image names",
    description=(
        "Returns the filenames available in ``demo-assets/plates/``.  "
        "Pass one of these as ``plate_name`` to the acoustic-event endpoint."
    ),
)
def list_demo_plates() -> dict:
    if not DEMO_PLATES.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                "demo-assets/plates/ not found. "
                "Run 'python setup_demo_assets.py' first."
            ),
        )
    files = sorted(p.name for p in DEMO_PLATES.iterdir() if p.is_file())
    return {"plates": files}


# ── Main endpoint ─────────────────────────────────────────────────────────────

@router.post(
    "/{junction_id}/acoustic-event",
    summary="Run the acoustic-detection demo pipeline for a junction",
    description=(
        "Accepts either a **file upload** (WAV) or a **``sample_name``** "
        "reference to a pre-recorded clip in ``demo-assets/sirens/``.\\n\\n"
        "Runs the full 4-stage pipeline with real inference at every step:\\n"
        "1. **mic_heard** — clip resolved\\n"
        "2. **siren_classified** — RandomForest on 80-dim MFCCs\\n"
        "3. **camera_triggered / anpr_confirmed** — OpenCV + EasyOCR on demo plate\\n"
        "4. **signal_flipped** — TraCI TLS green (if SUMO running + confirmed)\\n\\n"
        "Each stage is broadcast on ``/ws/logs`` before the response returns."
    ),
)
def acoustic_event(
    junction_id:  str,
    request:      Request,
    file:         Optional[UploadFile] = File(None,  description="WAV audio upload"),
    sample_name:  Optional[str]        = Form(None,  description="Filename from demo-assets/sirens/"),
    plate_name:   Optional[str]        = Form(None,  description=(
        "Filename from demo-assets/plates/ to use for the ANPR stage. "
        "Defaults to plate_demo.jpg (civilian / reject path). "
        "Pass plate_demo_authorized.jpg for the full green-light path."
    )),
) -> dict:
    """
    Synchronous handler — FastAPI runs in thread pool.
    All blocking I/O (librosa, sklearn, OpenCV, EasyOCR, TraCI) is safe here.
    """

    # ── Stage 1: mic_heard ────────────────────────────────────────────────────

    tmp_audio: Optional[Path] = None

    try:
        if file is not None:
            # Upload path
            suffix = Path(file.filename or "upload.wav").suffix or ".wav"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                shutil.copyfileobj(file.file, tmp)
                tmp_audio = Path(tmp.name)
            audio_label = file.filename or "upload"

        elif sample_name is not None:
            # Pre-recorded path — resolve and validate
            candidate = DEMO_SIRENS / sample_name
            if not candidate.exists():
                raise HTTPException(
                    status_code=404,
                    detail=(
                        f"Sample '{sample_name}' not found in demo-assets/sirens/. "
                        "Call GET /junction/demo-assets/list to see available samples."
                    ),
                )
            # Copy to temp so inference code always gets a plain file path
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                shutil.copy2(candidate, tmp.name)
                tmp_audio = Path(tmp.name)
            audio_label = sample_name

        else:
            raise HTTPException(
                status_code=422,
                detail="Provide either a file upload or a sample_name query parameter.",
            )

        _emit(
            "mic_heard", junction_id,
            f"[Acoustic Demo] mic_heard: clip='{audio_label}' junction={junction_id}",
            audio_label=audio_label,
        )
        log.info("[acoustic_demo] mic_heard: %s, junction=%s", audio_label, junction_id)

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Audio file error: {exc}") from exc

    # ── Stage 2: siren_classified ─────────────────────────────────────────────

    try:
        from audio_ml.infer import predict as _predict
        clf_result = _predict(tmp_audio)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc) + " — Run 'python -m audio_ml.train' first.",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=422, detail=f"Audio inference failed: {exc}"
        ) from exc
    finally:
        if tmp_audio and tmp_audio.exists():
            try:
                tmp_audio.unlink()
            except Exception:
                pass

    is_siren   = clf_result["is_siren"]
    confidence = clf_result["confidence"]
    label      = clf_result["label"]

    _emit(
        "siren_classified", junction_id,
        f"[Acoustic Demo] siren_classified: is_siren={is_siren} "
        f"conf={confidence:.3f} label={label!r}",
        is_siren=is_siren,
        confidence=confidence,
        label=label,
        all_proba=clf_result.get("all_proba", {}),
    )
    log.info(
        "[acoustic_demo] siren_classified: is_siren=%s conf=%.3f label=%r",
        is_siren, confidence, label,
    )

    # Build result dict — will accumulate through remaining stages
    result: dict = {
        "junction_id":        junction_id,
        "audio_label":        audio_label,
        "timestamp":          datetime.now(tz=timezone.utc).isoformat(),
        "mic_heard":          {"audio_label": audio_label},
        "siren_classified":   {
            "is_siren":   is_siren,
            "confidence": confidence,
            "label":      label,
            "all_proba":  clf_result.get("all_proba", {}),
        },
        "camera_triggered":   None,
        "anpr_confirmed":     None,
        "signal_flipped":     None,
    }

    if not (is_siren and confidence >= SIREN_CONFIDENCE_THRESHOLD):
        # Short-circuit: log reject path and return early
        _emit(
            "signal_flipped", junction_id,
            f"[Acoustic Demo] signal_flipped: SKIPPED — "
            f"{'not siren' if not is_siren else f'confidence {confidence:.0%} < threshold'}",
            is_siren=is_siren,
            confidence=confidence,
            signal_green=False,
            skipped=True,
            skip_reason=(
                "not_siren" if not is_siren
                else f"confidence_{confidence:.3f}_below_threshold_{SIREN_CONFIDENCE_THRESHOLD}"
            ),
        )
        result["signal_flipped"] = {
            "skipped":    True,
            "reason":     (
                "not_siren" if not is_siren
                else f"confidence {confidence:.0%} below threshold {SIREN_CONFIDENCE_THRESHOLD:.0%}"
            ),
            "signal_green": False,
        }
        return result

    # ── Stage 3: camera_triggered + anpr_confirmed ────────────────────────────

    # plate_name selects which demo plate to run through ANPR.
    # Default: plate_demo.jpg (civilian, reject path).
    # Authorized path: plate_demo_authorized.jpg (TN87C5106, ambulance).
    _plate_filename = plate_name if plate_name else "plate_demo.jpg"
    plate_image = DEMO_PLATES / _plate_filename
    # Safety: only allow files inside DEMO_PLATES (no path traversal)
    try:
        plate_image = plate_image.resolve()
        DEMO_PLATES.resolve()  # ensure DEMO_PLATES itself resolves
        plate_image.relative_to(DEMO_PLATES.resolve())  # raises ValueError if outside
    except ValueError:
        log.warning("[acoustic_demo] plate_name path traversal rejected: %s", plate_name)
        plate_image = DEMO_PLATES / "plate_demo.jpg"
    anpr_result: dict = {}

    if not plate_image.exists():
        log.warning("[acoustic_demo] plate image not found at %s", plate_image)
        camera_ok = False
        _emit(
            "camera_triggered", junction_id,
            "[Acoustic Demo] camera_triggered: plate image not found — ANPR skipped",
            camera_ok=False,
        )
        anpr_result = {
            "is_detected":   False,
            "is_authorized": False,
            "plate_text":    None,
            "confidence":    0.0,
            "ocr_engine":    "none",
            "skip_reason":   "plate_image_missing",
        }
    else:
        try:
            from anpr.detect_plate import detect_plate
            plate_res = detect_plate(str(plate_image))
        except Exception as exc:
            log.warning("[acoustic_demo] ANPR failed: %s", exc)
            plate_res = {
                "is_detected":   False,
                "is_authorized": False,
                "plate_text":    None,
                "confidence":    0.0,
                "ocr_engine":    "none",
                "error":         str(exc),
            }

        camera_ok = True
        _emit(
            "camera_triggered", junction_id,
            f"[Acoustic Demo] camera_triggered: plate={plate_res.get('plate_text')!r} "
            f"ocr={plate_res.get('ocr_engine')!r}",
            camera_ok=True,
            plate_text=plate_res.get("plate_text"),
            ocr_engine=plate_res.get("ocr_engine"),
        )

        is_authorized = bool(plate_res.get("is_authorized"))
        _emit(
            "anpr_confirmed", junction_id,
            f"[Acoustic Demo] anpr_confirmed: is_detected={plate_res.get('is_detected')} "
            f"is_authorized={is_authorized} "
            f"vehicle_type={plate_res.get('vehicle_type')!r}",
            is_detected=plate_res.get("is_detected"),
            is_authorized=is_authorized,
            plate_text=plate_res.get("plate_text"),
            vehicle_type=plate_res.get("vehicle_type"),
            confidence=plate_res.get("confidence"),
        )

        anpr_result = {
            "is_detected":   plate_res.get("is_detected"),
            "is_authorized": is_authorized,
            "plate_text":    plate_res.get("plate_text"),
            "raw_ocr_text":  plate_res.get("raw_ocr_text"),
            "vehicle_type":  plate_res.get("vehicle_type"),
            "confidence":    plate_res.get("confidence"),
            "ocr_engine":    plate_res.get("ocr_engine"),
        }

    result["camera_triggered"] = {"camera_ok": camera_ok, "image": str(plate_image.name)}
    result["anpr_confirmed"]   = anpr_result

    # ── Stage 4: signal_flipped ───────────────────────────────────────────────

    is_authorized = anpr_result.get("is_authorized", False)
    signal_green  = False
    signal_detail: dict = {}

    if not is_authorized:
        _emit(
            "signal_flipped", junction_id,
            f"[Acoustic Demo] signal_flipped: SKIPPED — plate not authorized "
            f"(plate={anpr_result.get('plate_text')!r})",
            is_authorized=False,
            signal_green=False,
            skipped=True,
            skip_reason="plate_not_authorized",
        )
        signal_detail = {
            "skipped":      True,
            "reason":       "plate_not_authorized",
            "signal_green": False,
        }
    else:
        sim = getattr(request.app.state, "sim", None)
        sim_connected = sim is not None and getattr(sim, "is_connected", False)

        if not sim_connected:
            _emit(
                "signal_flipped", junction_id,
                f"[Acoustic Demo] signal_flipped: SKIPPED — SUMO not running "
                f"(junction={junction_id})",
                is_authorized=True,
                signal_green=False,
                skipped=True,
                skip_reason="sim_not_running",
            )
            signal_detail = {
                "skipped":      True,
                "reason":       "sim_not_running",
                "signal_green": False,
            }
        else:
            # Flip the junction green via the same function used by GridController
            try:
                current_phase = sim.get_traffic_light_state(junction_id)
                all_green     = "G" * len(current_phase)
                sim.set_traffic_light_state(junction_id, all_green)
                signal_green = True
                log.info(
                    "[acoustic_demo] signal_flipped: junction=%s  phase=%s -> %s",
                    junction_id, current_phase, all_green,
                )
                _emit(
                    "signal_flipped", junction_id,
                    f"[Acoustic Demo] signal_flipped: junction={junction_id} -> ALL GREEN "
                    f"(was {current_phase!r})",
                    is_authorized=True,
                    signal_green=True,
                    previous_phase=current_phase,
                    new_phase=all_green,
                )
                signal_detail = {
                    "skipped":        False,
                    "signal_green":   True,
                    "previous_phase": current_phase,
                    "new_phase":      all_green,
                }
            except Exception as exc:
                log.error(
                    "[acoustic_demo] signal flip failed for junction=%s: %s",
                    junction_id, exc,
                )
                _emit(
                    "signal_flipped", junction_id,
                    f"[Acoustic Demo] signal_flipped: ERROR — {exc}",
                    is_authorized=True,
                    signal_green=False,
                    error=str(exc),
                )
                signal_detail = {
                    "skipped":      False,
                    "signal_green": False,
                    "error":        str(exc),
                }

    result["signal_flipped"] = signal_detail

    log.info(
        "[acoustic_demo] pipeline complete: junction=%s is_siren=%s "
        "conf=%.3f anpr_auth=%s signal_green=%s",
        junction_id, is_siren, confidence,
        anpr_result.get("is_authorized"), signal_green,
    )

    return result
