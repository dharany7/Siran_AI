"""
backend/routers/dispatch.py — POST /dispatch  (ambulance dispatch pipeline).

Orchestration order
-------------------
1. AmbulanceNavigatorAgent  (Gemini)
   → Reasons about fastest route given live SUMO traffic state.
   → Returns {"route": [...], "reasoning": "..."}  as JSON.
   → Full reasoning text is logged at INFO and included in the response.

2. GridControllerAgent  (TraCI + ETA math, no LLM)
   → Converts route to edges, schedules TLS green phases, inserts the
     ambulance vehicle into the live SUMO simulation, steps the simulation
     until the ambulance exits, then restores original TLS states.

3. DB logging
   → Writes a SirenEvent row containing the full negotiation log JSON.

Threading note
--------------
This handler is a plain ``def`` (not ``async def``).  FastAPI detects this and
automatically runs it in a thread-pool executor, which means:
  - Blocking Gemini HTTP call  ✓  (doesn't stall the event loop)
  - Blocking TraCI calls       ✓  (single thread, no race conditions)
  - DB session                 ✓  (sync SQLAlchemy)
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import SirenEvent, SecurityEvent, Driver
from backend.auth import get_current_driver
from agents.ambulance_navigator import AmbulanceNavigatorAgent
from agents.grid_controller import GridControllerAgent
from security.guard import guard as prompt_guard
from backend.routers.ws import bus as _bus

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/dispatch", tags=["Dispatch"])


# ── Request / response schemas ────────────────────────────────────────────────

class DispatchRequest(BaseModel):
    dest_hospital_id: int  = Field(..., description="Hospital ID from the hospitals table (GET /hospitals to list)")
    start_lat: float       = Field(13.0843, description="Driver's current latitude (WGS84)")
    start_lng: float       = Field(80.2371, description="Driver's current longitude (WGS84)")


class TLSScheduleEntry(BaseModel):
    junction_id:   str
    edge:          str
    distance_m:    float
    eta_seconds:   float
    green_at_step: int


class DispatchResponse(BaseModel):
    event_id:             int
    hospital_name:        str
    route_edges:          list[str]
    route_edge_count:     int
    distance_m:           float
    navigator_reasoning:  str
    model_used:           str
    timing_log:           list[str]
    tls_schedule:         list[TLSScheduleEntry]
    steps_taken:          int
    completed:            bool
    negotiation_log:      dict


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_sim(request: Request):
    """Read SumoEnv from the app-state singleton set by POST /sim/start."""
    return getattr(request.app.state, "sim", None)


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.post(
    "",
    response_model=DispatchResponse,
    summary="Dispatch an ambulance through the live simulation",
    description=(
        "Runs the full Siren AI MAS dispatch pipeline:\n\n"
        "**Step 1 — Navigator (Gemini):** Queries the live SUMO traffic state "
        "(TLS phase, vehicle count) and asks Gemini to reason about the fastest "
        "route. Gemini must respond with strict JSON "
        "`{\"route\":[...], \"reasoning\":\"...\"}`. Full reasoning is logged "
        "and returned.\n\n"
        "**Step 2 — Grid Controller (ETA math + TraCI):** Computes per-junction "
        "ETA (distance ÷ speed), inserts an ambulance vehicle into the live SUMO "
        "process via TraCI, turns each TLS junction green just before the "
        "ambulance arrives, then restores original phases.\n\n"
        "**Step 3 — DB logging:** Stores the full negotiation log in the "
        "`siren_events` table.\n\n"
        "**Requires `POST /sim/start` to have been called first.**"
    ),
)
def dispatch(
    body:           DispatchRequest,
    request:        Request,
    db:             Session = Depends(get_db),
    current_driver: Driver  = Depends(get_current_driver),
) -> DispatchResponse:
    """
    Synchronous handler — FastAPI runs this in a thread pool, making all
    blocking Gemini HTTP + TraCI calls safe without stalling the event loop.
    """

    # ── Pre-flight ────────────────────────────────────────────────────────────
    logger.info(
        "dispatch: authenticated driver=%s plate=%s",
        current_driver.id,
        current_driver.ambulance_plate,
    )
    sim = _get_sim(request)
    if sim is None or not getattr(sim, "is_connected", False):
        # ── Safety net: auto-start the simulation rather than returning an error ──
        logger.warning(
            "dispatch: simulation not running — attempting auto-start before proceeding"
        )
        try:
            from agents.sumo_env import SumoEnv
            sim = SumoEnv()
            sim.start()
            request.app.state.sim = sim
            logger.info("dispatch: auto-started simulation (step=%d)", sim.step_count)
        except Exception as _exc:
            logger.error("dispatch: auto-start failed: %s", _exc)
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Simulation is not running and could not be auto-started: {_exc}. "
                    "Check that SUMO is installed and sumo_net/kilpauk.net.xml exists."
                ),
            )

    # ── Resolve hospital ──────────────────────────────────────────────────────
    from backend.models import Hospital
    hospital = db.query(Hospital).filter(Hospital.id == body.dest_hospital_id).first()
    if hospital is None:
        raise HTTPException(
            status_code=404,
            detail=f"Hospital id={body.dest_hospital_id} not found. GET /hospitals to list.",
        )

    # ── Resolve geo-coordinates -> nearest SUMO edges ─────────────────────────
    start_edge = sim.nearest_edge(body.start_lat, body.start_lng)
    dest_edge  = sim.nearest_edge(hospital.lat, hospital.lng)

    if not start_edge:
        raise HTTPException(
            status_code=422,
            detail=f"Could not snap start ({body.start_lat},{body.start_lng}) to any SUMO edge.",
        )
    if not dest_edge:
        raise HTTPException(
            status_code=422,
            detail=f"Could not snap hospital '{hospital.name}' ({hospital.lat},{hospital.lng}) to any SUMO edge.",
        )

    if start_edge == dest_edge:
        raise HTTPException(
            status_code=422,
            detail=f"Start and destination snapped to the same edge ({start_edge}). Choose a different hospital.",
        )

    logger.info("="*60)
    logger.info(
        "DISPATCH: driver=(%s,%s) -> hospital=%s (%s,%s)",
        body.start_lat, body.start_lng,
        hospital.name, hospital.lat, hospital.lng,
    )
    logger.info("start_edge=%s  dest_edge=%s", start_edge, dest_edge)

    # ── Security guard ────────────────────────────────────────────────────────
    guard_input = f"start_lat={body.start_lat} start_lng={body.start_lng} hospital_id={body.dest_hospital_id} hospital_name={hospital.name}"
    guard_result = prompt_guard.check(guard_input)

    # Log every guard call to security_events (SAFE or BLOCKED)
    _verdict = "BLOCKED" if guard_result.blocked else "SAFE"
    try:
        sec_event = SecurityEvent(
            payload        = guard_input[:2048],
            verdict        = _verdict,
            layer_blocked  = guard_result.layer if guard_result.blocked else None,
            blocked_reason = guard_result.reason if guard_result.blocked else None,
            endpoint       = "/dispatch",
        )
        db.add(sec_event)
        db.commit()
    except Exception as _exc:
        logger.error("Security event DB write failed: %s", _exc)

    if guard_result.blocked:
        logger.warning(
            "DISPATCH BLOCKED by guard (layer=%d): %s",
            guard_result.layer, guard_result.reason,
        )
        raise HTTPException(
            status_code=403,
            detail={
                "error":  "Request blocked by security guard",
                "layer":  guard_result.layer,
                "reason": guard_result.reason,
            },
        )


    # ── Step 1: Route with duarouter ──────────────────────────────────────────
    from agents.grid_controller import route_via_duarouter, NET_FILE
    try:
        precomputed_route = route_via_duarouter(start_edge, dest_edge, NET_FILE)
        logger.info("duarouter route: %d edges", len(precomputed_route))
    except Exception as exc:
        logger.error("duarouter failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Routing failed: {exc}",
        )

    # ── Step 2: Read current sim state ────────────────────────────────────────
    try:
        sim_state = sim.get_state()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not read sim state: {exc}")

    logger.info(
        "Pre-dispatch sim state: step=%d  tls_count=%d  vehicles=%d",
        sim_state["step"], sim_state.get("tls_count", 0), sim_state["vehicle_count"],
    )

    # ── Step 3: AmbulanceNavigatorAgent (Gemini) ──────────────────────────────
    start_desc = f"({body.start_lat:.5f},{body.start_lng:.5f})"
    dest_desc  = f"{hospital.name} ({hospital.lat:.5f},{hospital.lng:.5f})"
    navigator  = AmbulanceNavigatorAgent()
    nav_result = navigator.run_sync({
        "start":       start_desc,
        "destination": dest_desc,
        "route":       precomputed_route,   # pass duarouter route
        "sim_state":   sim_state,
    })

    route      = nav_result.get("route", precomputed_route)
    reasoning  = nav_result.get("reasoning", "(no reasoning returned)")
    model_used = nav_result.get("model", "unknown")

    if nav_result.get("error"):
        logger.warning("Navigator used fallback route: %s", nav_result["error"])

    # ── Step 4: GridControllerAgent (TraCI + ETA math) ────────────────────────
    grid_ctrl   = GridControllerAgent()
    grid_result = grid_ctrl.run_sync({
        "route":       route,
        "start":       start_edge,
        "destination": dest_edge,
        "sim":         sim,
    })

    timing_log   = grid_result.get("timing_log", [])
    tls_schedule = grid_result.get("tls_schedule", [])
    steps_taken  = grid_result.get("steps_taken", 0)
    completed    = grid_result.get("completed", False)
    distance_m   = grid_result.get("distance_m", 0.0)

    logger.info(
        "GridController: completed=%s  steps=%d  distance=%.0fm",
        completed, steps_taken, distance_m,
    )

    # ── Step 5: Build negotiation log ─────────────────────────────────────────
    negotiation_log: dict = {
        "start_lat":    body.start_lat,
        "start_lng":    body.start_lng,
        "start_edge":   start_edge,
        "hospital":     hospital.name,
        "dest_edge":    dest_edge,
        "distance_m":   distance_m,
        "timestamp":    datetime.now(tz=timezone.utc).isoformat(),
        "navigator": {
            "model":        model_used,
            "route":        route,
            "reasoning":    reasoning,
            "raw_response": nav_result.get("raw_response", ""),
            "error":        nav_result.get("error"),
        },
        "grid_controller": {
            "veh_id":       grid_result.get("veh_id"),
            "edges_used":   grid_result.get("edges_used", []),
            "tls_schedule": [
                {k: v for k, v in s.items() if k not in ("applied", "restored")}
                for s in tls_schedule
            ],
            "timing_log":   timing_log,
            "steps_taken":  steps_taken,
            "completed":    completed,
            "error":        grid_result.get("error"),
        },
    }

    # ── Step 6: Persist to DB ─────────────────────────────────────────────────
    event_id = -1
    try:
        notes_json = json.dumps(negotiation_log, ensure_ascii=False)
        event = SirenEvent(
            siren_type = "ambulance_dispatch",
            confidence = f"{body.start_lat},{body.start_lng}->{hospital.name}",
            audio_file = None,
            notes      = notes_json,
        )
        db.add(event)
        db.commit()
        db.refresh(event)
        event_id = event.id
        logger.info("Dispatch logged as SirenEvent id=%d", event_id)
    except Exception as exc:
        logger.error("DB logging failed: %s", exc)

    # ── Build clean tls_schedule for Pydantic response ────────────────────────
    clean_schedule = [
        TLSScheduleEntry(
            junction_id   = s["junction_id"],
            edge          = s["edge"],
            distance_m    = s["distance_m"],
            eta_seconds   = s["eta_seconds"],
            green_at_step = s["green_at_step"],
        )
        for s in tls_schedule
    ]

    resp = DispatchResponse(
        event_id            = event_id,
        hospital_name       = hospital.name,
        route_edges         = route,
        route_edge_count    = len(route),
        distance_m          = distance_m,
        navigator_reasoning = reasoning,
        model_used          = model_used,
        timing_log          = timing_log,
        tls_schedule        = clean_schedule,
        steps_taken         = steps_taken,
        completed           = completed,
        negotiation_log     = negotiation_log,
    )

    # ── Broadcast to live dashboard ──────────────────────────────────────────
    _bus.publish_sync({
        "type":    "dispatch",
        "msg":     f"Ambulance dispatched to {hospital.name}  {len(route)} edges  "
                   f"{distance_m:.0f}m  completed={completed}",
        "payload": {
            "event_id":      event_id,
            "hospital_name": hospital.name,
            "route_edges":   route[:3],   # first 3 edges for preview
            "distance_m":    distance_m,
            "steps":         steps_taken,
            "completed":     completed,
            "reasoning":     reasoning[:200] if reasoning else "",
        },
    })

    return resp


# ── Cancel endpoint ────────────────────────────────────────────────

@router.post(
    "/{dispatch_id}/cancel",
    summary="Cancel a previously logged dispatch event",
    description=(
        "Marks a `SirenEvent` row as cancelled in the database by appending a "
        "cancellation payload to its `notes` JSON.  Called automatically by "
        "`POST /anpr/verify` when an unauthorized plate is detected."
    ),
)
def cancel_dispatch(
    dispatch_id: int,
    db: Session = Depends(get_db),
) -> dict:
    """
    Mark dispatch *dispatch_id* as cancelled.
    Returns 404 if the event does not exist.
    """
    event = db.query(SirenEvent).filter(SirenEvent.id == dispatch_id).first()
    if event is None:
        raise HTTPException(
            status_code=404,
            detail=f"No dispatch event with id={dispatch_id}.",
        )

    try:
        existing = json.loads(event.notes or "{}")
    except Exception:
        existing = {"original_notes": event.notes}

    if existing.get("cancelled"):
        return {"cancelled": True, "event_id": dispatch_id, "note": "Already cancelled."}

    existing["cancelled"]         = True
    existing["cancelled_at"]      = datetime.now(tz=timezone.utc).isoformat()
    existing["cancelled_reason"]  = "POST /dispatch/{id}/cancel called directly"
    event.notes = json.dumps(existing, ensure_ascii=False)
    db.commit()

    logger.warning("Dispatch event id=%d cancelled via API.", dispatch_id)
    return {"cancelled": True, "event_id": dispatch_id}
