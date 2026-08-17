"""
routers/sim.py — SUMO/TraCI simulation control & status endpoints.

The SumoEnv instance lives on ``request.app.state.sim`` so that every
request handler in the *same running FastAPI process* shares exactly one
TraCI connection.  This avoids the module-global pitfall where a helper
called from an external script mutates a different process's memory.

Endpoints
---------
POST /sim/start   – launch SUMO headlessly and open a TraCI connection
POST /sim/stop    – close TraCI and terminate SUMO cleanly
GET  /sim/status  – report connection liveness + current simulation step
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Request
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/sim", tags=["Simulation"])


# ── App-state helpers ─────────────────────────────────────────────────────────

def _get_sim(request: Request):
    """Return the SumoEnv stored in app state, or None."""
    return getattr(request.app.state, "sim", None)


def _set_sim(request: Request, sim) -> None:
    """Store a SumoEnv instance (or None) in app state."""
    request.app.state.sim = sim


# ── Response schemas ──────────────────────────────────────────────────────────

class SimStatusResponse(BaseModel):
    connected:         bool
    step:              int
    tls_id:            Optional[str]  = None
    tls_state:         Optional[str]  = None
    vehicle_count:     Optional[int]  = None
    ambulance_present: Optional[bool] = None
    message:           str


class SimActionResponse(BaseModel):
    success: bool
    message: str


# ── POST /sim/start ───────────────────────────────────────────────────────────

@router.post(
    "/start",
    response_model=SimActionResponse,
    summary="Start the SUMO simulation",
    description=(
        "Launches SUMO headlessly and opens a TraCI connection **inside this "
        "server process**. The live `SumoEnv` is stored in app state so that "
        "`GET /sim/status` reads from the same connection. "
        "Returns `success=false` if the simulation is already running."
    ),
)
async def sim_start(request: Request) -> SimActionResponse:
    """
    Start a headless SUMO simulation via TraCI.

    - Idempotent: calling again while already connected returns `success=false`.
    - Uses `sumo_net/cross.sumocfg` and the bundled 4-way intersection network.
    """
    existing = _get_sim(request)
    if existing is not None and getattr(existing, "is_connected", False):
        return SimActionResponse(
            success=False,
            message="Simulation is already running. Call POST /sim/stop first.",
        )

    try:
        from agents.sumo_env import SumoEnv
        sim = SumoEnv()
        sim.start()
        _set_sim(request, sim)
        logger.info("Simulation started and stored in app.state.sim (step=%d)", sim.step_count)
        return SimActionResponse(success=True, message="SUMO simulation started.")
    except FileNotFoundError as exc:
        logger.error("SUMO binary/config not found: %s", exc)
        return SimActionResponse(success=False, message=f"SUMO not found: {exc}")
    except Exception as exc:
        logger.error("Failed to start simulation: %s", exc)
        return SimActionResponse(success=False, message=f"Error: {exc}")


# ── POST /sim/stop ────────────────────────────────────────────────────────────

@router.post(
    "/stop",
    response_model=SimActionResponse,
    summary="Stop the SUMO simulation",
    description=(
        "Closes the TraCI connection and terminates the SUMO process cleanly. "
        "After this call `GET /sim/status` will return `connected=false`."
    ),
)
async def sim_stop(request: Request) -> SimActionResponse:
    """
    Stop the running SUMO simulation.

    - Safe to call even when no simulation is running.
    - Clears the `SumoEnv` from app state after closing.
    """
    sim = _get_sim(request)
    if sim is None or not getattr(sim, "is_connected", False):
        _set_sim(request, None)
        return SimActionResponse(success=False, message="No simulation is currently running.")

    try:
        sim.close()
        _set_sim(request, None)
        logger.info("Simulation stopped cleanly via POST /sim/stop")
        return SimActionResponse(success=True, message="SUMO simulation stopped.")
    except Exception as exc:
        logger.error("Error stopping simulation: %s", exc)
        _set_sim(request, None)
        return SimActionResponse(success=False, message=f"Stopped with error: {exc}")


# ── GET /sim/status ───────────────────────────────────────────────────────────

@router.get(
    "/status",
    response_model=SimStatusResponse,
    summary="SUMO/TraCI simulation status",
    description=(
        "Returns whether the SUMO TraCI connection is alive and the current "
        "simulation step count. `connected` is `true` **only** after "
        "`POST /sim/start` has been called on this running server instance."
    ),
)
async def sim_status(request: Request) -> SimStatusResponse:
    """
    Check the live SUMO/TraCI simulation state.

    - **connected** – `true` if a TraCI connection is open in this process
    - **step** – current simulation timestep (integer seconds elapsed)
    - **tls_state** – traffic-light phase string at the main junction (e.g. `'GrGr'`)
    - **vehicle_count** – number of vehicles currently in the network
    - **ambulance_present** – whether the ambulance vehicle is in the network
    """
    sim = _get_sim(request)

    if sim is None or not getattr(sim, "is_connected", False):
        return SimStatusResponse(
            connected=False,
            step=0,
            message="Simulation not started. POST /sim/start to begin.",
        )

    try:
        state = sim.get_state()
        return SimStatusResponse(
            connected=True,
            step=state["step"],
            tls_id=state.get("tls_id"),
            tls_state=state.get("tls_state"),
            vehicle_count=state.get("vehicle_count"),
            ambulance_present=state.get("ambulance_present"),
            message="ok",
        )
    except Exception as exc:
        logger.error("Error reading simulation state: %s", exc)
        return SimStatusResponse(
            connected=False,
            step=getattr(sim, "step_count", 0),
            message=f"TraCI error: {exc}",
        )
