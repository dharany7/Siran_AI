"""
agents/grid_controller.py — ETA-based traffic-light preemption for ambulance passage.

Real-network version (Kilpauk, Chennai — kilpauk.net.xml).

Changes from the synthetic version
-----------------------------------
- All hardcoded JUNCTION_EDGE_MAP / EDGE_TABLE / TLS_JUNCTIONS constants removed.
- Routing is done by duarouter (SUMO's shortest-path router), which reads the
  real network file and returns a sequence of real edge IDs.
- Edge lengths are read from TraCI at runtime (traci.lane.getLength).
- TLS detection is dynamic: any edge whose destination junction appears in
  traci.trafficlight.getIDList() is treated as a TLS junction.
- The green-corridor application loop is unchanged — it is TLS-ID-agnostic.

Payload keys (unchanged interface)
------------------------------------
route       : list[str]  — SUMO edge IDs from route_via_duarouter()
start       : str        — starting SUMO edge ID
destination : str        — destination SUMO edge ID
sim         : SumoEnv    — live SumoEnv instance

Returns (unchanged interface)
-------------------------------
dict with:
    veh_id        : str        — TraCI ID of the dispatched ambulance
    edges_used    : list[str]  — edge IDs assigned to the ambulance
    tls_schedule  : list[dict] — full timing plan (pre-execution)
    timing_log    : list[str]  — chronological action log
    steps_taken   : int        — simulation steps consumed
    completed     : bool       — True if ambulance reached destination
"""
from __future__ import annotations

import logging
import os
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TYPE_CHECKING

from agents.base_agent import BaseAgent

if TYPE_CHECKING:
    from agents.sumo_env import SumoEnv

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
AMBULANCE_SPEED    = 13.889   # m/s — ~50 km/h, typical urban emergency speed
PRE_GREEN_BUFFER   = 10.0    # seconds before ETA to switch to green
MAX_DISPATCH_STEPS = 600     # 10 min ceiling (real network is larger)

SUMO_HOME  = Path(os.environ.get("SUMO_HOME",
                  r"C:\Program Files (x86)\Eclipse\Sumo"))
DUAROUTER  = SUMO_HOME / "bin" / "duarouter.exe"
NET_FILE   = Path(__file__).parent.parent / "sumo_net" / "kilpauk.net.xml"


# ── Routing via duarouter ──────────────────────────────────────────────────────

def route_via_duarouter(
    start_edge: str,
    end_edge: str,
    net_file: Path = NET_FILE,
) -> list[str]:
    """
    Compute the shortest road-network path from *start_edge* to *end_edge*
    using SUMO's duarouter CLI.

    Returns an ordered list of SUMO edge IDs (the route).
    Raises RuntimeError if duarouter fails or produces no route.
    """
    if not DUAROUTER.exists():
        raise FileNotFoundError(f"duarouter not found at {DUAROUTER}")
    if not net_file.exists():
        raise FileNotFoundError(f"Network file not found: {net_file}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        trips_xml = tmp / "ambulance_trip.xml"
        routes_xml = tmp / "ambulance_route.xml"
        alt_xml    = tmp / "ambulance_route.alt.xml"

        # Write a single-vehicle trip file
        trips_xml.write_text(
            '<?xml version="1.0"?>\n'
            '<trips>\n'
            f'  <trip id="amb" depart="0" from="{start_edge}" to="{end_edge}"'
            '   type="ambulance"/>\n'
            '</trips>\n'
        )

        cmd = [
            str(DUAROUTER),
            "--net-file", str(net_file),
            "--trip-files", str(trips_xml),
            "-o", str(routes_xml),
            "--no-warnings",
            "--ignore-errors",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0 and not routes_xml.exists():
            raise RuntimeError(
                f"duarouter failed (rc={result.returncode}): {result.stderr[:500]}"
            )

        # Parse the output route XML
        if not routes_xml.exists():
            raise RuntimeError("duarouter produced no output file")

        tree = ET.parse(str(routes_xml))
        root = tree.getroot()

        # Find <vehicle id="amb"><route edges="..."/></vehicle>
        for vehicle in root.iter("vehicle"):
            route_el = vehicle.find("route")
            if route_el is not None:
                edges_str = route_el.get("edges", "")
                if edges_str:
                    edges = edges_str.strip().split()
                    logger.info(
                        "duarouter route: %d edges, %s ... %s",
                        len(edges), edges[0] if edges else "", edges[-1] if edges else "",
                    )
                    return edges

    raise RuntimeError("duarouter returned no route for the requested trip")


# ── Edge / TLS helpers (TraCI) ─────────────────────────────────────────────────

def get_edge_length(edge_id: str) -> float:
    """Return the length of *edge_id* in metres via TraCI."""
    import traci
    try:
        lanes = traci.edge.getLaneNumber(edge_id)
        if lanes > 0:
            return traci.lane.getLength(f"{edge_id}_0")
    except Exception:
        pass
    return 100.0   # safe fallback


def get_tls_junctions_from_traci() -> set[str]:
    """Return the set of all TLS-controlled junction IDs in the running sim."""
    import traci
    return set(traci.trafficlight.getIDList())


def edge_destination_junction(edge_id: str) -> str | None:
    """
    Return the destination junction ID of *edge_id* via TraCI.
    Internal junction edges (starting with ':') are skipped.
    """
    import traci
    if edge_id.startswith(":"):
        return None
    try:
        return traci.edge.getLastStepVehicleIDs  # just to test import
    except Exception:
        pass
    # Use network XML parsing as fallback (no TraCI call for this info)
    return None


# ── TLS schedule builder ───────────────────────────────────────────────────────

def compute_tls_schedule(
    edge_ids: list[str],
    base_step: int,
    tls_set: set[str],
) -> list[dict]:
    """
    Build a per-junction green-phase timing plan for the ambulance route.

    For each edge in the route, reads the real edge length from TraCI,
    accumulates distance, computes ETA, and checks whether the edge's
    destination junction has a TLS (by matching node IDs embedded in the
    edge ID against the TLS set).

    SUMO edge IDs for OSM-derived networks follow the pattern:
      "<osmNodeId>#<segmentIndex>"  or  "-<osmNodeId>#<segmentIndex>"
    The destination junction of edge "123456789#0" is junction "123456789".

    Args:
        edge_ids  : Ordered SUMO edge IDs the ambulance will traverse.
        base_step : Simulation step at dispatch time.
        tls_set   : Set of TLS junction IDs (from traci.trafficlight.getIDList()).

    Returns:
        List of dicts — one per TLS junction on the route:
            junction_id   : str   — TraCI TLS ID
            edge          : str   — edge leading into this junction
            distance_m    : float — cumulative distance from dispatch point
            eta_seconds   : float — ETA for this junction
            green_at_step : int   — sim step to apply green phase
            applied       : bool  — False initially
            restored      : bool  — False initially
    """
    schedule: list[dict] = []
    cumulative_m = 0.0

    for edge_id in edge_ids:
        if edge_id.startswith(":"):
            continue   # skip internal junction edges

        length = get_edge_length(edge_id)
        cumulative_m += length
        eta = cumulative_m / AMBULANCE_SPEED

        # Infer destination junction from edge ID
        # SUMO OSM edges: "123456789#0" -> junction "123456789"
        # Reverse edges:  "-123456789#0" -> junction "123456789"
        raw = edge_id.lstrip("-").split("#")[0]
        dst_junction = raw

        if dst_junction in tls_set:
            green_at = base_step + max(0, int(eta - PRE_GREEN_BUFFER))
            schedule.append({
                "junction_id":   dst_junction,
                "edge":          edge_id,
                "distance_m":    round(cumulative_m, 1),
                "eta_seconds":   round(eta, 2),
                "green_at_step": green_at,
                "applied":       False,
                "restored":      False,
            })

    return schedule


# ── Agent class ───────────────────────────────────────────────────────────────

class GridControllerAgent(BaseAgent):
    """
    Controls traffic lights along the ambulance route with ETA-based timing.

    Works with any SUMO network — reads real edge lengths and TLS IDs
    from the live TraCI session. No hardcoded network topology.

    Payload keys
    ------------
    route       : list[str]  — SUMO edge IDs (from duarouter or Navigator)
    start       : str        — starting SUMO edge ID
    destination : str        — destination SUMO edge ID
    sim         : SumoEnv    — live SumoEnv instance from app.state.sim
    """

    def __init__(self) -> None:
        super().__init__(name="grid_controller")

    def run_sync(self, payload: dict) -> dict:
        """
        Execute the ambulance dispatch on the real road network.

        1. Retrieve the pre-computed route (list of edge IDs) from payload.
        2. Query real edge lengths + TLS set from TraCI.
        3. Build TLS timing schedule (ETA-based).
        4. Add the ambulance vehicle to the SUMO simulation.
        5. Step the simulation, applying green phases at the right moments.
        6. Restore original TLS phases after the ambulance clears.
        """
        import traci

        sim: SumoEnv = payload["sim"]
        route: list[str] = payload["route"]       # edge IDs
        start_edge: str  = payload["start"]
        dest_edge: str   = payload["destination"]

        if not route:
            raise ValueError("Empty route — cannot dispatch")

        # Filter internal edges
        driveable = [e for e in route if not e.startswith(":")]
        if not driveable:
            raise ValueError("Route contains only internal edges")

        # ── Build TLS schedule ────────────────────────────────────────────
        tls_set     = get_tls_junctions_from_traci()
        base_step   = sim.step_count
        tls_schedule = compute_tls_schedule(driveable, base_step, tls_set)

        total_m = sum(get_edge_length(e) for e in driveable)

        logger.info(
            "GridController: route=%d edges, %.0f m, %d TLS junctions",
            len(driveable), total_m, len(tls_schedule),
        )

        timing_log: list[str] = [
            f"Dispatch: {len(driveable)} edges, {total_m:.0f} m, "
            f"{len(tls_schedule)} TLS junctions"
        ]

        # ── Save original TLS states ──────────────────────────────────────
        saved_phases: dict[str, str] = {}
        for entry in tls_schedule:
            jid = entry["junction_id"]
            if jid not in saved_phases:
                try:
                    saved_phases[jid] = sim.get_traffic_light_state(jid)
                except Exception:
                    saved_phases[jid] = ""

        # ── Add ambulance to simulation ───────────────────────────────────
        veh_id = f"ambulance_{base_step}"
        try:
            traci.vehicle.add(
                vehID=veh_id,
                routeID="",
                typeID="DEFAULT_VEHTYPE",
                depart="now",
                departLane="best",
                departSpeed="max",
            )
            traci.vehicle.setRoute(veh_id, driveable)
            traci.vehicle.setSpeedMode(veh_id, 0)   # ignore speed limits
            traci.vehicle.setMaxSpeed(veh_id, AMBULANCE_SPEED * 1.5)
            timing_log.append(f"Step {sim.step_count}: ambulance {veh_id} added")
        except Exception as exc:
            logger.warning("Could not add ambulance vehicle: %s", exc)
            timing_log.append(f"WARNING: ambulance add failed: {exc}")

        # ── Simulation loop ───────────────────────────────────────────────
        completed = False
        steps_run = 0

        for _ in range(MAX_DISPATCH_STEPS):
            state = sim.step()
            steps_run += 1
            current_step = state["step"]

            # Apply green phases at scheduled times
            for entry in tls_schedule:
                if not entry["applied"] and current_step >= entry["green_at_step"]:
                    jid = entry["junction_id"]
                    try:
                        current_phase = sim.get_traffic_light_state(jid)
                        all_green = "G" * len(current_phase)
                        sim.set_traffic_light_state(jid, all_green)
                        entry["applied"] = True
                        msg = (
                            f"Step {current_step}: GREEN at {jid} "
                            f"({entry['distance_m']:.0f} m, ETA {entry['eta_seconds']:.1f}s)"
                        )
                        timing_log.append(msg)
                        logger.info(msg)
                    except Exception as exc:
                        timing_log.append(f"Step {current_step}: TLS {jid} error: {exc}")

            # Restore phases 5 steps after applying
            for entry in tls_schedule:
                if (entry["applied"] and not entry["restored"]
                        and current_step >= entry["green_at_step"] + 5):
                    jid = entry["junction_id"]
                    if saved_phases.get(jid):
                        try:
                            sim.set_traffic_light_state(jid, saved_phases[jid])
                            entry["restored"] = True
                            timing_log.append(
                                f"Step {current_step}: restored TLS {jid}"
                            )
                        except Exception:
                            pass

            # Check if ambulance completed the route
            if veh_id not in traci.vehicle.getIDList():
                completed = True
                timing_log.append(
                    f"Step {current_step}: ambulance {veh_id} reached destination"
                )
                break

        # Restore any un-restored TLS phases
        for entry in tls_schedule:
            if entry["applied"] and not entry["restored"]:
                jid = entry["junction_id"]
                if saved_phases.get(jid):
                    try:
                        sim.set_traffic_light_state(jid, saved_phases[jid])
                    except Exception:
                        pass

        return {
            "veh_id":       veh_id,
            "edges_used":   driveable,
            "tls_schedule": tls_schedule,
            "timing_log":   timing_log,
            "steps_taken":  steps_run,
            "completed":    completed,
            "distance_m":   round(total_m, 1),
        }

    async def run(self, payload: dict) -> dict:
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.run_sync, payload)
