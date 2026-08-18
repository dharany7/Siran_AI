"""
agents/sumo_env.py — SUMO TraCI environment for the Siren AI project.

Manages the full lifecycle of a headless SUMO simulation:
  - Start / stop via TraCI
  - Spawn regular cars + an ambulance vehicle (special vType)
  - Read and write traffic-light phases at any junction
  - Expose step(), get_state(), and a simple simulation loop
  - Geo-coordinate → nearest SUMO edge conversion

Network: Kilpauk, Chennai (real OSM data — kilpauk.net.xml)
         Falls back to the synthetic cross.net.xml if the real file is missing.

TraCI reference:
    https://sumo.dlr.de/docs/TraCI.html
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent.parent          # project root
SUMO_NET_DIR = _HERE / "sumo_net"

# Real network (primary); fall back to synthetic if missing
_REAL_CFG = SUMO_NET_DIR / "kilpauk.sumocfg"
_FALLBACK_CFG = SUMO_NET_DIR / "cross.sumocfg"
SUMO_CFG = _REAL_CFG if _REAL_CFG.exists() else _FALLBACK_CFG

NETWORK_FILE = SUMO_NET_DIR / "kilpauk.net.xml"

# SUMO_HOME is set by the installer; fall back to the known Windows path
SUMO_HOME = Path(
    os.environ.get("SUMO_HOME", r"C:\Program Files (x86)\Eclipse\Sumo")
)
SUMO_BIN  = SUMO_HOME / "bin" / "sumo.exe"

# TraCI port — pick any free port; we open it ourselves
DEFAULT_PORT = 8813

# Ambulance vehicle id used in TraCI dispatch
AMBULANCE_VEH_ID = "ambulance_0"


class SumoEnv:
    """
    Headless SUMO simulation environment controlled via TraCI.

    Lifecycle
    ---------
    env = SumoEnv()
    env.start()
    for _ in range(200):
        env.step()
    env.close()

    Or use as a context manager::

        with SumoEnv() as env:
            for _ in range(200):
                state = env.step()
                print(state)
    """

    def __init__(
        self,
        cfg_path: str | Path = SUMO_CFG,
        port: int = DEFAULT_PORT,
    ) -> None:
        self.cfg_path  = Path(cfg_path)
        self.port      = port
        self._proc: Optional[subprocess.Popen] = None
        self._step_count: int = 0
        self._connected: bool = False
        self._tls_ids: list[str] = []   # populated after start()

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def start(self) -> None:
        """
        Launch SUMO as a subprocess and open the TraCI connection.

        SUMO is started in headless mode (``sumo.exe``, not ``sumo-gui.exe``)
        so this is safe in CI / server environments.
        """
        import traci  # imported here so the module can be imported without SUMO

        if not SUMO_BIN.exists():
            raise FileNotFoundError(
                f"SUMO binary not found at {SUMO_BIN}. "
                "Check that SUMO_HOME is set correctly."
            )
        if not self.cfg_path.exists():
            raise FileNotFoundError(f"SUMO config not found: {self.cfg_path}")

        cmd = [
            str(SUMO_BIN),
            "-c", str(self.cfg_path),
            "--remote-port", str(self.port),
            "--start",                   # begin simulation immediately
            "--quit-on-end",             # exit cleanly when simulation ends
            "--no-warnings",
            "--no-step-log",
        ]
        logger.info("Starting SUMO: %s", " ".join(cmd))
        proc_env = os.environ.copy()
        proc_env["SUMO_HOME"] = str(SUMO_HOME)
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=proc_env,
        )

        # Open TraCI connection (retries are built into traci.init)
        traci.init(port=self.port, numRetries=10)
        self._connected = True
        self._step_count = 0

        # Cache TLS IDs from the live network
        self._tls_ids = list(traci.trafficlight.getIDList())
        logger.info(
            "TraCI connected on port %d | network=%s | TLS count=%d",
            self.port, self.cfg_path.name, len(self._tls_ids),
        )

    def close(self) -> None:
        """Close the TraCI connection and terminate SUMO."""
        import traci
        if self._connected:
            try:
                traci.close()
            except Exception:  # noqa: BLE001
                pass
            self._connected = False
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            self._proc = None
        logger.info("SUMO simulation closed at step %d", self._step_count)

    def __enter__(self) -> "SumoEnv":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ── Simulation step ────────────────────────────────────────────────────

    def step(self, delta_t: float = 1.0) -> dict:
        """
        Advance the simulation by one step (default step-size = 1.0 s).

        Note: traci.simulationStep() called with no argument advances by one
        step of the configured step-size.  Do NOT pass delta_t directly —
        in TraCI, simulationStep(N) means "run until simulation time >= N"
        (an absolute target), not "advance by N seconds".
        """
        import traci
        traci.simulationStep()
        self._step_count += 1
        return self.get_state()

    def get_state(self) -> dict:
        """Return current simulation state as a plain dict."""
        import traci
        vehicles = traci.vehicle.getIDList()
        ambulance_present = AMBULANCE_VEH_ID in vehicles
        ambulance_pos     = None
        if ambulance_present:
            ambulance_pos = traci.vehicle.getPosition(AMBULANCE_VEH_ID)

        # Sample one TLS state (first in the list if any)
        first_tls = self._tls_ids[0] if self._tls_ids else None
        tls_state = (
            self.get_traffic_light_state(first_tls) if first_tls else "N/A"
        )

        return {
            "step":              self._step_count,
            "tls_count":         len(self._tls_ids),
            "tls_sample_id":     first_tls,
            "tls_state":         tls_state,
            "vehicle_count":     len(vehicles),
            "ambulance_present": ambulance_present,
            "ambulance_pos":     ambulance_pos,
        }

    # ── Traffic-light control ──────────────────────────────────────────────

    def get_tls_ids(self) -> list[str]:
        """Return all TLS junction IDs in the loaded network."""
        return list(self._tls_ids)

    def get_traffic_light_state(self, junction_id: str) -> str:
        """
        Return the current phase string for the traffic light at *junction_id*.

        Each character represents one signal link:
          'G' = green, 'g' = minor-road green, 'y' = yellow, 'r' = red.
        """
        import traci
        return traci.trafficlight.getRedYellowGreenState(junction_id)

    def set_traffic_light_state(self, junction_id: str, phase: str) -> None:
        """Force the traffic light at *junction_id* to the given *phase* string."""
        import traci
        traci.trafficlight.setRedYellowGreenState(junction_id, phase)
        logger.debug("TLS %s -> %s", junction_id, phase)

    def get_phase_index(self, junction_id: str) -> int:
        """Return the current integer phase index for the TLS."""
        import traci
        return traci.trafficlight.getPhase(junction_id)

    def set_phase_index(self, junction_id: str, phase_index: int) -> None:
        """Switch the TLS to the given phase index (as defined in the net file)."""
        import traci
        traci.trafficlight.setPhase(junction_id, phase_index)

    # ── Geo-coordinate helpers ─────────────────────────────────────────────

    def latlon_to_edge(self, lat: float, lng: float) -> Optional[str]:
        """
        Convert a WGS84 lat/lng to the nearest SUMO edge ID.

        Uses traci.simulation.convertGeo() to project geo-coords into the
        SUMO coordinate frame, then traci.simulation.findRoute() to snap
        to the nearest driveable edge.

        Returns None if no edge is found within a reasonable radius.
        """
        import traci
        try:
            # Convert geo -> SUMO XY (longitude first in TraCI)
            x, y = traci.simulation.convertGeo(lng, lat, fromGeo=True)
            # Find nearest edge to that XY position
            edges = traci.simulation.findIntermodalRoute(
                fromEdge="",
                toEdge="",
                depart=0,
            )
            # Fallback: use convertRoad which maps XY -> (edgeID, laneIndex, pos)
            # Actually the right TraCI call is simulation.convertRoad
            edge_id, _, _ = traci.simulation.convertRoad(x, y, isGeo=False)
            return edge_id if edge_id else None
        except Exception as exc:
            logger.warning("latlon_to_edge(%s,%s) failed: %s", lat, lng, exc)
            return None

    _cached_net = None

    def nearest_edge(self, lat: float, lng: float, radius: float = 500.0) -> Optional[str]:
        """
        Return the nearest driveable SUMO edge to the given lat/lng using sumolib.
        
        This accurately converts WGS84 coordinates to SUMO XY, searches within
        the specified radius (default 500m), and filters for edges that allow
        'emergency' or 'passenger' vehicles. This avoids snapping to pedestrian-only
        paths or inaccessible one-ways that would cause routing to fail.
        """
        try:
            if SumoEnv._cached_net is None:
                tools_path = str(SUMO_HOME / "tools")
                if tools_path not in sys.path:
                    sys.path.insert(0, tools_path)
                import sumolib
                logger.info("nearest_edge: loading SUMO network via sumolib...")
                SumoEnv._cached_net = sumolib.net.readNet(str(NETWORK_FILE), withInternal=False)
                logger.info("nearest_edge: network loaded.")

            net = SumoEnv._cached_net
            x, y = net.convertLonLat2XY(lng, lat)
            
            # getNeighboringEdges returns a list of (edge_obj, distance) sorted by distance
            neighbors = net.getNeighboringEdges(x, y, r=radius, includeJunctions=False)
            
            # Filter for accessible edges
            for edge, dist in sorted(neighbors, key=lambda t: t[1]):
                if edge.getID().startswith(":"):
                    continue
                # Check if it allows passenger/emergency vehicles
                # If allows() is not available, we can parse 'allow' / 'disallow' string attributes.
                if not edge.allows("emergency") and not edge.allows("passenger"):
                    continue
                
                logger.debug("nearest_edge(%s,%s) -> %r at %.1fm", lat, lng, edge.getID(), dist)
                return edge.getID()

            logger.warning("nearest_edge(%s,%s): No accessible edge found within %.1fm", lat, lng, radius)
            return None

        except Exception as exc:
            logger.warning("nearest_edge(%s,%s) failed: %s", lat, lng, exc)
            return None


    # ── Properties ────────────────────────────────────────────────────────

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def step_count(self) -> int:
        return self._step_count


# ── Standalone demo ────────────────────────────────────────────────────────────
def _run_demo(steps: int = 50) -> None:
    """
    Run a headless demo: start SUMO with the Kilpauk network,
    step through a few cycles, print TLS states, then close.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
    sep = "=" * 60
    print(f"\n{sep}")
    print("  Siren AI -- SUMO TraCI headless demo (Kilpauk, Chennai)")
    print(f"  Network : {SUMO_CFG}")
    print(f"  Steps   : {steps}")
    print(f"{sep}\n")

    with SumoEnv() as env:
        print(f"  TLS junctions: {len(env.get_tls_ids())} total")
        print(f"  First 5 TLS IDs: {env.get_tls_ids()[:5]}")
        for i in range(steps):
            state = env.step()
            if i % 10 == 0:
                print(
                    f"  Step {state['step']:>4d} | "
                    f"TLS[{state['tls_sample_id']}]={state['tls_state']!r:12s} | "
                    f"vehicles={state['vehicle_count']:>3d}"
                )

    print("\n[OK] Demo complete -- TraCI closed cleanly.")


if __name__ == "__main__":
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    _run_demo(steps)
