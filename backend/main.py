"""
main.py — FastAPI application factory and entry point.

Run with:
    uvicorn backend.main:app --reload
or use the project root convenience script:
    python run.py
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.config import settings
from backend.database import init_db
from backend.routers import health, sim, start, security, audio, anpr, ws, auth, acoustic_demo
from backend.routers.route_geometry import router as route_geometry_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: initialise DB and auto-start SUMO simulation."""
    import asyncio, logging
    _log = logging.getLogger(__name__)

    # ── Startup ───────────────────────────────────────────────────────────────
    init_db()

    # Start SUMO in a thread-pool executor so the blocking subprocess I/O
    # does not stall the asyncio event loop during startup.
    def _start_sim():
        try:
            from agents.sumo_env import SumoEnv
            sim = SumoEnv()
            sim.start()
            app.state.sim = sim
            _log.info("✅ SUMO simulation auto-started (step=%d)", sim.step_count)
        except FileNotFoundError as exc:
            _log.error("❌ SUMO binary/config not found — sim not started: %s", exc)
            app.state.sim = None
        except Exception as exc:
            _log.error("❌ SUMO auto-start failed: %s", exc)
            app.state.sim = None

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _start_sim)

    # ── Startup snap-check: update is_routable for all hospitals ─────────────
    # Uses sumolib (no TraCI needed) so it runs even if SUMO is offline.
    # Excluded hospitals are logged at WARNING level; no UI error is shown.
    def _snap_check():
        import math, xml.etree.ElementTree as ET, os, sys
        from pathlib import Path
        try:
            from backend.database import SessionLocal
            from backend.models   import Hospital

            net_path = Path(__file__).parent.parent / "sumo_net" / "kilpauk.net.xml"
            if not net_path.exists():
                _log.warning("snap_check: net.xml not found — skipping")
                return

            # Parse edge midpoints without sumolib (avoids optional dep)
            tree = ET.parse(str(net_path))
            root = tree.getroot()
            loc  = root.find("location")
            ox, oy   = [float(v) for v in loc.get("netOffset", "0,0").split(",")]
            proj_str = loc.get("projParameter", "")

            midpoints = []
            for edge_el in root.iter("edge"):
                if edge_el.get("id", "").startswith(":"):
                    continue
                for lane_el in edge_el.findall("lane"):
                    shape = lane_el.get("shape", "")
                    if not shape:
                        continue
                    pts = [tuple(float(v) for v in p.split(",")) for p in shape.split()]
                    if pts:
                        midpoints.append((
                            sum(p[0] for p in pts) / len(pts),
                            sum(p[1] for p in pts) / len(pts),
                        ))
                    break

            def _project(lat, lng):
                try:
                    from pyproj import Proj
                    ux, uy = Proj(proj_str)(lng, lat)
                    return ux + ox, uy + oy
                except Exception:
                    LAT0, LON0 = 13.083, 80.237
                    M = 111_320.0
                    import math
                    ux = LON0 * M + (lng - LON0) * M
                    uy = LAT0 * M + (lat - LAT0) * (M * math.cos(math.radians(LAT0)))
                    return ux + ox, uy + oy

            MAX_DIST = 200.0   # metres — matches nearest_edge(radius=) in sumo_env.py
            db = SessionLocal()
            try:
                hospitals = db.query(Hospital).order_by(Hospital.id).all()
                excluded, updated = [], 0
                for h in hospitals:
                    sx, sy = _project(h.lat, h.lng)
                    dist   = min(math.hypot(sx - mx, sy - my) for mx, my in midpoints)
                    routable = dist <= MAX_DIST
                    if h.is_routable != routable:
                        h.is_routable = routable
                        updated += 1
                    if not routable:
                        excluded.append((h.name, h.lat, h.lng, dist))
                db.commit()
                _log.info(
                    "snap_check: %d hospitals checked, %d updated, %d excluded",
                    len(hospitals), updated, len(excluded),
                )
                for name, lat, lng, dist in excluded:
                    _log.warning(
                        "snap_check EXCLUDED: %r (%.6f, %.6f) nearest_edge=%.1fm > %.0fm",
                        name, lat, lng, dist, MAX_DIST,
                    )
            finally:
                db.close()
        except Exception as exc:
            _log.warning("snap_check: failed (hospitals not filtered): %s", exc)

    await loop.run_in_executor(None, _snap_check)

    yield  # ── Application runs ─────────────────────────────────────────────

    # ── Shutdown ──────────────────────────────────────────────────────────────
    sim = getattr(app.state, "sim", None)
    if sim is not None and getattr(sim, "is_connected", False):
        try:
            sim.close()
            _log.info("🛑 SUMO simulation closed on server shutdown.")
        except Exception as exc:
            _log.warning("SUMO close error on shutdown: %s", exc)


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        description=(
            "Siren AI — Multi-Agent System for real-time siren detection, "
            "ANPR, and emergency response optimisation."
        ),
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── CORS ──────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.debug else [],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routers ───────────────────────────────────────────────────────────────
    app.include_router(health.router)
    app.include_router(auth.router)         # /auth/signup, /auth/login
    app.include_router(sim.router)
    app.include_router(start.router)
    app.include_router(security.router)
    app.include_router(audio.router)
    app.include_router(anpr.router)
    app.include_router(ws.router)       # WebSocket live-feed hub
    app.include_router(route_geometry_router)  # GET /route/geometry
    app.include_router(acoustic_demo.router)   # POST /junction/{id}/acoustic-event
    from backend.routers import corridor
    app.include_router(corridor.router)   # POST /corridor/sensor-trigger

    # ── Hospitals listing ─────────────────────────────────────────────────────
    from fastapi import Depends
    from sqlalchemy.orm import Session
    from backend.database import get_db
    from backend.models import Hospital

    @app.get("/hospitals", tags=["Hospitals"], summary="List all routable hospitals")
    def list_hospitals(db: Session = Depends(get_db)):
        """
        Returns only hospitals whose coordinates can be snapped to a SUMO edge
        within 200 m (``is_routable = True``).  Non-routable hospitals are
        excluded silently — they are filtered at server startup by the
        automatic snap-check and logged to the backend console.
        Run ``python validate_hospitals.py`` to re-populate the flag manually.
        """
        hosps = (
            db.query(Hospital)
            .filter(Hospital.is_routable == True)   # noqa: E712 — SQLAlchemy requires ==
            .order_by(Hospital.id)
            .all()
        )
        return [
            {
                "id":          h.id,
                "name":        h.name,
                "lat":         h.lat,
                "lng":         h.lng,
                "address":     h.address,
                "is_routable": h.is_routable,
            }
            for h in hosps
        ]


    # ── Static frontend ───────────────────────────────────────────────────────
    # Serve at root so http://localhost:8000/ opens the dashboard directly.
    # /dashboard kept as an alias for backwards-compatibility.
    app.mount("/dashboard", StaticFiles(directory="frontend", html=True), name="frontend-alias")
    app.mount("/",          StaticFiles(directory="frontend", html=True), name="frontend")

    return app


app = create_app()
