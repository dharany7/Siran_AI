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
from backend.routers import health, sim, dispatch, security, audio, anpr, ws, auth
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
    app.include_router(dispatch.router)
    app.include_router(security.router)
    app.include_router(audio.router)
    app.include_router(anpr.router)
    app.include_router(ws.router)       # WebSocket live-feed hub
    app.include_router(route_geometry_router)  # GET /route/geometry

    # ── Hospitals listing ─────────────────────────────────────────────────────
    from fastapi import Depends
    from sqlalchemy.orm import Session
    from backend.database import get_db
    from backend.models import Hospital

    @app.get("/hospitals", tags=["Hospitals"], summary="List all routable hospitals")
    def list_hospitals(db: Session = Depends(get_db)):
        """
        Returns only hospitals whose coordinates can be snapped to a SUMO edge
        within 150 m (``is_routable = True``).  Non-routable hospitals cause
        dispatch to fail immediately, so they are excluded from the driver's list.
        Run ``python validate_hospitals.py`` to (re-)populate the flag.
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
