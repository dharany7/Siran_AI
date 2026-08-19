"""
backend/routers/route_geometry.py

GET /route/geometry?edges=edgeA,edgeB,edgeC...

Parses kilpauk.net.xml to extract the real WGS84 polyline for a sequence
of SUMO edge IDs, as returned by duarouter.

Projection info from the net.xml location block:
  netOffset = "-413093.52,-1435364.74"
  projParameter = "+proj=utm +zone=44 +ellps=WGS84 +datum=WGS84 +units=m +no_defs"

SUMO XY → real-world conversion:
  utm_x = sumo_x + offset_x     (add the stored offset back)
  utm_y = sumo_y + offset_y
  lon, lat = pyproj.Proj(proj_string)(utm_x, utm_y, inverse=True)
"""
from __future__ import annotations

import logging
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/route", tags=["Route"])

_HERE    = Path(__file__).parent.parent.parent   # project root
NET_FILE = _HERE / "sumo_net" / "kilpauk.net.xml"

# ── Cache ─────────────────────────────────────────────────────────────────────
# Populated once on first request to avoid re-parsing 14 MB XML every call.
_edge_shapes: Optional[dict[str, list[tuple[float, float]]]] = None  # edge_id → [(x,y), ...]
_net_offset:  Optional[tuple[float, float]] = None
_proj_str:    Optional[str] = None


def _parse_net() -> None:
    """Parse kilpauk.net.xml once and cache edge shapes + projection info."""
    global _edge_shapes, _net_offset, _proj_str

    logger.info("Parsing SUMO net: %s …", NET_FILE)
    tree = ET.parse(str(NET_FILE))
    root = tree.getroot()

    # ── Location / projection ─────────────────────────────────────────────
    loc = root.find("location")
    if loc is None:
        raise RuntimeError("<location> element not found in net.xml")

    offset_str = loc.get("netOffset", "0,0")
    ox, oy     = [float(v) for v in offset_str.split(",")]
    _net_offset = (ox, oy)
    _proj_str   = loc.get("projParameter", "")
    logger.info("Net offset: (%s, %s)  proj: %s", ox, oy, _proj_str)

    # ── Edge shapes (use first lane of each edge) ─────────────────────────
    shapes: dict[str, list[tuple[float, float]]] = {}
    for edge_el in root.iter("edge"):
        eid = edge_el.get("id", "")
        if eid.startswith(":"):    # internal junction edge — skip
            continue
        for lane_el in edge_el.findall("lane"):
            shape_str = lane_el.get("shape", "")
            if shape_str:
                pts = []
                for pair in shape_str.split():
                    sx, sy = [float(v) for v in pair.split(",")]
                    pts.append((sx, sy))
                if pts:
                    shapes[eid] = pts
                break   # first lane is representative

    _edge_shapes = shapes
    logger.info("Cached %d edge shapes", len(shapes))


def _sumo_xy_to_latlng(sx: float, sy: float) -> tuple[float, float]:
    """
    Convert SUMO Cartesian (sx, sy) → (lat, lng) WGS84.

    SUMO stores coordinates as:  sumo_xy = real_utm - netOffset
    So:  real_utm = sumo_xy - netOffset   (note: minus, not plus)
    Then invert the UTM projection → lon/lat.

    Verified: edge '-1039027733#0' at SUMO (3923, 10323) → lat=13.076 lon=80.234 ✓
    """
    ox, oy = _net_offset
    utm_x  = sx - ox   # subtract offset (SUMO convention)
    utm_y  = sy - oy

    try:
        from pyproj import Proj  # noqa: PLC0415
        proj     = Proj(_proj_str)
        lon, lat = proj(utm_x, utm_y, inverse=True)
        return float(lat), float(lon)
    except Exception as exc:
        logger.warning("pyproj conversion failed: %s — using linear approx", exc)
        import math
        LAT0, LON0 = 13.083, 80.237
        M_DEG_LAT  = 111_320.0
        M_DEG_LON  = 111_320.0 * math.cos(math.radians(LAT0))
        lat = LAT0 + utm_y / M_DEG_LAT
        lon = LON0 + utm_x / M_DEG_LON
        return lat, lon



def _edges_to_coordinates(edge_ids: list[str]) -> list[list[float]]:
    """Convert a list of SUMO edge IDs → [[lat, lng], ...] for Leaflet."""
    if _edge_shapes is None or _net_offset is None:
        _parse_net()

    coords: list[list[float]] = []
    seen:   set[tuple]        = set()
    missing = 0

    for eid in edge_ids:
        # Try exact match first, then reversed-sign variant
        pts = _edge_shapes.get(eid)
        if pts is None and eid.startswith("-"):
            pts = _edge_shapes.get(eid[1:])  # try without leading '-'
            if pts:
                pts = list(reversed(pts))     # reverse direction
        if pts is None and not eid.startswith("-"):
            pts = _edge_shapes.get(f"-{eid}")
            if pts:
                pts = list(reversed(pts))

        if pts is None:
            missing += 1
            logger.debug("Edge not in cache: %r", eid)
            continue

        for sx, sy in pts:
            lat, lon = _sumo_xy_to_latlng(sx, sy)
            pt = (round(lat, 7), round(lon, 7))
            if pt not in seen:
                seen.add(pt)
                coords.append([pt[0], pt[1]])

    if missing:
        logger.info("%d/%d edges not found in net", missing, len(edge_ids))

    return coords


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.get(
    "/geometry",
    summary="Convert SUMO edge IDs → WGS84 polyline coordinates",
    description=(
        "Parses `kilpauk.net.xml` and returns `[[lat, lng], ...]` for the given "
        "comma-separated SUMO edge IDs. Used by the driver frontend to draw the "
        "real street-following route on the Leaflet map. "
        "No SUMO/TraCI connection required — reads the static net file only."
    ),
)
def route_geometry(
    edges: str = Query(
        ...,
        description="Comma-separated SUMO edge IDs (as returned by POST /start in route_edges)",
    ),
    max_edges: int = Query(300, description="Limit edges processed (for performance)"),
):
    if not NET_FILE.exists():
        raise HTTPException(
            status_code=503,
            detail=f"SUMO network file not found at {NET_FILE}.",
        )

    edge_ids = [e.strip() for e in edges.split(",") if e.strip()][:max_edges]
    if not edge_ids:
        raise HTTPException(status_code=422, detail="No edge IDs provided.")

    try:
        coords = _edges_to_coordinates(edge_ids)
    except Exception as exc:
        logger.exception("Geometry conversion error")
        raise HTTPException(status_code=500, detail=f"Geometry conversion failed: {exc}")

    if not coords:
        raise HTTPException(
            status_code=404,
            detail=(
                f"None of the {len(edge_ids)} provided edge IDs were found in "
                f"kilpauk.net.xml. Confirm edge IDs match the current network."
            ),
        )

    return {
        "edge_count":  len(edge_ids),
        "point_count": len(coords),
        "coordinates": coords,   # [[lat, lng], ...] — Leaflet L.polyline ready
    }


# ── Junction coordinate resolver ──────────────────────────────────────────────

# Populated lazily alongside _edge_shapes — holds junction_id → (sumo_x, sumo_y)
_junction_xy: dict[str, tuple[float, float]] | None = None


def _ensure_junction_cache() -> None:
    """
    Populate _junction_xy from kilpauk.net.xml <junction> elements.
    Calls _parse_net() first so the projection info is available.
    """
    global _junction_xy
    if _junction_xy is not None:
        return
    if _net_offset is None:
        _parse_net()   # also fills _net_offset / _proj_str

    tree = ET.parse(str(NET_FILE))
    root = tree.getroot()

    jmap: dict[str, tuple[float, float]] = {}
    for jel in root.iter("junction"):
        jid = jel.get("id", "")
        if jid.startswith(":"):
            continue   # internal connector junctions — skip
        try:
            jmap[jid] = (float(jel.get("x", 0)), float(jel.get("y", 0)))
        except ValueError:
            pass

    _junction_xy = jmap
    logger.info("Cached %d junction positions", len(jmap))


@router.get(
    "/junctions",
    summary="Convert SUMO junction IDs → WGS84 lat/lng points",
    description=(
        "Parses `kilpauk.net.xml` `<junction>` elements and returns `{junction_id: [lat, lng]}`"
        " for the given comma-separated junction IDs.  Used by the driver frontend to place "
        "traffic-signal markers on the Leaflet map during an active start."
    ),
)
def junction_positions(
    ids: str = Query(
        ...,
        description="Comma-separated SUMO/OSM junction IDs (as found in tls_schedule[].junction_id)",
    ),
):
    if not NET_FILE.exists():
        raise HTTPException(
            status_code=503,
            detail=f"SUMO network file not found at {NET_FILE}.",
        )

    requested = [j.strip() for j in ids.split(",") if j.strip()]
    if not requested:
        raise HTTPException(status_code=422, detail="No junction IDs provided.")

    try:
        _ensure_junction_cache()
    except Exception as exc:
        logger.exception("Junction cache build error")
        raise HTTPException(status_code=500, detail=f"Junction cache failed: {exc}")

    result: dict[str, list[float]] = {}
    missing: list[str] = []

    for jid in requested:
        xy = _junction_xy.get(jid)  # type: ignore[union-attr]
        if xy is None:
            missing.append(jid)
            continue
        lat, lon = _sumo_xy_to_latlng(xy[0], xy[1])
        result[jid] = [round(lat, 7), round(lon, 7)]

    if missing:
        logger.info("Junction IDs not found in net: %s", missing)

    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"None of {requested} found in kilpauk.net.xml junction list.",
        )

    return {
        "found":   len(result),
        "missing": missing,
        "positions": result,   # { junction_id: [lat, lng] }
    }
