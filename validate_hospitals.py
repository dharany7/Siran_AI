"""
validate_hospitals.py — One-time hospital routability check.

For every hospital in the database:
  1. Project its lat/lng → SUMO XY using the net.xml UTM projection
     (same logic as backend/routers/route_geometry.py — no SUMO/TraCI needed).
  2. Compute the distance from that XY point to the midpoint of every
     non-internal edge in kilpauk.net.xml.
  3. If the nearest edge midpoint is ≤ MAX_DIST_M metres away, mark the
     hospital is_routable = True; otherwise False.

Usage:
    python validate_hospitals.py [--dry-run] [--max-dist 150]

Flags:
    --dry-run   Print results without writing to the database.
    --max-dist  Maximum acceptable distance in metres (default 150).
"""
from __future__ import annotations

import argparse
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE     = Path(__file__).parent
NET_FILE = HERE / "sumo_net" / "kilpauk.net.xml"

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_MAX_DIST = 150.0   # metres — reject if nearest edge midpoint > this


# ─────────────────────────────────────────────────────────────────────────────
# Network parsing (no TraCI required)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_network(net_file: Path) -> tuple:
    """
    Parse kilpauk.net.xml and return:
        (net_offset, proj_str, edge_midpoints)

    edge_midpoints — list of (sumo_x, sumo_y) midpoints for every
    non-internal edge (one midpoint = average of first lane shape points).
    """
    print(f"Parsing network: {net_file} ...")
    tree = ET.parse(str(net_file))
    root = tree.getroot()

    # ── Projection ────────────────────────────────────────────────────────────
    loc = root.find("location")
    if loc is None:
        raise RuntimeError("<location> element not found in net.xml")
    ox, oy   = [float(v) for v in loc.get("netOffset", "0,0").split(",")]
    proj_str = loc.get("projParameter", "")

    # ── Edge midpoints (SUMO XY) ──────────────────────────────────────────────
    midpoints = []
    for edge_el in root.iter("edge"):
        eid = edge_el.get("id", "")
        if eid.startswith(":"):
            continue   # skip internal connector edges
        for lane_el in edge_el.findall("lane"):
            shape_str = lane_el.get("shape", "")
            if not shape_str:
                continue
            pts = []
            for pair in shape_str.split():
                sx, sy = [float(v) for v in pair.split(",")]
                pts.append((sx, sy))
            if pts:
                mx = sum(p[0] for p in pts) / len(pts)
                my = sum(p[1] for p in pts) / len(pts)
                midpoints.append((mx, my))
            break   # first lane is representative

    print(f"  -> {len(midpoints):,} edge midpoints loaded.")
    return (ox, oy), proj_str, midpoints


def _latlng_to_sumo_xy(lat, lng, net_offset, proj_str):
    """
    Project a WGS84 (lat, lng) to SUMO Cartesian (x, y).

    SUMO net.xml convention (from route_geometry.py):
        utm_xy = sumo_xy - netOffset   →   sumo_xy = utm_xy + netOffset

    netOffset for kilpauk.net.xml is (-413093.52, -1435364.74), so adding it
    shifts the large UTM values into the local SUMO coordinate space
    (convBoundary 0..8283, 0..18450).
    """
    ox, oy = net_offset
    try:
        from pyproj import Proj
        proj         = Proj(proj_str)
        utm_x, utm_y = proj(lng, lat)   # lon first for pyproj
        return utm_x + ox, utm_y + oy   # add offset (not subtract)
    except Exception:
        # Linear fallback (sufficient for 150 m tolerance at Kilpauk scale)
        LAT0, LON0 = 13.083, 80.237
        M_DEG_LAT  = 111_320.0
        M_DEG_LON  = 111_320.0 * math.cos(math.radians(LAT0))
        utm_x = LON0 * M_DEG_LON + (lng - LON0) * M_DEG_LON
        utm_y = LAT0 * M_DEG_LAT + (lat - LAT0) * M_DEG_LAT
        return utm_x + ox, utm_y + oy   # add offset (not subtract)


def _nearest_dist(sx, sy, midpoints):
    """Distance (metres) to the closest edge midpoint in SUMO XY space."""
    best = math.inf
    for mx, my in midpoints:
        d = math.hypot(sx - mx, sy - my)
        if d < best:
            best = d
    return best


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(dry_run: bool, max_dist: float) -> None:
    if not NET_FILE.exists():
        sys.exit(f"ERROR: Network file not found: {NET_FILE}")

    # ── Parse network ─────────────────────────────────────────────────────────
    net_offset, proj_str, midpoints = _parse_network(NET_FILE)
    if not midpoints:
        sys.exit("ERROR: No edge midpoints extracted — is kilpauk.net.xml intact?")

    # ── Load hospitals from DB ────────────────────────────────────────────────
    sys.path.insert(0, str(HERE))
    from backend.database import SessionLocal
    from backend.models   import Hospital

    db = SessionLocal()
    try:
        hospitals = db.query(Hospital).order_by(Hospital.id).all()
    except Exception as exc:
        db.close()
        sys.exit(f"ERROR reading hospitals: {exc}")

    if not hospitals:
        db.close()
        sys.exit("ERROR: No hospitals in DB. Run setup_real_data.py first.")

    print(f"\nChecking {len(hospitals)} hospitals against kilpauk.net.xml")
    print(f"Threshold: {max_dist} m  |  dry_run: {dry_run}\n")
    print(f"{'ID':>4}  {'Name':<46}  {'Dist(m)':>8}  Status")
    print("-" * 76)

    passed, failed = [], []

    for h in hospitals:
        sx, sy   = _latlng_to_sumo_xy(h.lat, h.lng, net_offset, proj_str)
        dist     = _nearest_dist(sx, sy, midpoints)
        routable = dist <= max_dist
        status   = "PASS" if routable else "FAIL"
        print(f"{h.id:>4}  {h.name:<46}  {dist:>8.1f}  {status}")

        if routable:
            passed.append((h.id, h.name, dist))
        else:
            failed.append((h.id, h.name, dist, h.lat, h.lng))

        if not dry_run:
            h.is_routable = routable

    print("-" * 76)
    print(f"\nResult: {len(passed)} PASS  /  {len(failed)} FAIL  "
          f"(threshold {max_dist} m, total {len(hospitals)})\n")

    if failed:
        print("Failing hospitals (outside SUMO network coverage):")
        for hid, name, dist, lat, lng in failed:
            print(f"  [{hid:>3}] {name}  ({lat}, {lng})  nearest={dist:.0f}m")
        print()

    if dry_run:
        print("[DRY-RUN] No database changes written.")
    else:
        try:
            db.commit()
            print(f"[OK] is_routable column updated for all {len(hospitals)} hospitals.")
        except Exception as exc:
            db.rollback()
            sys.exit(f"ERROR committing to DB: {exc}")
        finally:
            db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate hospitals against kilpauk.net.xml and set is_routable.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Print results without writing to the database")
    parser.add_argument("--max-dist", type=float, default=DEFAULT_MAX_DIST,
                        help=f"Max distance to nearest edge in metres (default {DEFAULT_MAX_DIST})")
    args = parser.parse_args()
    main(dry_run=args.dry_run, max_dist=args.max_dist)
