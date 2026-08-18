"""
widen_network.py
================
Widens the Siren AI SUMO road network so every hospital in hospitals.json
falls inside network coverage and snaps to an edge.

Steps
-----
1. Compute bounding box from hospitals.json with +0.005 deg padding.
2. Re-download OSM for the wider bbox (overwrites kilpauk.osm).
3. Re-run netconvert to regenerate kilpauk.net.xml.
4. Re-generate background traffic (randomTrips.py / duarouter).
5. Rewrite kilpauk.sumocfg with the new bbox comment.
6. Run hospital-snapping validation against the new network (no DB write required
   unless --update-db flag is passed).
7. Time a duarouter route across the wider network and log latency.

Usage
-----
    python widen_network.py                  # dry-run snap check (no DB write)
    python widen_network.py --update-db      # also update is_routable in DB
    python widen_network.py --skip-download  # skip OSM+netconvert (reuse existing files)
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

# Force UTF-8 output on Windows (avoids cp1252 UnicodeEncodeError)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import requests

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
SUMO_NET_DIR = PROJECT_ROOT / "sumo_net"
SUMO_HOME    = Path(os.environ.get("SUMO_HOME", r"C:\Program Files (x86)\Eclipse\Sumo"))
NETCONVERT   = SUMO_HOME / "bin" / "netconvert.exe"
DUAROUTER    = SUMO_HOME / "bin" / "duarouter.exe"
RANDOM_TRIPS = SUMO_HOME / "tools" / "randomTrips.py"

OSM_FILE  = SUMO_NET_DIR / "kilpauk.osm"
NET_FILE  = SUMO_NET_DIR / "kilpauk.net.xml"
ROU_FILE  = SUMO_NET_DIR / "kilpauk_bg.rou.xml"
CFG_FILE  = SUMO_NET_DIR / "kilpauk.sumocfg"
HOSP_JSON = SUMO_NET_DIR / "hospitals.json"

OVERPASS_HEADERS = {
    "User-Agent": "SirenAI/1.0 (emergency-vehicle routing research; contact@example.com)",
    "Accept":     "application/xml, text/xml, */*",
}

PAD_DEG      = 0.005   # padding added to each side of the hospital bbox
DEFAULT_MAX_DIST = 200  # metres — must match nearest_edge(radius=) in sumo_env.py

SEP = "=" * 70


def _sep(n: int, msg: str) -> None:
    n_str = str(n)
    print(f"\n[{n_str}/7] {msg}")
    print("-" * 60)


# ── Step 1: Compute bbox ───────────────────────────────────────────────────────

def compute_bbox(pad: float = PAD_DEG) -> dict:
    _sep(1, "Computing bounding box from hospitals.json ...")

    if not HOSP_JSON.exists():
        sys.exit(f"ERROR: {HOSP_JSON} not found. Run setup_real_network.py first.")

    hospitals = json.loads(HOSP_JSON.read_text(encoding="utf-8"))
    lats = [h["lat"] for h in hospitals]
    lngs = [h["lng"] for h in hospitals]

    bbox = {
        "south": min(lats) - pad,
        "north": max(lats) + pad,
        "west":  min(lngs) - pad,
        "east":  max(lngs) + pad,
    }

    print(f"  Hospitals       : {len(hospitals)}")
    print(f"  lat range       : {min(lats):.6f} .. {max(lats):.6f}")
    print(f"  lng range       : {min(lngs):.6f} .. {max(lngs):.6f}")
    print(f"  Padding         : ±{pad}°")
    print(f"  NEW BBOX        : S={bbox['south']:.4f}  W={bbox['west']:.4f}"
          f"  N={bbox['north']:.4f}  E={bbox['east']:.4f}")

    old = dict(south=13.070, west=80.227, north=13.097, east=80.253)
    outside_old = [h for h in hospitals
                   if not (old["south"] <= h["lat"] <= old["north"] and
                           old["west"]  <= h["lng"] <= old["east"])]
    if outside_old:
        print(f"\n  Hospitals outside OLD bbox ({len(outside_old)}):")
        for h in outside_old:
            print(f"    {h['name']:50s}  ({h['lat']:.6f}, {h['lng']:.6f})")
    else:
        print("  All hospitals already inside old bbox (but padding ensures edge snap).")

    return bbox


# ── Step 2: Download OSM ───────────────────────────────────────────────────────

def download_osm(bbox: dict) -> None:
    _sep(2, "Downloading OSM for wider bbox from Overpass ...")

    overpass_url = "https://overpass-api.de/api/interpreter"
    q = (
        f"[out:xml][timeout:180];"
        f"(way[\"highway\"]({bbox['south']},{bbox['west']},{bbox['north']},{bbox['east']});"
        f"node[\"highway\"]({bbox['south']},{bbox['west']},{bbox['north']},{bbox['east']});"
        f"relation[\"highway\"]({bbox['south']},{bbox['west']},{bbox['north']},{bbox['east']}););"
        f"out body;>;out skel qt;"
    )

    for attempt in range(1, 4):
        try:
            print(f"  Attempt {attempt}/3 ...")
            r = requests.get(overpass_url, params={"data": q},
                             headers=OVERPASS_HEADERS, timeout=240)
            r.raise_for_status()
            osm_bytes = r.content
            print(f"  Downloaded {len(osm_bytes)/1e6:.1f} MB")
            break
        except Exception as exc:
            print(f"  WARN: attempt {attempt} failed: {exc}")
            if attempt < 3:
                overpass_url = "https://overpass.kumi.systems/api/interpreter"
                time.sleep(5)
            else:
                # Fallback: bbox export
                url2 = (f"https://overpass-api.de/api/map?"
                        f"bbox={bbox['west']},{bbox['south']},{bbox['east']},{bbox['north']}")
                print(f"  Trying fallback: {url2}")
                try:
                    r2 = requests.get(url2, headers=OVERPASS_HEADERS, timeout=240)
                    r2.raise_for_status()
                    osm_bytes = r2.content
                    print(f"  Fallback download: {len(osm_bytes)/1e6:.1f} MB")
                    break
                except Exception as exc2:
                    sys.exit(f"ERROR: All OSM download attempts failed: {exc2}")

    OSM_FILE.write_bytes(osm_bytes)
    print(f"  Saved -> {OSM_FILE}  ({OSM_FILE.stat().st_size/1e6:.1f} MB)")


# ── Step 3: Run netconvert ─────────────────────────────────────────────────────

def run_netconvert() -> tuple[int, int, int]:
    _sep(3, "Converting OSM -> SUMO net.xml (netconvert) ...")

    if not NETCONVERT.exists():
        sys.exit(f"ERROR: netconvert not found at {NETCONVERT}. Set SUMO_HOME.")

    cmd = [
        str(NETCONVERT),
        "--osm-files",        str(OSM_FILE),
        "-o",                 str(NET_FILE),
        "--tls.discard-simple",
        "--tls.join",
        "--osm.bike-access",
        "--geometry.remove",
        "--roundabouts.guess",
        "--junctions.join",
        "--no-internal-links",
        "--keep-edges.by-vclass", "passenger",
        "--no-warnings",
    ]

    print("  Running netconvert ...")
    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0

    if result.returncode != 0:
        print("  netconvert STDERR:", result.stderr[-2000:])
        sys.exit("ERROR: netconvert failed.")

    net_xml    = NET_FILE.read_text(encoding="utf-8", errors="ignore")
    junc_count = net_xml.count("<junction ")
    edge_count = net_xml.count('<edge id="')
    tls_count  = net_xml.count("<tlLogic ")
    size_mb    = NET_FILE.stat().st_size / 1e6

    print(f"  Done in {elapsed:.1f}s")
    print(f"  Network : {junc_count} junctions, {edge_count} edges, {tls_count} TLS")
    print(f"  File    : {NET_FILE.name}  ({size_mb:.1f} MB)")
    return junc_count, edge_count, tls_count


# ── Step 4: Regenerate background traffic ─────────────────────────────────────

def regen_traffic() -> None:
    _sep(4, "Regenerating background traffic (randomTrips.py) ...")

    if not RANDOM_TRIPS.exists():
        print(f"  WARN: randomTrips.py not found at {RANDOM_TRIPS} — skipping.")
        _write_empty_rou()
        return

    trips_file = SUMO_NET_DIR / "kilpauk_trips.xml"
    cmd = [
        sys.executable, str(RANDOM_TRIPS),
        "--net-file", str(NET_FILE),
        "-o",         str(trips_file),
        "-r",         str(ROU_FILE),
        "--end",      "3600",
        "--period",   "8",
        "--validate",
        "--remove-loops",
        "--seed",     "42",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(SUMO_NET_DIR))
    if result.returncode != 0:
        print("  WARN: randomTrips.py errors:", result.stderr[-300:])
        _write_empty_rou()
    else:
        count = ROU_FILE.read_text().count("<vehicle")
        print(f"  Generated ~{count} background vehicles -> {ROU_FILE.name}")


def _write_empty_rou() -> None:
    ROU_FILE.write_text(
        '<?xml version="1.0"?>\n<routes '
        'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
        'xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">'
        '\n</routes>\n'
    )
    print("  Fallback: empty route file written.")


# ── Step 5: Rewrite sumocfg ────────────────────────────────────────────────────

def rewrite_sumocfg(bbox: dict) -> None:
    _sep(5, "Rewriting kilpauk.sumocfg ...")

    cfg = f"""<?xml version="1.0" encoding="UTF-8"?>
<!--
  kilpauk.sumocfg - SUMO config for Siren AI real-world network
  Area: Kilpauk + surrounding medical district, Chennai, India
  OSM bbox: S={bbox['south']:.4f} W={bbox['west']:.4f} N={bbox['north']:.4f} E={bbox['east']:.4f}
  Generated by widen_network.py
-->
<sumoConfiguration
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
    xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">

    <input>
        <net-file   value="kilpauk.net.xml"/>
        <route-files value="kilpauk_bg.rou.xml"/>
    </input>

    <time>
        <begin value="0"/>
        <end   value="3600"/>
    </time>

    <report>
        <verbose     value="false"/>
        <no-step-log value="true"/>
        <no-warnings value="true"/>
    </report>

    <processing>
        <ignore-route-errors value="true"/>
    </processing>

</sumoConfiguration>
"""
    CFG_FILE.write_text(cfg, encoding="utf-8")
    print(f"  Saved -> {CFG_FILE}")


# ── Step 6: Hospital snap validation ──────────────────────────────────────────

def _parse_network(net_file: Path) -> tuple:
    tree = ET.parse(str(net_file))
    root = tree.getroot()
    loc  = root.find("location")
    if loc is None:
        raise RuntimeError("<location> element not found in net.xml")
    ox, oy   = [float(v) for v in loc.get("netOffset", "0,0").split(",")]
    proj_str = loc.get("projParameter", "")

    midpoints = []
    for edge_el in root.iter("edge"):
        eid = edge_el.get("id", "")
        if eid.startswith(":"):
            continue
        for lane_el in edge_el.findall("lane"):
            shape_str = lane_el.get("shape", "")
            if not shape_str:
                continue
            pts = [tuple(float(v) for v in p.split(",")) for p in shape_str.split()]
            if pts:
                mx = sum(p[0] for p in pts) / len(pts)
                my = sum(p[1] for p in pts) / len(pts)
                midpoints.append((mx, my))
            break
    return (ox, oy), proj_str, midpoints


def _latlng_to_sumo_xy(lat, lng, net_offset, proj_str):
    ox, oy = net_offset
    try:
        from pyproj import Proj
        proj = Proj(proj_str)
        ux, uy = proj(lng, lat)
        return ux + ox, uy + oy
    except Exception:
        LAT0, LON0 = 13.083, 80.237
        M_LAT = 111_320.0
        M_LON = 111_320.0 * math.cos(math.radians(LAT0))
        ux = LON0 * M_LON + (lng - LON0) * M_LON
        uy = LAT0 * M_LAT + (lat - LAT0) * M_LAT
        return ux + ox, uy + oy


def validate_snapping(max_dist: float, update_db: bool) -> tuple[list, list]:
    _sep(6, f"Validating hospital snapping (threshold={max_dist}m) ...")

    print(f"  Parsing {NET_FILE.name} ...")
    t0 = time.perf_counter()
    net_offset, proj_str, midpoints = _parse_network(NET_FILE)
    parse_ms = (time.perf_counter() - t0) * 1000
    print(f"  {len(midpoints):,} edge midpoints loaded in {parse_ms:.0f}ms")

    hospitals = json.loads(HOSP_JSON.read_text(encoding="utf-8"))
    print(f"\n  {'Name':<50}  {'Dist(m)':>8}  Status")
    print("  " + "-" * 66)

    passed, failed = [], []
    for h in hospitals:
        sx, sy = _latlng_to_sumo_xy(h["lat"], h["lng"], net_offset, proj_str)
        dist   = min(math.hypot(sx - mx, sy - my) for mx, my in midpoints)
        ok     = dist <= max_dist
        status = "PASS" if ok else "FAIL"
        print(f"  {h['name']:<50}  {dist:>8.1f}  {status}")
        (passed if ok else failed).append(dict(**h, dist_m=round(dist, 1)))

    print("  " + "-" * 66)
    print(f"\n  Result: {len(passed)} PASS  /  {len(failed)} FAIL  "
          f"(threshold {max_dist}m, total {len(hospitals)})\n")

    if failed:
        print("  FAIL — hospitals still outside network coverage:")
        for h in failed:
            print(f"    {h['name']:50s}  ({h['lat']}, {h['lng']})  nearest={h['dist_m']}m")
    else:
        print("  All hospitals snap successfully — zero failures.")

    if update_db:
        _update_db_routability(hospitals, max_dist)

    return passed, failed


def _update_db_routability(hospitals, max_dist):
    print("\n  Updating is_routable in database ...")
    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from backend.database import SessionLocal
        from backend.models   import Hospital as HospitalModel
        db = SessionLocal()
        net_offset, proj_str, midpoints = _parse_network(NET_FILE)
        for h in hospitals:
            sx, sy = _latlng_to_sumo_xy(h["lat"], h["lng"], net_offset, proj_str)
            dist   = min(math.hypot(sx - mx, sy - my) for mx, my in midpoints)
            routable = dist <= max_dist
            row = db.query(HospitalModel).filter(HospitalModel.name == h["name"]).first()
            if row:
                row.is_routable = routable
        db.commit()
        db.close()
        print("  [OK] is_routable updated for all hospitals.")
    except Exception as exc:
        print(f"  WARN: DB update failed: {exc}")


# ── Step 7: Duarouter latency benchmark ───────────────────────────────────────

def benchmark_duarouter() -> None:
    _sep(7, "Timing duarouter route across wider network ...")

    if not DUAROUTER.exists():
        print(f"  WARN: duarouter not found at {DUAROUTER} — skipping benchmark.")
        return

    # Parse network to pick two routable edge IDs far apart.
    # Strategy: collect all non-internal edge IDs, then test with duarouter
    # until we find a from/to pair that produces a valid route.
    print("  Selecting benchmark edges from network ...")
    tree = ET.parse(str(NET_FILE))
    root = tree.getroot()
    edges = []
    for e in root.iter("edge"):
        eid = e.get("id", "")
        # Skip internal connectors and edges with no lanes
        if eid.startswith(":"):
            continue
        if not e.findall("lane"):
            continue
        # Only include edges that allow passenger vehicles
        allow = e.get("allow", "")
        disallow = e.get("disallow", "")
        if "passenger" in disallow:
            continue
        edges.append(eid)

    if len(edges) < 10:
        print("  WARN: Too few routable edges for benchmark — skipping.")
        return

    # Sample candidate pairs spread across the edge list
    n_edges   = len(edges)
    candidates = [
        (edges[0],              edges[n_edges // 2]),
        (edges[1],              edges[n_edges * 2 // 3]),
        (edges[n_edges // 4],  edges[n_edges * 3 // 4]),
        (edges[n_edges // 3],  edges[n_edges - 2]),
    ]
    print(f"  {n_edges} candidate edges; will try up to {len(candidates)} from/to pairs.")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        trips_xml  = tmpdir / "bench_trip.xml"
        routes_xml = tmpdir / "bench_routes.xml"

        found = False
        for from_edge, to_edge in candidates:
            if from_edge == to_edge:
                continue
            trips_xml.write_text(
                '<?xml version="1.0"?>\n<trips>\n'
                f'  <trip id="bench" depart="0" from="{from_edge}" to="{to_edge}"/>\n'
                '</trips>\n'
            )
            cmd = [
                str(DUAROUTER),
                "--net-file",   str(NET_FILE),
                "--route-files", str(trips_xml),
                "-o",           str(routes_xml),
                "--no-warnings",
                "--error-log",  str(tmpdir / "dua_errors.log"),
            ]
            t0 = time.perf_counter()
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            if result.returncode == 0 and routes_xml.exists():
                route_xml = routes_xml.read_text()
                edge_count = route_xml.count(" edges=") or route_xml.count("<edge ")
                print(f"  from={from_edge!r}  to={to_edge!r}")
                print(f"  Route computed in {elapsed_ms:.0f}ms  ({edge_count} route segments)")
                if elapsed_ms < 500:
                    print("  [OK] Latency acceptable (< 500ms)")
                elif elapsed_ms < 2000:
                    print("  [OK] Latency acceptable (< 2s)")
                else:
                    print(f"  [WARN] Latency {elapsed_ms:.0f}ms may impact dispatch")
                found = True
                break
            else:
                print(f"  Pair {from_edge!r} -> {to_edge!r} not routable, trying next ...")

        if not found:
            print("  WARN: No routable pair found in candidates — duarouter benchmark skipped.")
            print("  (This is a known issue if the network has disconnected components.)")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Widen the Siren AI SUMO road network to cover all hospitals.",
    )
    parser.add_argument("--skip-download",  action="store_true",
                        help="Skip OSM download + netconvert (reuse existing files)")
    parser.add_argument("--update-db",      action="store_true",
                        help="Update is_routable in the database after snapping check")
    parser.add_argument("--max-dist",       type=float, default=DEFAULT_MAX_DIST,
                        help=f"Max snap distance in metres (default {DEFAULT_MAX_DIST})")
    parser.add_argument("--pad",            type=float, default=PAD_DEG,
                        help=f"Bbox padding in degrees (default {PAD_DEG})")
    args = parser.parse_args()

    print(SEP)
    print("  Siren AI — Network Widening Script")
    print(SEP)

    bbox = compute_bbox(pad=args.pad)

    if not args.skip_download:
        download_osm(bbox)
        junc, edges, tls = run_netconvert()
        regen_traffic()
        rewrite_sumocfg(bbox)
    else:
        print("\n[--skip-download] Reusing existing OSM / net.xml files.")

    passed, failed = validate_snapping(
        max_dist=args.max_dist,
        update_db=args.update_db,
    )

    benchmark_duarouter()

    # -- Update setup_real_network.py BBOX constant --------------------------
    print("\n[final] Updating BBOX in setup_real_network.py ...")
    print("-" * 60)
    _patch_setup_bbox(bbox)

    print(f"\n{SEP}")
    print("  Widening complete!")
    print(f"  New BBOX : S={bbox['south']:.4f}  W={bbox['west']:.4f}"
          f"  N={bbox['north']:.4f}  E={bbox['east']:.4f}")
    print(f"  Snap     : {len(passed)} PASS  /  {len(failed)} FAIL")
    print(f"  Net file : {NET_FILE} ({NET_FILE.stat().st_size/1e6:.1f} MB)")
    print(f"\n  Next: restart uvicorn and POST /sim/start")
    if args.update_db:
        print("  DB      : is_routable updated.")
    else:
        print("  DB note : run with --update-db to update is_routable flags.")
    print(SEP)


def _patch_setup_bbox(bbox: dict) -> None:
    """Update the BBOX dict literal in setup_real_network.py."""
    path = PROJECT_ROOT / "setup_real_network.py"
    if not path.exists():
        print("  WARN: setup_real_network.py not found — skipping patch.")
        return

    text = path.read_text(encoding="utf-8")
    old_block = (
        'BBOX = {\n'
        '    "south": 13.070,\n'
        '    "west":  80.227,\n'
        '    "north": 13.097,\n'
        '    "east":  80.253,\n'
        '}'
    )
    new_block = (
        f'BBOX = {{\n'
        f'    "south": {bbox["south"]:.4f},\n'
        f'    "west":  {bbox["west"]:.4f},\n'
        f'    "north": {bbox["north"]:.4f},\n'
        f'    "east":  {bbox["east"]:.4f},\n'
        f'}}'
    )
    if old_block in text:
        path.write_text(text.replace(old_block, new_block), encoding="utf-8")
        print(f"  [OK] BBOX updated in {path.name}")
    else:
        print(f"  WARN: Could not find old BBOX block to patch — edit {path.name} manually.")
        print(f"  New value: {new_block}")


if __name__ == "__main__":
    main()
