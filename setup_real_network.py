"""
setup_real_network.py
=====================
One-shot script that builds the Kilpauk, Chennai real road network for Siren AI.

Steps
-----
1. Download kilpauk.osm from the Overpass API (bounding box ~3 km x 3 km)
2. Run netconvert to produce kilpauk.net.xml (with real traffic signals)
3. Query Overpass for amenity=hospital in the same bbox -> seed hospitals table
4. Generate synthetic background traffic via SUMO randomTrips.py
5. Write kilpauk.sumocfg

Run once:
    python setup_real_network.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# Kilpauk, Chennai — compact medical district with multiple real hospitals
BBOX = {
    "south": 13.070,
    "west":  80.227,
    "north": 13.097,
    "east":  80.253,
}

PROJECT_ROOT = Path(__file__).parent
SUMO_NET_DIR = PROJECT_ROOT / "sumo_net"
SUMO_HOME    = Path(os.environ.get("SUMO_HOME",
                    r"C:\Program Files (x86)\Eclipse\Sumo"))
NETCONVERT   = SUMO_HOME / "bin" / "netconvert.exe"
DUAROUTER    = SUMO_HOME / "bin" / "duarouter.exe"
RANDOM_TRIPS = SUMO_HOME / "tools" / "randomTrips.py"

OSM_FILE  = SUMO_NET_DIR / "kilpauk.osm"
NET_FILE  = SUMO_NET_DIR / "kilpauk.net.xml"
ROU_FILE  = SUMO_NET_DIR / "kilpauk_bg.rou.xml"
CFG_FILE  = SUMO_NET_DIR / "kilpauk.sumocfg"

SUMO_NET_DIR.mkdir(exist_ok=True)

SEP = "=" * 60


def step(n, msg):
    print(f"\n[{n}/5] {msg}")


# ---------------------------------------------------------------------------
# 1. Download OSM
# ---------------------------------------------------------------------------
step(1, "Downloading Kilpauk OSM data from Overpass API ...")

overpass_url = "https://overpass-api.de/api/interpreter"
# [out:xml] fetches all ways, nodes, relations in bbox
overpass_query = (
    f"[out:xml][timeout:120];"
    f"(way[\"highway\"]({BBOX['south']},{BBOX['west']},{BBOX['north']},{BBOX['east']});"
    f"node[\"highway\"]({BBOX['south']},{BBOX['west']},{BBOX['north']},{BBOX['east']});"
    f"relation[\"highway\"]({BBOX['south']},{BBOX['west']},{BBOX['north']},{BBOX['east']}););"
    f"out body;>;out skel qt;"
)

HEADERS = {
    "User-Agent": "SirenAI/1.0 (emergency-vehicle routing research; contact@example.com)",
    "Accept": "application/xml, text/xml, */*",
}

# Retry up to 3 times (Overpass can be slow)
for attempt in range(1, 4):
    try:
        print(f"  Attempt {attempt}/3 ...")
        r = requests.get(
            overpass_url,
            params={"data": overpass_query},
            headers=HEADERS,
            timeout=180,
        )
        r.raise_for_status()
        osm_bytes = r.content
        print(f"  Downloaded {len(osm_bytes)/1e6:.1f} MB")
        break
    except Exception as exc:
        print(f"  WARN: Overpass attempt {attempt} failed: {exc}")
        if attempt < 3:
            # Try alternative Overpass instance
            overpass_url = "https://overpass.kumi.systems/api/interpreter"
            time.sleep(5)
        else:
            # Last resort: OSM export API (bbox download)
            print("  Falling back to export.openstreetmap.org ...")
            osm_export = (
                f"https://overpass-api.de/api/map?"
                f"bbox={BBOX['west']},{BBOX['south']},{BBOX['east']},{BBOX['north']}"
            )
            try:
                r2 = requests.get(osm_export, headers=HEADERS, timeout=180)
                r2.raise_for_status()
                osm_bytes = r2.content
                print(f"  Fallback download: {len(osm_bytes)/1e6:.1f} MB")
                break
            except Exception as exc2:
                print(f"  ERROR: All download attempts failed: {exc2}")
                sys.exit(1)

OSM_FILE.write_bytes(osm_bytes)
print(f"  Saved -> {OSM_FILE}")

# ---------------------------------------------------------------------------
# 2. Run netconvert
# ---------------------------------------------------------------------------
step(2, "Converting OSM -> SUMO network (netconvert) ...")

netconvert_cmd = [
    str(NETCONVERT),
    "--osm-files", str(OSM_FILE),
    "-o", str(NET_FILE),
    # Traffic signals: import OSM-tagged signals as TLS junctions
    "--tls.discard-simple",       # keep only real signalised junctions
    "--tls.join",                 # merge nearby TLS nodes
    "--osm.bike-access",          # include bike lanes (realistic road widths)
    "--geometry.remove",          # simplify intermediate geometry nodes
    "--roundabouts.guess",        # detect roundabouts automatically
    "--junctions.join",           # merge close junctions
    "--no-internal-links",        # simpler routing
    "--keep-edges.by-vclass", "passenger",  # keep only driveable roads
    "--no-warnings",
]

print("  Running:", " ".join(netconvert_cmd[:4]), "...")
result = subprocess.run(netconvert_cmd, capture_output=True, text=True)
if result.returncode != 0:
    print("  netconvert STDERR:", result.stderr[-2000:])
    print("  ERROR: netconvert failed. Exiting.")
    sys.exit(1)

# Count junctions and edges from the XML
net_xml = NET_FILE.read_text(encoding="utf-8", errors="ignore")
junc_count = net_xml.count("<junction ")
edge_count = net_xml.count('<edge id="')
tls_count  = net_xml.count("<tlLogic ")
print(f"  Network: {junc_count} junctions, {edge_count} edges, {tls_count} TLS junctions")
print(f"  Saved -> {NET_FILE}  ({NET_FILE.stat().st_size/1e3:.0f} KB)")

# ---------------------------------------------------------------------------
# 3. Query hospitals from Overpass + seed DB
# ---------------------------------------------------------------------------
step(3, "Querying hospitals from Overpass API ...")

hosp_query = (
    f"[out:json][timeout:60];"
    f"(node[\"amenity\"=\"hospital\"]({BBOX['south']},{BBOX['west']},{BBOX['north']},{BBOX['east']});"
    f"way[\"amenity\"=\"hospital\"]({BBOX['south']},{BBOX['west']},{BBOX['north']},{BBOX['east']}););"
    f"out center;"
)

try:
    hr = requests.get(
        "https://overpass-api.de/api/interpreter",
        params={"data": hosp_query},
        headers=HEADERS,
        timeout=90,
    )
    hr.raise_for_status()
    hosp_data = hr.json()
except Exception as exc:
    print(f"  WARN: Hospital query failed: {exc}")
    hosp_data = {"elements": []}

hospitals = []
for el in hosp_data.get("elements", []):
    tags = el.get("tags", {})
    name = tags.get("name") or tags.get("name:en") or "Unknown Hospital"
    if el["type"] == "node":
        lat, lng = el["lat"], el["lon"]
    elif el["type"] == "way":
        center = el.get("center", {})
        lat = center.get("lat", 0)
        lng = center.get("lon", 0)
    else:
        continue
    if lat and lng:
        hospitals.append({"name": name, "lat": lat, "lng": lng,
                          "address": tags.get("addr:full", "")})

print(f"  Found {len(hospitals)} hospitals")
for h in hospitals:
    print(f"    - {h['name']}  ({h['lat']:.5f}, {h['lng']:.5f})")

# Seed into PostgreSQL via SQLAlchemy
print("  Seeding hospitals table ...")
try:
    # Bootstrap app environment
    sys.path.insert(0, str(PROJECT_ROOT))
    from backend.database import SessionLocal, init_db
    from backend.models import Hospital

    init_db()
    db = SessionLocal()
    added = 0
    for h in hospitals:
        existing = db.query(Hospital).filter(Hospital.name == h["name"]).first()
        if not existing:
            db.add(Hospital(name=h["name"], lat=h["lat"], lng=h["lng"],
                            address=h["address"]))
            added += 1
    db.commit()
    db.close()
    print(f"  Seeded {added} new hospitals (skipped {len(hospitals)-added} duplicates)")
except Exception as exc:
    print(f"  WARN: DB seed failed: {exc}")
    print("  (Continue anyway — hospitals can be seeded manually)")

# Save hospital list as JSON for reference
hosp_json = SUMO_NET_DIR / "hospitals.json"
hosp_json.write_text(json.dumps(hospitals, indent=2, ensure_ascii=False))
print(f"  Saved hospital list -> {hosp_json}")

# ---------------------------------------------------------------------------
# 4. Generate background traffic
# ---------------------------------------------------------------------------
step(4, "Generating background traffic via randomTrips.py ...")

# randomTrips.py --net-file NET -o trips.xml -r routes.rou.xml -e END -p PERIOD
trips_file = SUMO_NET_DIR / "kilpauk_trips.xml"
rt_cmd = [
    sys.executable,
    str(RANDOM_TRIPS),
    "--net-file", str(NET_FILE),
    "-o", str(trips_file),
    "-r", str(ROU_FILE),
    "--end", "3600",
    "--period", "8",        # one vehicle every 8 s = ~450 vehicles/hour
    "--validate",           # use duarouter to ensure routes are valid
    "--remove-loops",
    "--seed", "42",
]

rt_result = subprocess.run(rt_cmd, capture_output=True, text=True, cwd=str(SUMO_NET_DIR))
if rt_result.returncode != 0:
    print("  WARN: randomTrips.py errors:", rt_result.stderr[-500:])
    # Write a minimal fallback empty route file
    ROU_FILE.write_text(
        '<?xml version="1.0"?>\n<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"'
        ' xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">\n</routes>\n'
    )
    print("  Fallback: empty route file written (no background traffic)")
else:
    rou_lines = ROU_FILE.read_text().count("<vehicle")
    print(f"  Generated ~{rou_lines} background vehicles -> {ROU_FILE}")

# ---------------------------------------------------------------------------
# 5. Write kilpauk.sumocfg
# ---------------------------------------------------------------------------
step(5, "Writing kilpauk.sumocfg ...")

sumocfg = f"""<?xml version="1.0" encoding="UTF-8"?>
<!--
  kilpauk.sumocfg - SUMO config for Siren AI real-world network
  Area: Kilpauk, Chennai, India
  OSM bbox: S={BBOX['south']} W={BBOX['west']} N={BBOX['north']} E={BBOX['east']}
  Generated by setup_real_network.py
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

CFG_FILE.write_text(sumocfg)
print(f"  Saved -> {CFG_FILE}")

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print("  Setup complete!")
print(f"  Network  : {NET_FILE.name}  ({junc_count} junctions, {tls_count} TLS)")
print(f"  Hospitals: {len(hospitals)} seeded")
print(f"  Config   : {CFG_FILE.name}")
print(f"\n  Next: restart uvicorn and POST /sim/start")
print(SEP)
