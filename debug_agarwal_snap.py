"""
debug_agarwal_snap.py
=====================
Investigates why Dr. Agarwal's Eye Hospital (13.074266, 80.2382156) fails to
snap in the live dispatch path, even though it passes the offline 150m check.

Steps:
  1. Load kilpauk.net.xml with sumolib and query all edges within 200m
     (with AND without vclass filter) — print id, type, distance, allow/disallow.
  2. Inspect the netconvert flags used, especially vclass and type filters.
  3. Re-run netconvert with relaxed filters, re-test snap for Agarwal + full list.
  4. Report findings: if no edge within 200m even in raw net, mark as excluded.
  5. Show what radius convertRoad actually uses and propose the fix.
"""
import sys, os, math, subprocess, time, xml.etree.ElementTree as ET
from pathlib import Path

# Force UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Target hospital ────────────────────────────────────────────────────────────
HOSP_NAME = "Dr. Agarwal's Eye Hospital"
HOSP_LAT  = 13.074266
HOSP_LNG  = 80.2382156
SEARCH_RADIUS = 300   # metres — generous

PROJECT   = Path(__file__).parent
NET_FILE  = PROJECT / "sumo_net" / "kilpauk.net.xml"
SUMO_HOME = Path(os.environ.get("SUMO_HOME", r"C:\Program Files (x86)\Eclipse\Sumo"))
NETCONVERT = SUMO_HOME / "bin" / "netconvert.exe"
OSM_FILE   = PROJECT / "sumo_net" / "kilpauk.osm"

SEP = "=" * 70
print(SEP)
print("  Debug: Dr. Agarwal hospital snap analysis")
print(SEP)

# ── 1. sumolib neighbour search ───────────────────────────────────────────────
print("\n[1/4] sumolib edge search within", SEARCH_RADIUS, "m ...")
print("-" * 60)

# Try to add sumolib to path
sumolib_path = SUMO_HOME / "tools"
if str(sumolib_path) not in sys.path:
    sys.path.insert(0, str(sumolib_path))

try:
    import sumolib
    print(f"  sumolib imported from: {sumolib.__file__}")
    net = sumolib.net.readNet(str(NET_FILE), withInternal=False)
    print(f"  Network loaded: {len(net.getEdges())} edges")

    # Convert lat/lng to SUMO XY
    x, y = net.convertLonLat2XY(HOSP_LNG, HOSP_LAT)
    print(f"\n  Hospital SUMO XY: ({x:.2f}, {y:.2f})")

    # getNeighboringEdges returns [(edge, distance), ...] sorted by distance
    neighbours = net.getNeighboringEdges(x, y, r=SEARCH_RADIUS, includeJunctions=False)
    neighbours_sorted = sorted(neighbours, key=lambda t: t[1])

    print(f"\n  Edges within {SEARCH_RADIUS}m  ({len(neighbours_sorted)} found):")
    print(f"  {'EdgeID':<35}  {'Type':<20}  {'Dist(m)':>8}  {'Allow'}")
    print("  " + "-" * 82)

    for edge, dist in neighbours_sorted[:30]:  # show up to 30
        allow    = edge.getParam("allow", "")
        disallow = edge.getParam("disallow", "")
        etype    = edge.getType() or ""
        print(f"  {edge.getID():<35}  {etype:<20}  {dist:>8.1f}  allow={allow!r}  disallow={disallow!r}")

    if not neighbours_sorted:
        print("  *** NO edges found within", SEARCH_RADIUS, "m ***")
        print("  This hospital's location may have no digitised road in OSM.")

    # Also check nearest edge regardless of radius
    if neighbours_sorted:
        nearest_edge, nearest_dist = neighbours_sorted[0]
        print(f"\n  Nearest edge: {nearest_edge.getID()!r}  dist={nearest_dist:.1f}m")
        print(f"  Edge type   : {nearest_edge.getType()!r}")
        print(f"  Speed limit : {nearest_edge.getSpeed():.1f} m/s")
        print(f"  Lanes       : {nearest_edge.getLaneNumber()}")

except ImportError as e:
    print(f"  WARN: sumolib not available ({e})")
    print(f"  Add {sumolib_path} to PYTHONPATH or check SUMO installation.")
    sumolib = None

# ── 2. Compare to nearby passing hospitals ────────────────────────────────────
print("\n[2/4] Comparing to nearby hospitals that pass snapping ...")
print("-" * 60)

NEARBY = [
    ("Dr.Manju Kulkarni Hospital",   13.0743836, 80.2367672),
    ("Zubeda Hospitals Ltd.",         13.0741955, 80.2357775),
    ("Josephs Nursing Home",          13.0739408, 80.2357346),
    ("Sundaram Cancer Centre",        13.07604,   80.2366224),
]

if sumolib is not None:
    for name, lat, lng in NEARBY:
        hx, hy = net.convertLonLat2XY(lng, lat)
        nb = net.getNeighboringEdges(hx, hy, r=200, includeJunctions=False)
        if nb:
            best_edge, best_dist = sorted(nb, key=lambda t: t[1])[0]
            print(f"  {name:<48}  nearest={best_dist:>6.1f}m  edge={best_edge.getID()!r}")
        else:
            print(f"  {name:<48}  *** NO edge within 200m ***")

    # Distance from Agarwal to the nearest-passing-hospital edge
    print()
    if neighbours_sorted:
        ne, nd = neighbours_sorted[0]
        print(f"  Agarwal nearest edge: {ne.getID()!r} at {nd:.1f}m")
        # What does convertRoad return for this XY?
        print(f"  (convertRoad uses default radius; sumolib shows real nearest edge)")

# ── 3. Inspect the current netconvert command ─────────────────────────────────
print("\n[3/4] Checking netconvert config (vclass / type filters) ...")
print("-" * 60)

CURRENT_NETCONVERT_FLAGS = [
    "--tls.discard-simple",
    "--tls.join",
    "--osm.bike-access",
    "--geometry.remove",
    "--roundabouts.guess",
    "--junctions.join",
    "--no-internal-links",
    "--keep-edges.by-vclass passenger",   # <-- KEY FILTER
    "--no-warnings",
]
print("  Current flags used in setup_real_network.py / widen_network.py:")
for f in CURRENT_NETCONVERT_FLAGS:
    print(f"    {f}")

print()
print("  '--keep-edges.by-vclass passenger' keeps only edges that allow")
print("  'passenger' vehicles.  Access roads tagged highway=service,")
print("  highway=living_street, or highway=residential in OSM still")
print("  allow passenger vehicles unless explicitly tagged otherwise.")
print()
print("  Checking OSM tags near hospital ...")

# Grep the OSM file for nodes near the hospital
try:
    osm_text = OSM_FILE.read_text(encoding="utf-8", errors="ignore")
    # Very rough: find all way IDs that mention Agarwal or nearby coords
    # Better: look for nodes within a small coordinate window
    import re
    lat_lo, lat_hi = 13.0738, 13.0748
    lng_lo, lng_hi = 80.2377, 80.2390

    # Find node IDs in the coordinate window
    node_ids_in_box = set()
    for m in re.finditer(r'<node id="(\d+)" [^>]*lat="([\d.]+)" lon="([\d.]+)"', osm_text):
        nid, nlat, nlon = m.group(1), float(m.group(2)), float(m.group(3))
        if lat_lo <= nlat <= lat_hi and lng_lo <= nlon <= lng_hi:
            node_ids_in_box.add(nid)

    print(f"  OSM nodes in lat=[{lat_lo},{lat_hi}] lon=[{lng_lo},{lng_hi}]: {len(node_ids_in_box)}")

    # Find ways containing those nodes
    ways_found = []
    for m in re.finditer(r'<way id="(\d+)"[^>]*>(.*?)</way>', osm_text, re.DOTALL):
        wid, wbody = m.group(1), m.group(2)
        refs = set(re.findall(r'<nd ref="(\d+)"', wbody))
        if refs & node_ids_in_box:
            highway_m = re.search(r'k="highway" v="([^"]+)"', wbody)
            oneway_m  = re.search(r'k="oneway" v="([^"]+)"', wbody)
            access_m  = re.search(r'k="access" v="([^"]+)"', wbody)
            motor_m   = re.search(r'k="motor_vehicle" v="([^"]+)"', wbody)
            name_m    = re.search(r'k="name" v="([^"]+)"', wbody)
            ways_found.append({
                "id":      wid,
                "highway": highway_m.group(1) if highway_m else "n/a",
                "name":    name_m.group(1)    if name_m    else "",
                "oneway":  oneway_m.group(1)  if oneway_m  else "no",
                "access":  access_m.group(1)  if access_m  else "",
                "motor":   motor_m.group(1)   if motor_m   else "",
            })

    if ways_found:
        print(f"  Ways through/near hospital area ({len(ways_found)}):")
        for w in ways_found:
            print(f"    way/{w['id']}  highway={w['highway']!r}  name={w['name']!r}"
                  f"  oneway={w['oneway']}  access={w['access']!r}  motor_vehicle={w['motor']!r}")
    else:
        print("  No OSM ways found in the coordinate window.")
        print("  This hospital may lack a digitised adjacent road in OSM.")
except Exception as e:
    print(f"  OSM parse error: {e}")

# ── 4. Snap distance summary and recommendation ───────────────────────────────
print("\n[4/4] Summary and recommendations ...")
print("-" * 60)

if sumolib is not None and neighbours_sorted:
    nd = neighbours_sorted[0][1]
    ne = neighbours_sorted[0][0]
    print(f"  Nearest edge in kilpauk.net.xml : {ne.getID()!r} at {nd:.1f}m")

    if nd <= 150:
        print(f"  CONCLUSION: Edge IS within 150m — offline validator shows PASS.")
        print(f"  The live TraCI convertRoad() call should also snap correctly.")
        print(f"  Root cause of dispatch 422s is likely NOT this hospital's coords")
        print(f"  but the duarouter 'type=\"ambulance\"' trip with no vtype defined.")
    elif nd <= 300:
        print(f"  CONCLUSION: Edge is {nd:.0f}m away — beyond 150m threshold.")
        print(f"  The road near this hospital may be tagged as a non-driveable type.")
        print(f"  Recommend: raise MAX_DIST in validate_hospitals to 200m, or")
        print(f"  rebuild with relaxed --keep-edges.by-vclass filter.")
    else:
        print(f"  CONCLUSION: No edge within 300m — hospital is outside road network.")
        print(f"  This is a genuine OSM data gap (no digitised road nearby).")
        print(f"  Mark as known-excluded; it cannot snap without OSM data improvement.")
else:
    print("  Could not run sumolib analysis — check SUMO installation.")

print()
print("  Also note: dispatch 422 errors come from TWO distinct causes:")
print("  A) snap failure (convertRoad returns empty / internal edge)")
print("  B) duarouter 'ambulance' vtype not defined -> no route produced")
print("  -> Fix B: remove type=\"ambulance\" from the duarouter trip XML")
print("     OR add a vtype definition with vClass=\"emergency\" to the trip.")
print()
print(SEP)
