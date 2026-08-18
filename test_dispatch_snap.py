"""
Standalone test script to verify snapping and routing logic for:
- Hospital ID 1 (Kilpauk Medical College)
- Hospital ID 27 (Dr. Agarwal's Eye Hospital)

Ensures that nearest_edge() properly finds a routable edge within 500m
and duarouter computes a valid path without 422 errors.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Force UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

from agents.sumo_env import SumoEnv
from agents.grid_controller import route_via_duarouter

# Test Coordinates (approximate start position for ambulance)
START_LAT = 13.0843
START_LNG = 80.2371

HOSPITALS = [
    {"id": 1, "name": "Kilpauk Medical College", "lat": 13.0785984, "lng": 80.2433369},
    {"id": 27, "name": "Dr. Agarwal's Eye Hospital", "lat": 13.074266, "lng": 80.2382156}
]

def main():
    print("============================================================")
    print("  Testing Dispatch Snapping and Routing")
    print("============================================================\n")
    
    env = SumoEnv()
    start_edge = env.nearest_edge(START_LAT, START_LNG, radius=500.0)
    
    if not start_edge:
        print(f"❌ Failed to snap START position ({START_LAT}, {START_LNG}) to an edge.")
        sys.exit(1)
        
    print(f"✅ Snapped START position to edge: '{start_edge}'")
    
    success_count = 0
    for hosp in HOSPITALS:
        print(f"\n--- Testing Hospital ID {hosp['id']}: {hosp['name']} ---")
        dest_edge = env.nearest_edge(hosp["lat"], hosp["lng"], radius=500.0)
        
        if not dest_edge:
            print(f"❌ Failed to snap {hosp['name']} to an edge within 500m.")
            continue
            
        print(f"✅ Snapped {hosp['name']} to edge: '{dest_edge}'")
        
        try:
            route = route_via_duarouter(start_edge, dest_edge)
            print(f"✅ duarouter computed a route with {len(route)} edges.")
            success_count += 1
        except Exception as e:
            print(f"❌ duarouter routing failed: {e}")
            
    print("\n============================================================")
    if success_count == len(HOSPITALS):
        print("  All tests PASSED! No 422/404 routing errors expected.")
    else:
        print(f"  {len(HOSPITALS) - success_count} tests FAILED.")
    print("============================================================")

if __name__ == "__main__":
    main()
