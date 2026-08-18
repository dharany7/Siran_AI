"""
Standalone test script to verify POST /corridor/sensor-trigger
Uploads a sample audio file and image to ensure the ML pipeline and DB logging work.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import httpx

# Force UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Ensure demo assets exist
DEMO_SIREN = Path(__file__).parent / "demo-assets" / "sirens" / "siren_1.wav"
DEMO_PLATE = Path(__file__).parent / "demo-assets" / "plates" / "plate_demo_authorized.jpg"

def main():
    print("============================================================")
    print("  Testing Corridor Sensor Trigger")
    print("============================================================\n")
    
    global DEMO_PLATE
    if not DEMO_SIREN.exists():
        print(f"❌ Missing {DEMO_SIREN} - please run setup_demo_assets.py")
        sys.exit(1)
        
    if not DEMO_PLATE.exists():
        print(f"❌ Missing {DEMO_PLATE} - please run setup_demo_assets.py")
        # Try fallback
        fallback = Path(__file__).parent / "demo-assets" / "plates" / "plate_demo.jpg"
        if fallback.exists():
            print(f"Using fallback: {fallback}")
            DEMO_PLATE = fallback
        else:
            sys.exit(1)

    url = "http://localhost:8080/corridor/sensor-trigger"
    data = {"junction_id": "252299774"}
    files = {
        "audio_file": ("siren.wav", open(DEMO_SIREN, "rb"), "audio/wav"),
        "camera_frame": ("plate.jpg", open(DEMO_PLATE, "rb"), "image/jpeg")
    }

    try:
        response = httpx.post(url, data=data, files=files, timeout=120.0)
        print(f"Status Code: {response.status_code}")
        print("Response JSON:")
        print(response.json())
        
        if response.status_code == 200:
            print("\n✅ Successfully triggered pipeline (HTTP 200).")
        else:
            print("\n❌ Pipeline returned an error.")
            
    except Exception as e:
        print(f"❌ Connection failed: {e}")

if __name__ == "__main__":
    main()
