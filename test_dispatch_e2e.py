"""
test_dispatch_e2e.py — End-to-end test for POST /dispatch multi-junction routing.

Starts the FastAPI server in a background thread, runs the full test
sequence, and prints a clear PASS/FAIL report.

Usage:
    python test_dispatch_e2e.py
"""
from __future__ import annotations

import sys
import time
import threading
import json

import requests
import uvicorn

BASE_URL = "http://127.0.0.1:8899"
START_TIMEOUT  = 15   # seconds to wait for server to be ready
DISPATCH_TIMEOUT = 300  # seconds for the dispatch call (SUMO steps take time)


def start_server():
    """Run the FastAPI app in a background thread."""
    import os
    os.environ.setdefault("SUMO_HOME", r"C:\Program Files (x86)\Eclipse\Sumo")
    from backend.main import app
    config = uvicorn.Config(app, host="127.0.0.1", port=8899, log_level="warning")
    server = uvicorn.Server(config)
    server.run()


def wait_for_server(url: str, timeout: int = 15) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{url}/health", timeout=2)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def main():
    sep = "=" * 65
    print(f"\n{sep}")
    print("  Siren AI - POST /dispatch end-to-end test")
    print("  Network: sumo_net/cross.net.xml (3-TLS linear chain)")
    print("  Route  : 1 -> J1 -> J2 -> J3 -> 3")
    print(f"{sep}\n")

    # -- Start server -----------------------------------------------
    print("[1/5] Starting FastAPI server on port 8899 ...")
    t = threading.Thread(target=start_server, daemon=True)
    t.start()

    if not wait_for_server(BASE_URL, timeout=START_TIMEOUT):
        print("FAIL: Server did not become ready in time.")
        sys.exit(1)
    print("      Server is up.\n")

    # ── Start simulation ───────────────────────────────────────────
    print("[2/5] POST /sim/start  (launch SUMO + open TraCI) …")
    r = requests.post(f"{BASE_URL}/sim/start", timeout=30)
    body = r.json()
    print("      status=%d  success=%s  message=%r" % (r.status_code, body.get('success'), body.get('message')))
    if not body.get("success"):
        print(f"FAIL: Could not start simulation: {body}")
        sys.exit(1)
    print()

    # ── Wait a moment for SUMO to initialise ──────────────────────
    time.sleep(2)

    # ── Check sim status ───────────────────────────────────────────
    print("[3/5] GET /sim/status  (confirm TraCI connected) …")
    r = requests.get(f"{BASE_URL}/sim/status", timeout=10)
    status = r.json()
    print(f"      connected={status.get('connected')}  step={status.get('step')}  "
          f"tls_state={status.get('tls_state')!r}  vehicles={status.get('vehicle_count')}")
    if not status.get("connected"):
        print("FAIL: Simulation did not connect.")
        sys.exit(1)
    print()

    # ── Dispatch ambulance ─────────────────────────────────────────
    print("[4/5] POST /dispatch  {\"start\":\"1\",\"destination\":\"3\"} …")
    print("      (this runs the SUMO sim until the ambulance exits — may take ~20–60s)")
    t0 = time.time()
    r = requests.post(
        f"{BASE_URL}/dispatch",
        json={"start": "1", "destination": "3"},
        timeout=DISPATCH_TIMEOUT,
    )
    elapsed = time.time() - t0
    print(f"      HTTP {r.status_code}  ({elapsed:.1f}s)\n")

    if r.status_code != 200:
        print(f"FAIL: dispatch returned HTTP {r.status_code}")
        print(r.text[:2000])
        sys.exit(1)

    resp = r.json()

    # ── Evaluate results ───────────────────────────────────────────
    print("[5/5] Validating response …\n")

    route          = resp.get("route", [])
    completed      = resp.get("completed", False)
    tls_schedule   = resp.get("tls_schedule", [])
    steps_taken    = resp.get("steps_taken", 0)
    reasoning      = resp.get("navigator_reasoning", "")
    timing_log     = resp.get("timing_log", [])

    # Print full response (pretty)
    print("  ── Dispatch response ──────────────────────────────────────")
    print(f"  event_id    : {resp.get('event_id')}")
    print(f"  route       : {route}")
    print(f"  completed   : {completed}")
    print(f"  steps_taken : {steps_taken}")
    print(f"  model_used  : {resp.get('model_used')}")
    print(f"\n  tls_schedule ({len(tls_schedule)} entries):")
    for entry in tls_schedule:
        print(f"    junction={entry['junction_id']:<4s}  "
              f"distance={entry['distance_m']:>7.1f} m  "
              f"ETA={entry['eta_seconds']:>6.1f} s  "
              f"green_at_step={entry['green_at_step']}")
    print(f"\n  navigator_reasoning (first 300 chars):")
    print(f"    {reasoning[:300]!r}")
    print(f"\n  timing_log ({len(timing_log)} entries):")
    for line in timing_log[:15]:
        print(f"    {line}")
    if len(timing_log) > 15:
        print(f"    … ({len(timing_log) - 15} more entries)")
    print()

    # ── Assertions ─────────────────────────────────────────────────
    failures = []

    if not completed:
        failures.append("completed is False — ambulance did not reach destination")

    if "1" not in route or "3" not in route:
        failures.append(f"route does not contain both '1' and '3': {route}")

    tls_junctions = [e["junction_id"] for e in tls_schedule]
    if len(set(tls_junctions)) < 3:
        failures.append(
            f"tls_schedule has fewer than 3 distinct junctions: {tls_junctions}"
        )

    expected_junctions = {"J1", "J2", "J3"}
    missing = expected_junctions - set(tls_junctions)
    if missing:
        failures.append(f"tls_schedule missing expected junctions: {missing}")

    if len(route) < 5:
        failures.append(
            f"route too short — expected at least 5 junctions [1,J1,J2,J3,3], got: {route}"
        )

    print(sep)
    if failures:
        print("  RESULT: FAIL")
        for f in failures:
            print(f"    ✗ {f}")
    else:
        print("  RESULT: PASS ✓")
        print(f"    ✓ completed=True")
        print(f"    ✓ route passes through {len(route)} junctions: {route}")
        print(f"    ✓ tls_schedule has {len(tls_schedule)} distinct TLS junctions activated:")
        for entry in tls_schedule:
            print(f"        {entry['junction_id']} @ step {entry['green_at_step']}  "
                  f"(ETA {entry['eta_seconds']}s,  dist {entry['distance_m']}m)")
    print(sep)

    # cleanup
    try:
        requests.post(f"{BASE_URL}/sim/stop", timeout=10)
    except Exception:
        pass

    sys.exit(0 if not failures else 1)


if __name__ == "__main__":
    main()
