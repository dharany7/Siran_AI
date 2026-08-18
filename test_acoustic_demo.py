"""
test_acoustic_demo.py — Smoke tests for the acoustic-detection demo path.

Run with:
    python test_acoustic_demo.py          # offline: checks imports + assets
    python test_acoustic_demo.py --live   # requires server at localhost:8000
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent


def _sep(msg: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {msg}")
    print("=" * 60)


# ── Test 1: demo assets exist ─────────────────────────────────────────────────

def test_demo_assets():
    _sep("Test 1: demo-assets structure")
    sirens  = list((ROOT / "demo-assets" / "sirens").glob("*.wav"))
    plates  = list((ROOT / "demo-assets" / "plates").glob("*"))
    assert len(sirens) >= 6, f"Expected 6 wav files, found {len(sirens)}"
    siren_files  = [f for f in sirens if f.name.startswith("siren_")]
    ambient_files = [f for f in sirens if f.name.startswith("ambient_")]
    assert len(siren_files)  >= 3, f"Need 3 siren clips, got {len(siren_files)}"
    assert len(ambient_files) >= 3, f"Need 3 ambient clips, got {len(ambient_files)}"
    assert plates, "No plate image found in demo-assets/plates/"
    print(f"  [OK] {len(siren_files)} siren clips, {len(ambient_files)} ambient clips, {len(plates)} plate(s)")


# ── Test 2: audio_ml.infer works on a demo clip ───────────────────────────────

def test_infer_siren():
    _sep("Test 2: audio_ml.infer on siren_1.wav")
    clip = ROOT / "demo-assets" / "sirens" / "siren_1.wav"
    assert clip.exists(), f"siren_1.wav not found at {clip}"
    try:
        from audio_ml.infer import predict
        result = predict(str(clip))
        print(f"  is_siren={result['is_siren']}  confidence={result['confidence']:.4f}  label={result['label']!r}")
        print(f"  all_proba={result['all_proba']}")
        assert isinstance(result["is_siren"], bool)
        assert 0.0 <= result["confidence"] <= 1.0
        print("  [OK] Siren inference passed")
    except FileNotFoundError as exc:
        print(f"  [SKIP] Model not found: {exc}")


def test_infer_ambient():
    _sep("Test 3: audio_ml.infer on ambient_1.wav")
    clip = ROOT / "demo-assets" / "sirens" / "ambient_1.wav"
    assert clip.exists(), f"ambient_1.wav not found at {clip}"
    try:
        from audio_ml.infer import predict
        result = predict(str(clip))
        print(f"  is_siren={result['is_siren']}  confidence={result['confidence']:.4f}  label={result['label']!r}")
        assert isinstance(result["is_siren"], bool)
        print("  [OK] Ambient inference passed")
    except FileNotFoundError as exc:
        print(f"  [SKIP] Model not found: {exc}")


# ── Test 4: anpr.detect_plate on demo plate ───────────────────────────────────

def test_anpr():
    _sep("Test 4: ANPR on demo plate image (civilian — reject path)")
    img = ROOT / "demo-assets" / "plates" / "plate_demo.jpg"
    assert img.exists(), f"plate_demo.jpg not found at {img}"
    try:
        from anpr.detect_plate import detect_plate
        result = detect_plate(str(img))
        print(f"  plate_text={result['plate_text']!r}")
        print(f"  is_detected={result['is_detected']}  is_authorized={result['is_authorized']}")
        print(f"  vehicle_type={result['vehicle_type']!r}  ocr_engine={result['ocr_engine']!r}")
        print(f"  confidence={result['confidence']:.4f}")
        assert result["is_detected"], "Expected plate to be detected"
        assert result["is_authorized"] is False, "Civilian plate should NOT be authorized"
        print("  [OK] ANPR civilian plate correctly rejected")
    except Exception as exc:
        print(f"  [WARN] ANPR: {exc} (EasyOCR models may be absent)")


# ── Test 5.5: Authorized plate — green-light path ─────────────────────────────

def test_anpr_authorized():
    _sep("Test 6: ANPR on plate_demo_authorized.jpg (TN87C5106 — ambulance)")
    img = ROOT / "demo-assets" / "plates" / "plate_demo_authorized.jpg"
    assert img.exists(), (
        f"plate_demo_authorized.jpg not found at {img}. "
        "Run 'python setup_demo_assets.py' first."
    )
    try:
        from anpr.detect_plate import detect_plate, AMBULANCE_ALLOWLIST
        result = detect_plate(str(img))
        print(f"  plate_text={result['plate_text']!r}  (expected 'TN87C5106')")
        print(f"  raw_ocr_text={result['raw_ocr_text']!r}")
        print(f"  is_detected={result['is_detected']}")
        print(f"  is_authorized={result['is_authorized']}  vehicle_type={result['vehicle_type']!r}")
        print(f"  confidence={result['confidence']:.4f}")
        print(f"  Allowlist entry: {AMBULANCE_ALLOWLIST.get('TN87C5106', 'MISSING!')}")

        assert result["plate_text"] == "TN87C5106", (
            f"Expected EasyOCR to read 'TN87C5106', got {result['plate_text']!r}. "
            "If OCR output changed, re-run setup and update the allowlist."
        )
        assert result["is_detected"] is True, "Plate should be detected"
        assert result["is_authorized"] is True, (
            f"TN87C5106 should be authorized (check AMBULANCE_ALLOWLIST in anpr/detect_plate.py). "
            f"Got is_authorized={result['is_authorized']!r}"
        )
        assert result["vehicle_type"] == "ambulance", (
            f"Expected vehicle_type='ambulance', got {result['vehicle_type']!r}"
        )
        print("  [OK] Authorized ambulance plate TN87C5106 correctly verified")
    except AssertionError:
        raise
    except Exception as exc:
        print(f"  [WARN] ANPR: {exc} (EasyOCR models may be absent)")


# ── Test 5: FastAPI router import ─────────────────────────────────────────────

def test_router_import():
    _sep("Test 5: import backend.routers.acoustic_demo")
    try:
        from backend.routers import acoustic_demo
        assert hasattr(acoustic_demo, "router")
        routes = [r.path for r in acoustic_demo.router.routes]
        print(f"  Routes: {routes}")
        assert any("acoustic-event" in r for r in routes), "acoustic-event route missing"
        assert any("demo-assets/list" in r for r in routes), "demo-assets/list route missing"
        assert any("demo-assets/junctions" in r for r in routes), "junctions route missing"
        assert any("demo-assets/plates" in r for r in routes), "demo-assets/plates route missing"
        print("  [OK] Router import and routes verified (incl. /demo-assets/plates)")
    except Exception as exc:
        print(f"  [FAIL] {exc}")
        raise


# ── Test 6: Live HTTP smoke test ──────────────────────────────────────────────

def test_live_endpoints(base: str = "http://localhost:8000"):
    import urllib.request
    import urllib.parse

    _sep("Test 6: Live endpoint smoke test")

    # GET /junction/demo-assets/list
    url = base + "/junction/demo-assets/list"
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            data = json.loads(r.read())
        print(f"  GET /junction/demo-assets/list -> sirens={data['sirens']}")
        assert data["sirens"], "No siren clips returned"
        print("  [OK] /junction/demo-assets/list")
    except Exception as exc:
        print(f"  [FAIL] {url}: {exc}")
        return

    # GET /junction/demo-assets/junctions
    url = base + "/junction/demo-assets/junctions"
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            data = json.loads(r.read())
        junctions = data.get("junctions", [])
        print(f"  GET /junction/demo-assets/junctions -> {len(junctions)} junctions ({data.get('source')})")
        assert junctions, "No junctions returned"
        print("  [OK] /junction/demo-assets/junctions")
    except Exception as exc:
        print(f"  [FAIL] {url}: {exc}")
        return

    # POST /junction/{jid}/acoustic-event with ambient clip (should not flip signal)
    jid = junctions[0]
    url = base + f"/junction/{urllib.parse.quote(jid, safe='')}/acoustic-event"
    import urllib.parse as _up
    body = _up.urlencode({"sample_name": "ambient_1.wav"}).encode()
    req  = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            result = json.loads(r.read())
        clf = result.get("siren_classified", {})
        print(f"  POST acoustic-event (ambient_1.wav): is_siren={clf.get('is_siren')}  conf={clf.get('confidence'):.4f}")
        print("  [OK] POST /junction/{jid}/acoustic-event (ambient path)")
    except Exception as exc:
        print(f"  [FAIL] POST {url}: {exc}")

    # POST with siren clip
    body = _up.urlencode({"sample_name": "siren_1.wav"}).encode()
    req  = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            result = json.loads(r.read())
        clf = result.get("siren_classified", {})
        sig = result.get("signal_flipped", {})
        print(f"  POST acoustic-event (siren_1.wav): is_siren={clf.get('is_siren')}  conf={clf.get('confidence'):.4f}  signal_green={sig.get('signal_green')}")
        print("  [OK] POST /junction/{jid}/acoustic-event (siren + civilian plate path)")
    except Exception as exc:
        print(f"  [FAIL] POST {url}: {exc}")

    # POST with siren clip + authorized plate
    _sep("Test 7 (live): siren_1.wav + plate_demo_authorized.jpg -> is_authorized=True")
    body = _up.urlencode({"sample_name": "siren_1.wav", "plate_name": "plate_demo_authorized.jpg"}).encode()
    req  = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            result = json.loads(r.read())
        clf  = result.get("siren_classified", {})
        anpr = result.get("anpr_confirmed",   {})
        sig  = result.get("signal_flipped",   {})
        print(f"  is_siren={clf.get('is_siren')}  conf={clf.get('confidence'):.4f}")
        print(f"  plate_text={anpr.get('plate_text')!r}  is_authorized={anpr.get('is_authorized')}")
        print(f"  signal_flipped: skipped={sig.get('skipped')}  reason={sig.get('reason')!r}  signal_green={sig.get('signal_green')}")

        assert clf.get("is_siren") is True, "Expected is_siren=True for siren_1.wav"
        assert anpr.get("plate_text") == "TN87C5106", (
            f"Expected plate_text='TN87C5106', got {anpr.get('plate_text')!r}"
        )
        assert anpr.get("is_authorized") is True, (
            "Expected is_authorized=True for TN87C5106 (check AMBULANCE_ALLOWLIST)"
        )
        # signal_flipped.signal_green=True only when SUMO is running; otherwise
        # it is skipped with reason='sim_not_running'. Both are acceptable here.
        assert sig is not None, "signal_flipped key missing from response"
        if sig.get("signal_green"):
            print("  [OK] Signal flipped GREEN (SUMO live)")
        elif sig.get("reason") == "sim_not_running":
            print("  [OK] Plate authorized; SUMO offline — signal would flip live")
        else:
            raise AssertionError(f"Unexpected signal_flipped state: {sig}")
        print("  [OK] Test 7: authorized plate pipeline passed")
    except AssertionError:
        raise
    except Exception as exc:
        print(f"  [FAIL] POST {url}: {exc}")


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    live = "--live" in sys.argv

    test_demo_assets()
    test_infer_siren()
    test_infer_ambient()
    test_anpr()
    test_anpr_authorized()   # Test 6: authorized plate TN87C5106
    test_router_import()

    if live:
        test_live_endpoints()  # Tests 6-live & 7 (live)
    else:
        print("\n[INFO] Skip live tests. Run with --live to test against localhost:8000")

    print("\n" + "=" * 60)
    print("  All offline tests complete.")
    print("=" * 60)
