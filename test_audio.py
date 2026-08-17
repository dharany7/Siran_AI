"""
test_audio.py — End-to-end test for POST /audio/detect.

Generates three synthetic WAV files:
  1. siren_test.wav  — two-tone siren sweep (expect is_siren=True)
  2. traffic_test.wav — broadband noise (expect is_siren=False)
  3. tone_test.wav   — stationary 440 Hz tone (expect is_siren=False)

Uploads each to POST /audio/detect and prints the full JSON response.
Also starts the simulation first so auto-dispatch can be observed.
"""
import json
import urllib.request
import urllib.error
import tempfile
from pathlib import Path

import numpy as np
from scipy.io import wavfile

# ── Audio generation helpers ─────────────────────────────────────────────────

SR = 22_050   # Hz
DUR = 3.0     # seconds

def make_siren(seed=0):
    rng = np.random.default_rng(seed)
    t   = np.linspace(0, DUR, int(SR * DUR), endpoint=False)
    lo, hi = 760, 1050
    sweep = np.where(np.sin(2 * np.pi * 1.0 * t) > 0, hi, lo).astype(float)
    phase = 2 * np.pi * np.cumsum(sweep) / SR
    audio = np.sin(phase)
    audio += rng.normal(0, 0.04, size=t.shape)
    return (audio / np.max(np.abs(audio) + 1e-9) * 32767).astype(np.int16)

def make_traffic(seed=1):
    rng = np.random.default_rng(seed)
    t   = np.linspace(0, DUR, int(SR * DUR), endpoint=False)
    white = rng.standard_normal(t.shape).astype(np.float32)
    from scipy.signal import butter, sosfilt
    sos   = butter(2, 800 / (SR / 2), btype="low", output="sos")
    audio = sosfilt(sos, white) * 2.5
    return (audio / (np.max(np.abs(audio)) + 1e-9) * 32767).astype(np.int16)

def make_tone(freq=440, seed=2):
    rng = np.random.default_rng(seed)
    t   = np.linspace(0, DUR, int(SR * DUR), endpoint=False)
    audio = np.sin(2 * np.pi * freq * t).astype(np.float32)
    audio += rng.normal(0, 0.03, size=t.shape).astype(np.float32)
    return (audio / (np.max(np.abs(audio)) + 1e-9) * 32767).astype(np.int16)


# ── HTTP helpers ──────────────────────────────────────────────────────────────

BASE = "http://127.0.0.1:8000"

def post_json(path, body=None):
    url  = BASE + path
    data = json.dumps(body or {}).encode()
    req  = urllib.request.Request(url, data=data, method="POST",
               headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read()), r.status
    except urllib.error.HTTPError as e:
        return json.loads(e.read().decode()), e.code

def post_wav(wav_path: str, filename: str = "test.wav"):
    """Multipart file upload to /audio/detect."""
    boundary = "----SirenAIBoundary"
    with open(wav_path, "rb") as f:
        file_data = f.read()

    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: audio/wav\r\n\r\n"
    ).encode() + file_data + f"\r\n--{boundary}--\r\n".encode()

    url = BASE + "/audio/detect"
    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read()), r.status
    except urllib.error.HTTPError as e:
        return json.loads(e.read().decode()), e.code


# ── Main ──────────────────────────────────────────────────────────────────────

SEP = "=" * 62

def run_tests():
    # Start sim so auto-dispatch can fire if siren is detected
    print(SEP)
    print("  Starting SUMO simulation...")
    r, s = post_json("/sim/start")
    print(f"  /sim/start -> {s} | {r.get('message', r)}")
    print()

    # Build test WAVs in a temp dir
    tmp = Path(tempfile.mkdtemp())
    siren_wav   = tmp / "siren_test.wav"
    traffic_wav = tmp / "traffic_test.wav"
    tone_wav    = tmp / "tone_test.wav"

    wavfile.write(str(siren_wav),   SR, make_siren())
    wavfile.write(str(traffic_wav), SR, make_traffic())
    wavfile.write(str(tone_wav),    SR, make_tone())

    tests = [
        ("SIREN (expect: is_siren=True,  dispatch_triggered=True)",  siren_wav,   "siren_test.wav"),
        ("TRAFFIC (expect: is_siren=False, dispatch_triggered=False)", traffic_wav, "traffic_test.wav"),
        ("TONE 440Hz (expect: is_siren=False, dispatch_triggered=False)", tone_wav, "tone_test.wav"),
    ]

    for label, wav_path, filename in tests:
        print(SEP)
        print(f"  TEST: {label}")
        print(f"  File: {filename}  ({wav_path.stat().st_size:,} bytes)")
        print()

        result, status = post_wav(str(wav_path), filename)

        print("  HTTP status         :", status)
        print("  is_siren            :", result.get("is_siren"))
        print("  confidence          :", result.get("confidence"))
        print("  label               :", result.get("label"))
        print("  all_proba           :", result.get("all_proba"))
        print("  synthetic_model     :", result.get("synthetic_model"))
        print("  dispatch_triggered  :", result.get("dispatch_triggered"))
        print()
        print("  message:")
        print("   ", result.get("message"))

        if result.get("dispatch_result"):
            dr = result["dispatch_result"]
            print()
            print("  dispatch_result:")
            print("    route           :", dr.get("route"))
            print("    model_used      :", dr.get("model_used"))
            print("    steps_taken     :", dr.get("steps_taken"))
            print("    completed       :", dr.get("completed"))
            print("    navigator_reasoning (first 120 chars):")
            print("     ", str(dr.get("navigator_reasoning", ""))[:120])
            print()
            print("    timing_log:")
            for line in (dr.get("timing_log") or [])[:8]:
                print("      ", line)
        print()

        # Stop and restart sim between tests so each starts fresh
        post_json("/sim/stop")
        post_json("/sim/start")

    print(SEP)
    print("  All tests complete.")
    print()
    print("  Full JSON from TEST 1 (siren):")
    # Re-run siren test for clean final JSON
    wavfile.write(str(siren_wav), SR, make_siren(seed=99))
    r_final, _ = post_wav(str(siren_wav), "siren_final.wav")
    # Strip dispatch_result for brevity
    display = {k: v for k, v in r_final.items() if k != "dispatch_result"}
    print(json.dumps(display, indent=2))


if __name__ == "__main__":
    run_tests()
