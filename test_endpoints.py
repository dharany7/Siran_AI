"""
test_endpoints.py — Real response test for POST /audio/detect and POST /anpr/verify.

Audio: generates synthetic WAV tones via librosa (chirp / tonality) + scipy fallback.
ANPR:  generates synthetic plate images with OpenCV and uploads them.
Both print the FULL raw JSON response exactly as the server returns it.
"""
from __future__ import annotations
import json
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
from scipy.io import wavfile

BASE = "http://127.0.0.1:8000"
SEP  = "=" * 66

# ─────────────────────────────────────────────────────────────────────────────
# 1. HTTP helpers
# ─────────────────────────────────────────────────────────────────────────────

def post_json(path: str, body: dict | None = None) -> tuple[dict, int]:
    url  = BASE + path
    data = json.dumps(body or {}).encode()
    req  = urllib.request.Request(
        url, data=data, method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read()), r.status
    except urllib.error.HTTPError as e:
        return json.loads(e.read().decode() or "{}"), e.code


def _multipart_body(fields: list[tuple]) -> tuple[bytes, str]:
    """Build a multipart/form-data body from a list of (name, value, mime, filename?) tuples."""
    boundary = "----SirenEndpointTestBoundary777"
    body = b""
    for item in fields:
        name, value = item[0], item[1]
        mime     = item[2] if len(item) > 2 else None
        filename = item[3] if len(item) > 3 else None
        if filename:
            body += (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
                f"Content-Type: {mime}\r\n\r\n"
            ).encode() + value + b"\r\n"
        else:
            body += (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
                f"{value}\r\n"
            ).encode()
    body += f"--{boundary}--\r\n".encode()
    return body, boundary


def post_wav(wav_path: str, filename: str = "test.wav") -> tuple[dict, int]:
    with open(wav_path, "rb") as f:
        file_data = f.read()
    body, boundary = _multipart_body([
        ("file", file_data, "audio/wav", filename),
    ])
    req = urllib.request.Request(
        BASE + "/audio/detect", data=body, method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read()), r.status
    except urllib.error.HTTPError as e:
        return json.loads(e.read().decode() or "{}"), e.code


def post_image(img_path: str, dispatch_id: int | None = None) -> tuple[dict, int]:
    suffix = Path(img_path).suffix.lower()
    mime   = "image/png" if suffix == ".png" else "image/jpeg"
    filename = Path(img_path).name
    with open(img_path, "rb") as f:
        img_data = f.read()
    fields = [("file", img_data, mime, filename)]
    if dispatch_id is not None:
        fields.append(("dispatch_id", str(dispatch_id)))
    body, boundary = _multipart_body(fields)
    req = urllib.request.Request(
        BASE + "/anpr/verify", data=body, method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read()), r.status
    except urllib.error.HTTPError as e:
        return json.loads(e.read().decode() or "{}"), e.code


# ─────────────────────────────────────────────────────────────────────────────
# 2. WAV generators  (using librosa where available, scipy fallback)
# ─────────────────────────────────────────────────────────────────────────────

SR  = 22_050   # sample rate (Hz)
DUR = 3.0      # seconds

def _write_wav(path: str, samples: np.ndarray, sr: int = SR) -> str:
    """Save float32/int16 ndarray to a WAV file; normalise and convert if needed."""
    if samples.dtype != np.int16:
        peak = np.max(np.abs(samples)) + 1e-9
        samples = (samples / peak * 32_767).astype(np.int16)
    wavfile.write(path, sr, samples)
    print(f"  wrote {path}  ({Path(path).stat().st_size:,} bytes)")
    return path


def make_siren_librosa(path: str) -> str:
    """
    Two-tone emergency-siren sweep using librosa.chirp (760↔1050 Hz, 1 Hz rate).
    Falls back to pure numpy if librosa is unavailable.
    """
    t = np.linspace(0, DUR, int(SR * DUR), endpoint=False)
    try:
        import librosa
        # librosa.tone / librosa.chirp generate float32 audio
        half = int(SR * DUR / 2)
        up   = librosa.chirp(fmin=760,  fmax=1050, sr=SR, duration=DUR/2)
        down = librosa.chirp(fmin=1050, fmax=760,  sr=SR, duration=DUR/2)
        audio = np.concatenate([up, down])
        # Add slight noise for realism
        audio += np.random.default_rng(0).normal(0, 0.03, audio.shape).astype(np.float32)
        print("  [librosa.chirp] siren WAV generated")
    except ImportError:
        print("  [librosa not installed — falling back to numpy sweep]")
        lo, hi = 760.0, 1050.0
        sweep = np.where(np.sin(2 * np.pi * 1.0 * t) > 0, hi, lo)
        phase = 2 * np.pi * np.cumsum(sweep) / SR
        audio = np.sin(phase).astype(np.float32)
        audio += np.random.default_rng(0).normal(0, 0.04, audio.shape).astype(np.float32)
    return _write_wav(path, audio)


def make_pure_tone_librosa(path: str, freq: float = 440.0) -> str:
    """Pure 440 Hz tone using librosa.tone (or numpy fallback)."""
    try:
        import librosa
        audio = librosa.tone(frequency=freq, sr=SR, duration=DUR)
        print(f"  [librosa.tone] {freq} Hz tone WAV generated")
    except (ImportError, AttributeError):
        # librosa < 0.10 doesn't have librosa.tone
        try:
            import librosa
            t = np.linspace(0, DUR, int(SR * DUR), endpoint=False)
            audio = np.sin(2 * np.pi * freq * t).astype(np.float32)
            print(f"  [numpy fallback] {freq} Hz tone WAV generated")
        except ImportError:
            t = np.linspace(0, DUR, int(SR * DUR), endpoint=False)
            audio = np.sin(2 * np.pi * freq * t).astype(np.float32)
            print(f"  [numpy] {freq} Hz tone WAV generated")
    return _write_wav(path, audio)


def make_traffic_noise(path: str) -> str:
    """Low-frequency bandlimited white noise to simulate traffic."""
    from scipy.signal import butter, sosfilt
    rng   = np.random.default_rng(42)
    noise = rng.standard_normal(int(SR * DUR)).astype(np.float32)
    sos   = butter(2, 800 / (SR / 2), btype="low", output="sos")
    audio = sosfilt(sos, noise) * 2.5
    print("  [scipy.butter] traffic noise WAV generated")
    return _write_wav(path, audio)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Plate image generators (OpenCV)
# ─────────────────────────────────────────────────────────────────────────────

def make_plate_image(text: str, path: str) -> str:
    import cv2
    img = np.ones((130, 520, 3), dtype=np.uint8) * 255
    cv2.rectangle(img, (4, 4), (515, 125), (10, 10, 10), 4)
    cv2.rectangle(img, (8, 8), (511, 121), (245, 250, 200), -1)
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 1.9, 5
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    x = (520 - tw) // 2
    y = (130 + th) // 2 + 4
    cv2.putText(img, text, (x, y), font, scale, (5, 5, 5), thick)
    cv2.imwrite(path, img)
    print(f"  plate image: {path}  ({Path(path).stat().st_size:,} bytes)")
    return path


def make_blank_image(path: str) -> str:
    import cv2
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    for i in range(300):
        img[i] = [int(i * 0.5), int(i * 0.3), 180]
    cv2.circle(img, (200, 150), 80, (255, 200, 0), -1)
    cv2.imwrite(path, img)
    print(f"  blank image:  {path}  ({Path(path).stat().st_size:,} bytes)")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 4. Server health check
# ─────────────────────────────────────────────────────────────────────────────

def wait_for_server(retries: int = 15, delay: float = 1.5) -> bool:
    for i in range(retries):
        try:
            with urllib.request.urlopen(BASE + "/docs", timeout=3):
                return True
        except Exception:
            if i == 0:
                print(f"  Waiting for server at {BASE}…", end="", flush=True)
            else:
                print(".", end="", flush=True)
            time.sleep(delay)
    print(" TIMEOUT")
    return False


# ─────────────────────────────────────────────────────────────────────────────
# 5. Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── Start server ──────────────────────────────────────────────────────────
    print(SEP)
    print("  Siren AI — POST /audio/detect & POST /anpr/verify live test")
    print(SEP)
    print()

    server_proc = None
    try:
        urllib.request.urlopen(BASE + "/docs", timeout=2)
        print("  ✓ Server already running at", BASE)
    except Exception:
        print("  Starting FastAPI server (uvicorn) on port 8000…")
        server_proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "backend.main:app",
             "--port", "8000", "--log-level", "warning"],
            cwd=str(Path(__file__).parent),
        )
        if not wait_for_server():
            print("  ERROR: server did not start")
            server_proc.terminate()
            sys.exit(1)
        print("\n  ✓ Server started.")
    print()

    # ── Build test assets ─────────────────────────────────────────────────────
    tmp = Path(tempfile.mkdtemp())
    print("  Generating test assets in", tmp)
    siren_wav   = make_siren_librosa(str(tmp / "siren_librosa.wav"))
    tone_wav    = make_pure_tone_librosa(str(tmp / "tone_440hz.wav"))
    traffic_wav = make_traffic_noise(str(tmp / "traffic_noise.wav"))
    auth_png    = make_plate_image("TN09AX4471", str(tmp / "authorized.png"))
    civ_png     = make_plate_image("MH99ZZ1234", str(tmp / "civilian.png"))
    blank_png   = make_blank_image(str(tmp / "noplate.png"))
    print()

    # ──────────────────────────────────────────────────────────────────────────
    #  AUDIO TESTS
    # ──────────────────────────────────────────────────────────────────────────
    print(SEP)
    print("  ❶  POST /audio/detect TESTS")
    print(SEP)

    audio_cases = [
        ("SIREN (librosa chirp 760↔1050 Hz) — expect is_siren=True",  siren_wav,   "siren_librosa.wav"),
        ("TONE 440 Hz (stationary) — expect is_siren=False",           tone_wav,    "tone_440hz.wav"),
        ("TRAFFIC NOISE (bandlimited white noise) — expect is_siren=False", traffic_wav, "traffic_noise.wav"),
    ]

    for label, wav_path, filename in audio_cases:
        print()
        print(f"  ── {label}")
        print(f"     file: {filename}  ({Path(wav_path).stat().st_size:,} bytes)")
        result, status = post_wav(wav_path, filename)
        print(f"     HTTP {status}")
        print()
        # Print FULL JSON response
        print(json.dumps(result, indent=4))
        print()

    # ──────────────────────────────────────────────────────────────────────────
    #  ANPR TESTS
    # ──────────────────────────────────────────────────────────────────────────
    print(SEP)
    print("  ❷  POST /anpr/verify TESTS")
    print(SEP)

    anpr_cases = [
        ("AUTHORIZED plate TN09AX4471 — expect is_authorized=True",  auth_png,  None),
        ("CIVILIAN plate MH99ZZ1234  — expect is_authorized=False", civ_png,   1),
        ("BLANK image (no plate)     — expect is_detected=False",   blank_png, None),
    ]

    for label, img_path, dispatch_id in anpr_cases:
        print()
        print(f"  ── {label}")
        print(f"     file: {Path(img_path).name}  ({Path(img_path).stat().st_size:,} bytes)")
        if dispatch_id:
            print(f"     dispatch_id: {dispatch_id}")
        result, status = post_image(img_path, dispatch_id)
        print(f"     HTTP {status}")
        print()
        # Print FULL JSON response
        print(json.dumps(result, indent=4))
        print()

    print(SEP)
    print("  All tests complete.")
    print(SEP)

    if server_proc:
        server_proc.terminate()


if __name__ == "__main__":
    main()
