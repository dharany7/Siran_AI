"""
test_anpr.py — End-to-end HTTP test for POST /anpr/verify.

Generates synthetic PNG plate images (lossless) and uploads each to the
running FastAPI server. Tests:
  1. Authorized ambulance plate  → is_authorized=True
  2. Civilian plate              → is_authorized=False, security event logged
  3. No plate image              → is_detected=False, graceful handling
"""
import json, tempfile, urllib.request, urllib.error
from pathlib import Path
import numpy as np
import cv2

BASE  = "http://127.0.0.1:8000"
FONT  = cv2.FONT_HERSHEY_SIMPLEX


# ── Image generators ──────────────────────────────────────────────────────────

def make_plate(text: str, path: str) -> str:
    img = np.ones((130, 520, 3), dtype=np.uint8) * 255
    cv2.rectangle(img, (4,4), (515,125), (10,10,10), 4)
    cv2.rectangle(img, (8,8), (511,121), (245,250,200), -1)
    fs, th_k = 1.9, 5
    (tw,th),_ = cv2.getTextSize(text, FONT, fs, th_k)
    x=(520-tw)//2; y=(130+th)//2+4
    cv2.putText(img, text, (x,y), FONT, fs, (5,5,5), th_k)
    cv2.imwrite(path, img)
    print(f"  {path}  ({Path(path).stat().st_size:,} bytes)")
    return path

def make_blank(path: str) -> str:
    img = np.zeros((300,400,3), dtype=np.uint8)
    for i in range(300): img[i] = [int(i*.5), int(i*.3), 180]
    cv2.circle(img,(200,150),80,(255,200,0),-1)
    cv2.imwrite(path, img)
    print(f"  {path}  ({Path(path).stat().st_size:,} bytes)")
    return path


# ── HTTP helpers ──────────────────────────────────────────────────────────────

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

def post_image(img_path: str, dispatch_id=None) -> tuple:
    boundary = "----SirenANPRBoundary"
    filename = Path(img_path).name
    with open(img_path, "rb") as f:
        img_data = f.read()
    mime = "image/png" if filename.endswith(".png") else "image/jpeg"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: {mime}\r\n\r\n"
    ).encode() + img_data + b"\r\n"
    if dispatch_id is not None:
        body += (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="dispatch_id"\r\n\r\n'
            f"{dispatch_id}\r\n"
        ).encode()
    body += f"--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        BASE + "/anpr/verify", data=body, method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read()), r.status
    except urllib.error.HTTPError as e:
        return json.loads(e.read().decode()), e.code


# ── Tests ─────────────────────────────────────────────────────────────────────

SEP = "=" * 62

def run():
    tmp = Path(tempfile.mkdtemp())
    auth_png     = str(tmp / "authorized.png")
    civilian_png = str(tmp / "civilian.png")
    blank_png    = str(tmp / "noplate.png")

    print(SEP)
    print("  Generating test images…")
    make_plate("TN09AX4471", auth_png)
    make_plate("MH99ZZ1234", civilian_png)
    make_blank(blank_png)
    print()

    cases = [
        ("TEST 1 — AUTHORIZED (TN09AX4471)", auth_png, None),
        ("TEST 2 — CIVILIAN   (MH99ZZ1234)", civilian_png, 1),
        ("TEST 3 — NO PLATE   (plain image)", blank_png, None),
    ]

    for label, path, dispatch_id in cases:
        print(SEP)
        print(f"  {label}")
        result, status = post_image(path, dispatch_id)
        print(f"  HTTP status        : {status}")
        print(f"  is_detected        : {result.get('is_detected')}")
        print(f"  plate_text         : {result.get('plate_text')}")
        print(f"  is_authorized      : {result.get('is_authorized')}")
        print(f"  vehicle_type       : {result.get('vehicle_type')}")
        print(f"  confidence         : {result.get('confidence')}")
        print(f"  dispatch_cancelled : {result.get('dispatch_cancelled')}")
        print(f"  security_event_id  : {result.get('security_event_id')}")
        print(f"  message:")
        print(f"    {result.get('message')}")
        print()

    # Full JSON output for the civilian plate test (most interesting case)
    print(SEP)
    print("  FULL JSON — TEST 2 (civilian plate, dispatch_cancelled):")
    r2, _ = post_image(civilian_png, 1)
    print(json.dumps(r2, indent=2))
    print(SEP)
    print("  All HTTP tests complete.")

if __name__ == "__main__":
    run()
