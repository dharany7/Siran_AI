"""
anpr/detect_plate.py — Licence plate detection using OpenCV + EasyOCR.

Detection strategy
------------------
Sole OCR engine — EasyOCR (requires craft_mlt_25k.pth + english_g2.pth
in ~/.EasyOCR/model/; download_enabled=False so it never auto-fetches):
  1. _find_plate_region  — bilateral filter → Canny edges → contour search
     → first 4-sided rectangle with plate-like aspect ratio (1.5–6.5).
     If found, EasyOCR runs on that crop; otherwise on the full frame.
  2. _try_easyocr        — runs EasyOCR on the (possibly cropped) ROI,
     returns (text, confidence) pairs sorted by confidence descending.
  3. _pick_best_from_easyocr — prefer strict Indian plate regex, then
     lax alphanumeric regex, then raw joined fallback.

The old Tier-1 character-template matcher (FONT_HERSHEY_SIMPLEX IoU)
has been removed.  It produced correct results only for synthetic images
drawn with the exact same font; on real photographs it generated garbage
(e.g. 'L8IZSII555XI' instead of a real plate string).

Graceful degradation:
  - EasyOCR models absent  → is_detected=False, ocr_engine='none'.
  - No plate in image      → is_detected=False, no crash.
  - Plate fills full frame → _find_plate_region returns None; EasyOCR
    runs on the full image and usually still finds the text.

Public API:
    from anpr.detect_plate import detect_plate
    result = detect_plate("photo.png")
    # Keys: plate_text, is_detected, is_authorized, vehicle_type,
    #       confidence, plate_region, ocr_engine, raw_ocr_text
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional, TypedDict

import cv2
import numpy as np

log = logging.getLogger(__name__)


# ── Allowlist ─────────────────────────────────────────────────────────────────
AMBULANCE_ALLOWLIST: dict[str, str] = {
    "TN09AX4471": "ambulance",
    "TN09AX4472": "ambulance",
    "TN22AX1234": "ambulance",
    "KA05AB1234": "fire_engine",
    "MH12DE1234": "police",
    "DL01AB1234": "ambulance",
    "TN01AB0001": "ambulance",
    "AP01AB0001": "ambulance",
    "KL07AX9999": "ambulance",
    "HR26DQ1234": "police",
    # ── Demo plate: EasyOCR read from test_real_plates/plate4.webp ──────────
    # Verified offline: detect_plate('demo-assets/plates/plate_demo_authorized.jpg')
    # → plate_text='TN87C5106', confidence=0.4666  (raw_ocr='TN87 C5106')
    "TN87C5106":  "ambulance",
}

_PLATE_RE     = re.compile(r'^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{3,4}$')
_PLATE_RE_LAX = re.compile(r'^[A-Z0-9]{6,14}$')

_EASYOCR_CACHE = Path.home() / ".EasyOCR" / "model"


# ── Result type ───────────────────────────────────────────────────────────────

class PlateResult(TypedDict):
    plate_text:    Optional[str]
    raw_ocr_text:  Optional[str]
    is_detected:   bool
    is_authorized: Optional[bool]
    vehicle_type:  str
    confidence:    float
    plate_region:  Optional[tuple]
    ocr_engine:    str


# ── Normalise helpers ─────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", text.upper())


# ── Plate-region detection ────────────────────────────────────────────────────

def _find_plate_region(img: np.ndarray) -> Optional[tuple]:
    h_img, w_img = img.shape[:2]
    img_area     = h_img * w_img

    gray     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    filtered = cv2.bilateralFilter(gray, 11, 17, 17)
    edges    = cv2.Canny(filtered, 30, 200)
    kernel   = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges    = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(
        edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 0.003 * img_area or area > 0.97 * img_area:
            continue
        peri   = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.020 * peri, True)
        if len(approx) == 4:
            x, y, cw, ch = cv2.boundingRect(approx)
            if ch == 0:
                continue
            if 1.5 <= cw / ch <= 6.5:
                return (x, y, cw, ch)
    return None



# ── EasyOCR (sole OCR engine) ─────────────────────────────────────────────────

_easyocr_reader = None


def _try_easyocr(roi: np.ndarray) -> list[tuple[str, float]]:
    det = _EASYOCR_CACHE / "craft_mlt_25k.pth"
    rec = _EASYOCR_CACHE / "english_g2.pth"
    if not (det.exists() and rec.exists()):
        return []
    global _easyocr_reader
    try:
        import easyocr
        if _easyocr_reader is None:
            _easyocr_reader = easyocr.Reader(
                ["en"], gpu=False, verbose=False, download_enabled=False
            )
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB) if roi.ndim == 3 else roi
        results = _easyocr_reader.readtext(roi_rgb, detail=1)
        return sorted(
            [(str(t), float(c)) for (_, t, c) in results],
            key=lambda x: x[1], reverse=True,
        )
    except Exception as exc:
        log.warning("EasyOCR skipped: %s", exc)
        return []


def _pick_best_from_easyocr(
    candidates: list[tuple[str, float]],
) -> tuple[Optional[str], Optional[str], float]:
    for raw, conf in candidates:
        norm = _normalize(raw)
        if _PLATE_RE.match(norm) and conf >= 0.15:
            return norm, raw, conf
    for raw, conf in candidates:
        norm = _normalize(raw)
        if _PLATE_RE_LAX.match(norm) and conf >= 0.15:
            return norm, raw, conf
    if candidates:
        joined = _normalize(" ".join(r for r, _ in candidates))
        if 6 <= len(joined) <= 14:
            avg = sum(c for _, c in candidates) / len(candidates)
            return joined, " ".join(r for r, _ in candidates), avg
    return None, None, 0.0


# ── Public API ────────────────────────────────────────────────────────────────

def detect_plate(image_path: str | Path) -> PlateResult:
    """
    Detect and recognise a licence plate.  Always returns a PlateResult —
    never raises on missing plate.  Raises ValueError only if the file
    cannot be loaded by OpenCV.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Cannot load image: {image_path}")

    log.info("detect_plate: %s  shape=%s", image_path, img.shape)

    # ── Plate region ──────────────────────────────────────────────────────────
    region_box = _find_plate_region(img)
    if region_box:
        x, y, rw, rh = region_box
        hi, wi = img.shape[:2]
        pad = 5
        roi = img[max(0, y - pad): min(hi, y + rh + pad),
                  max(0, x - pad): min(wi, x + rw + pad)]
    else:
        roi = img

    engine:     str            = "none"
    plate_norm: Optional[str]  = None
    plate_raw:  Optional[str]  = None
    confidence: float          = 0.0

    # ── EasyOCR (sole engine) ─────────────────────────────────────────────────
    ocr_candidates = _try_easyocr(roi)
    if ocr_candidates:
        plate_norm, plate_raw, confidence = _pick_best_from_easyocr(ocr_candidates)
        engine = "easyocr" if plate_norm else "none"
    else:
        log.warning("detect_plate: EasyOCR returned no candidates — "
                    "check ~/.EasyOCR/model/ for craft_mlt_25k.pth + english_g2.pth")

    # ── Allowlist lookup ──────────────────────────────────────────────────────
    is_authorized: Optional[bool] = None
    vehicle_type = "unknown"

    if plate_norm:
        key = _normalize(plate_norm)
        if key in AMBULANCE_ALLOWLIST:
            is_authorized = True
            vehicle_type  = AMBULANCE_ALLOWLIST[key]
        else:
            is_authorized = False
            vehicle_type  = "civilian"

    return PlateResult(
        plate_text    = plate_norm  if plate_norm else None,
        raw_ocr_text  = plate_raw   if plate_raw  else None,
        is_detected   = plate_norm is not None,
        is_authorized = is_authorized,
        vehicle_type  = vehicle_type,
        confidence    = round(confidence, 4),
        plate_region  = region_box,
        ocr_engine    = engine,
    )
