"""
test_anpr_direct.py — Standalone smoke test for anpr.detect_plate.

No server required. Generates three synthetic PNG plate images (lossless)
and runs detect_plate() directly, printing full results and pass/fail.
"""
import sys
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, ".")
from anpr.detect_plate import (
    detect_plate, AMBULANCE_ALLOWLIST,
    _ocr_char_matching, _match_char, _CHAR_TEMPLATES,
)

FONT = cv2.FONT_HERSHEY_SIMPLEX


# ── Image generators ──────────────────────────────────────────────────────────

def make_plate_png(text: str, path: str) -> str:
    """Synthetic plate PNG — same parameters as detect_plate's char templates."""
    img = np.ones((130, 520, 3), dtype=np.uint8) * 255
    cv2.rectangle(img, (4, 4),   (515, 125), (10, 10, 10), 4)
    cv2.rectangle(img, (8, 8),   (511, 121), (245, 250, 200), -1)
    fs, thick = 1.9, 5
    (tw, th), _ = cv2.getTextSize(text, FONT, fs, thick)
    x = (520 - tw) // 2
    y = (130 + th) // 2 + 4
    cv2.putText(img, text, (x, y), FONT, fs, (5, 5, 5), thick)
    cv2.imwrite(path, img)
    sz = Path(path).stat().st_size
    print(f"  Created: {path}  ({sz:,} bytes, PNG lossless)")
    return path


def make_blank_png(path: str) -> str:
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    for i in range(300):
        img[i] = [int(i * 0.5), int(i * 0.3), 180]
    cv2.circle(img, (200, 150), 80, (255, 200, 0), -1)
    cv2.circle(img, (100, 80),  40, (0, 200, 255), -1)
    cv2.imwrite(path, img)
    sz = Path(path).stat().st_size
    print(f"  Created: {path}  ({sz:,} bytes)")
    return path


# ── Character template sanity check ──────────────────────────────────────────

def print_char_sample():
    """Show IoU scores for a few known characters to validate templates."""
    test_chars = "TNA40"
    print("  Character template IoU sanity check:")
    for ch in test_chars:
        canvas = np.ones((150, 150), dtype=np.uint8) * 255
        cv2.putText(canvas, ch, (15, 110), FONT, 3.0, 0, 7)
        _, bw = cv2.threshold(canvas, 200, 255, cv2.THRESH_BINARY_INV)
        coords = cv2.findNonZero(bw)
        if coords is None:
            continue
        x, y, w, h = cv2.boundingRect(coords)
        blob = bw[y: y + h, x: x + w]
        best_ch, best_iou = _match_char(blob)
        print(f"    Template '{ch}' → matched '{best_ch}'  IoU={best_iou:.4f}  "
              f"{'✓' if best_ch == ch else '✗ WRONG'}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

SEP = "=" * 62


def run():
    print(SEP)
    print("  Generating synthetic test images (PNG)…")
    auth_png     = make_plate_png("TN09AX4471", "test_authorized.png")
    civilian_png = make_plate_png("MH99ZZ1234", "test_civilian.png")
    blank_png    = make_blank_png("test_blank.png")
    print()

    print_char_sample()

    cases = [
        (
            "TEST 1 — AUTHORIZED plate  (TN09AX4471 in allowlist)",
            auth_png,
            lambda r: r["is_detected"] and r["is_authorized"] is True
                       and r["vehicle_type"] == "ambulance",
            "expect: is_detected=True, is_authorized=True, vehicle_type=ambulance",
        ),
        (
            "TEST 2 — CIVILIAN plate    (MH99ZZ1234 NOT in allowlist)",
            civilian_png,
            lambda r: r["is_detected"] and r["is_authorized"] is False
                       and r["vehicle_type"] == "civilian",
            "expect: is_detected=True, is_authorized=False, vehicle_type=civilian",
        ),
        (
            "TEST 3 — No plate          (plain gradient, graceful handling)",
            blank_png,
            lambda r: not r["is_detected"],
            "expect: is_detected=False, no crash",
        ),
    ]

    all_passed = True
    for label, path, check, expectation in cases:
        print(SEP)
        print(f"  {label}")
        print(f"  {expectation}")
        result = detect_plate(path)
        print()
        print(f"  plate_text   : {result['plate_text']}")
        print(f"  raw_ocr_text : {result['raw_ocr_text']!r}")
        print(f"  is_detected  : {result['is_detected']}")
        print(f"  is_authorized: {result['is_authorized']}")
        print(f"  vehicle_type : {result['vehicle_type']}")
        print(f"  confidence   : {result['confidence']}")
        print(f"  ocr_engine   : {result['ocr_engine']}")
        print(f"  plate_region : {result['plate_region']}")

        ok = check(result)
        all_passed = all_passed and ok
        print(f"\n  {'✓ PASS' if ok else '✗ FAIL'}")
        print()

    print(SEP)
    if all_passed:
        print("  ✓ All 3 tests PASSED.")
    else:
        print("  Some tests FAILED.")
    print(SEP)


if __name__ == "__main__":
    run()
