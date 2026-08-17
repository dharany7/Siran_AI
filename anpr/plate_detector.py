"""
plate_detector.py — ANPR module using OpenCV + EasyOCR.

Workflow:
    1. Pre-process the image (grayscale, bilateral filter, edge detection).
    2. Find contours and isolate the most plate-like rectangle.
    3. Run EasyOCR over the cropped region.
    4. Return the raw text and bounding-box coordinates.

Usage:
    from anpr.plate_detector import PlateDetector
    detector = PlateDetector()
    result = detector.detect("car.jpg")
    print(result.text, result.confidence)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import easyocr
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PlateResult:
    text: str
    confidence: float
    bbox: Optional[List] = field(default=None)   # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    image_path: Optional[str] = field(default=None)


class PlateDetector:
    """
    Automatic Number Plate Recognition using OpenCV + EasyOCR.
    The EasyOCR Reader is loaded once and reused for efficiency.
    """

    def __init__(self, languages: list[str] | None = None) -> None:
        self._reader = easyocr.Reader(languages or ["en"], gpu=False)

    def detect(self, image_path: str) -> Optional[PlateResult]:
        """
        Detect and read the licence plate in an image file.

        Args:
            image_path: Path to the source image.

        Returns:
            PlateResult if a plate is found, else None.
        """
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Cannot open image: {image_path}")

        # ── Pre-processing ───────────────────────────────────────────────────
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        edges = cv2.Canny(gray, 30, 200)

        # ── Contour search ───────────────────────────────────────────────────
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

        plate_crop = None
        for cnt in contours:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.018 * peri, True)
            if len(approx) == 4:  # rectangle → likely a plate
                x, y, w, h = cv2.boundingRect(approx)
                plate_crop = gray[y : y + h, x : x + w]
                break

        target = plate_crop if plate_crop is not None else gray

        # ── OCR ──────────────────────────────────────────────────────────────
        results = self._reader.readtext(target)
        if not results:
            logger.warning("No text found in %s", image_path)
            return None

        # Pick the result with the highest confidence
        best = max(results, key=lambda r: r[2])
        bbox, text, confidence = best
        return PlateResult(
            text=text.upper().strip(),
            confidence=float(confidence),
            bbox=bbox,
            image_path=image_path,
        )
