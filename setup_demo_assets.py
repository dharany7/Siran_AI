"""
setup_demo_assets.py — One-time (idempotent) demo-asset population.

Creates demo-assets/
    sirens/
        siren_1.wav  siren_2.wav  siren_3.wav
        ambient_1.wav ambient_2.wav ambient_3.wav
    plates/
        plate_demo.jpg

Sources
-------
Siren clips  : audio_ml/samples/siren/   (UrbanSound8K class-8 wavs)
Ambient clips: audio_ml/samples/other/   (various non-siren classes)
Plate image  : test_real_plates/plate2.jpg

Run:
    python setup_demo_assets.py
"""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).parent

# ── Source directories ────────────────────────────────────────────────────────
SIREN_SRC   = ROOT / "audio_ml" / "samples" / "siren"
AMBIENT_SRC = ROOT / "audio_ml" / "samples" / "other"
PLATES_SRC  = ROOT / "test_real_plates"

# ── Destination directories ───────────────────────────────────────────────────
DEMO_ROOT   = ROOT / "demo-assets"
SIREN_DST   = DEMO_ROOT / "sirens"
PLATE_DST   = DEMO_ROOT / "plates"

# ── How many clips of each kind to expose ─────────────────────────────────────
N_SIREN   = 3
N_AMBIENT = 3


def _pick_wavs(src_dir: Path, n: int) -> list[Path]:
    """Return the first *n* wav files found in *src_dir* (sorted by name)."""
    wavs = sorted(src_dir.glob("*.wav"))
    if len(wavs) < n:
        raise FileNotFoundError(
            f"Need {n} wavs in {src_dir}, found {len(wavs)}. "
            "Run audio data setup first."
        )
    return wavs[:n]


def main() -> None:
    SIREN_DST.mkdir(parents=True, exist_ok=True)
    PLATE_DST.mkdir(parents=True, exist_ok=True)

    # ── Siren clips ───────────────────────────────────────────────────────────
    for i, src in enumerate(_pick_wavs(SIREN_SRC, N_SIREN), start=1):
        dst = SIREN_DST / f"siren_{i}.wav"
        if not dst.exists():
            shutil.copy2(src, dst)
            print(f"  [OK] {src.name}  ->  demo-assets/sirens/siren_{i}.wav")
        else:
            print(f"  [--] siren_{i}.wav already present")

    # ── Ambient clips ─────────────────────────────────────────────────────────
    for i, src in enumerate(_pick_wavs(AMBIENT_SRC, N_AMBIENT), start=1):
        dst = SIREN_DST / f"ambient_{i}.wav"
        if not dst.exists():
            shutil.copy2(src, dst)
            print(f"  [OK] {src.name}  ->  demo-assets/sirens/ambient_{i}.wav")
        else:
            print(f"  [--] ambient_{i}.wav already present")

    # ── Plate images ─────────────────────────────────────────────────────────
    # plate_demo.jpg          → civilian plate (ANPR reject path demo)
    # plate_demo_authorized.jpg → ambulance plate TN87C5106 (green-light path demo)
    plate_src = PLATES_SRC / "plate2.jpg"
    plate_dst = PLATE_DST / "plate_demo.jpg"
    if plate_src.exists():
        if not plate_dst.exists():
            shutil.copy2(plate_src, plate_dst)
            print(f"  [OK] plate2.jpg  ->  demo-assets/plates/plate_demo.jpg")
        else:
            print("  [--] plate_demo.jpg already present")
    else:
        print(f"  [!!] {plate_src} not found -- ANPR stage will use fallback")

    # Authorized plate: plate4.webp converted to JPEG via OpenCV (webp → BGR → JPEG)
    auth_src = PLATES_SRC / "plate4.webp"
    auth_dst = PLATE_DST / "plate_demo_authorized.jpg"
    if auth_src.exists():
        if not auth_dst.exists():
            try:
                import cv2
                img = cv2.imread(str(auth_src))
                if img is not None:
                    cv2.imwrite(str(auth_dst), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    print(f"  [OK] plate4.webp  ->  demo-assets/plates/plate_demo_authorized.jpg")
                else:
                    shutil.copy2(auth_src, auth_dst)
                    print(f"  [OK] plate4.webp  ->  demo-assets/plates/plate_demo_authorized.jpg (raw copy)")
            except ImportError:
                shutil.copy2(auth_src, auth_dst)
                print(f"  [OK] plate4.webp  ->  demo-assets/plates/plate_demo_authorized.jpg (raw copy; cv2 absent)")
        else:
            print("  [--] plate_demo_authorized.jpg already present")
    else:
        print(f"  [!!] {auth_src} not found -- authorized-plate demo path unavailable")

    print("\ndemo-assets setup complete.")
    print("Contents:")
    for p in sorted(DEMO_ROOT.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
