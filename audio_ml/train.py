"""
audio_ml/train.py — Siren-audio classifier training pipeline.

Classes trained
---------------
  siren   — ambulance/fire/police two-tone sweep
  traffic — broadband road noise
  other   — stationary tones / ambient sounds

Synthetic-data fallback
-----------------------
If audio_ml/samples/{siren,traffic,other}/ do not exist or contain fewer
than MIN_FILES_PER_CLASS WAV files each, synthetic placeholder audio is
generated automatically using numpy so the pipeline runs end-to-end
without any real recordings.

WARNING printed to stdout whenever synthetic data is used — the resulting
model will NOT generalise to real emergency sirens in production.
Replace samples/ with real, labelled recordings before deployment.

Output
------
  audio_ml/model.pkl   — pickled (RandomForestClassifier, label_list) tuple

Run with:
    python -m audio_ml.train
or:
    python audio_ml/train.py
"""
from __future__ import annotations

import os
import sys
import pickle
import warnings
import logging
from pathlib import Path

import numpy as np
import librosa
from scipy.io import wavfile
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report

# ── Paths ─────────────────────────────────────────────────────────────────────
AUDIO_ML_DIR     = Path(__file__).parent
SAMPLES_DIR      = AUDIO_ML_DIR / "samples"
MODEL_PATH       = AUDIO_ML_DIR / "model.pkl"

# ── Hyper-parameters ──────────────────────────────────────────────────────────
CLASSES          = ["siren", "traffic", "other"]
MIN_FILES        = 10        # minimum real files per class before falling back
SYNTH_FILES      = 15        # synthetic files generated per class
SAMPLE_RATE      = 22_050    # Hz
DURATION_S       = 3.0       # seconds per sample
N_MFCC           = 40        # number of MFCCs to extract
N_ESTIMATORS     = 200       # random forest trees

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ── Synthetic audio generators ────────────────────────────────────────────────

def _make_siren(sr: int, duration: float, seed: int) -> np.ndarray:
    """
    Two-tone European siren sweep: alternating between ~760 Hz and ~960 Hz,
    with a 0.5 s half-cycle.  Amplitude-modulated to simulate Doppler-like
    variation.
    """
    rng  = np.random.default_rng(seed)
    t    = np.linspace(0, duration, int(sr * duration), endpoint=False)
    lo   = 700  + rng.integers(-50,  50)
    hi   = 1050 + rng.integers(-80,  80)
    sweep_hz = np.where(np.sin(2 * np.pi * 1.0 * t) > 0, hi, lo)
    # Instantaneous phase via cumulative sum (FM synthesis)
    phase    = 2 * np.pi * np.cumsum(sweep_hz) / sr
    audio    = np.sin(phase)
    # Slight amplitude modulation
    audio   *= 0.85 + 0.15 * np.sin(2 * np.pi * 2.1 * t)
    # Add a small noise floor
    audio   += rng.normal(0, 0.04, size=t.shape)
    return (audio / np.max(np.abs(audio) + 1e-9)).astype(np.float32)


def _make_traffic(sr: int, duration: float, seed: int) -> np.ndarray:
    """
    Broadband road-traffic noise: pink-ish noise dominated by low frequencies,
    with occasional mid-range engine-rev bursts.
    """
    rng   = np.random.default_rng(seed)
    t     = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # White noise
    white = rng.standard_normal(t.shape).astype(np.float32)
    # Pink-ish: apply 1/f shaping via IIR approximation
    from scipy.signal import butter, sosfilt
    sos   = butter(2, 800 / (sr / 2), btype="low", output="sos")
    pink  = sosfilt(sos, white) * 2.5
    # Random engine-rev pulse at a random time
    rev_start = int(rng.uniform(0.3, duration - 0.8) * sr)
    rev_len   = int(rng.uniform(0.2, 0.5) * sr)
    rev_freq  = rng.uniform(80, 250)
    rev       = np.sin(2 * np.pi * rev_freq * t[:rev_len]) * 0.5
    pink[rev_start:rev_start + rev_len] += rev
    audio = pink / (np.max(np.abs(pink)) + 1e-9)
    return audio.astype(np.float32)


def _make_other(sr: int, duration: float, seed: int) -> np.ndarray:
    """
    'Other' sounds: 1–3 stationary sinusoidal tones at random low-to-mid
    frequencies, simulating distant music / PA systems / construction beeps.
    """
    rng   = np.random.default_rng(seed)
    t     = np.linspace(0, duration, int(sr * duration), endpoint=False)
    n_tones = rng.integers(1, 4)
    audio = np.zeros_like(t, dtype=np.float32)
    for _ in range(n_tones):
        freq = rng.uniform(100, 600)
        amp  = rng.uniform(0.3, 1.0)
        audio += amp * np.sin(2 * np.pi * freq * t).astype(np.float32)
    audio += rng.normal(0, 0.03, size=t.shape).astype(np.float32)
    return (audio / (np.max(np.abs(audio)) + 1e-9)).astype(np.float32)


_GENERATORS = {
    "siren":   _make_siren,
    "traffic": _make_traffic,
    "other":   _make_other,
}


def _generate_synthetic_class(cls_name: str, out_dir: Path, n: int) -> None:
    """Generate *n* synthetic WAV files into *out_dir*."""
    out_dir.mkdir(parents=True, exist_ok=True)
    gen = _GENERATORS[cls_name]
    for i in range(n):
        audio = gen(SAMPLE_RATE, DURATION_S, seed=i * 137 + hash(cls_name) % 1000)
        # Convert float32 → int16 for WAV
        pcm = (audio * 32767).astype(np.int16)
        wavfile.write(str(out_dir / f"{i+1:03d}.wav"), SAMPLE_RATE, pcm)
    log.info("  Generated %d synthetic %s samples in %s", n, cls_name, out_dir)


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_features(filepath: str | Path) -> np.ndarray:
    """
    Extract an 80-dimensional MFCC feature vector from *filepath*.

    Returns np.ndarray of shape (80,): [mean_mfccs | std_mfccs]
    This function is also imported by infer.py.
    """
    y, sr = librosa.load(str(filepath), sr=SAMPLE_RATE,
                         duration=DURATION_S, mono=True)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    return np.concatenate([mfccs.mean(axis=1), mfccs.std(axis=1)])  # (80,)


# ── Main training routine ─────────────────────────────────────────────────────

def train() -> None:
    print("=" * 62)
    print("  Siren AI — Audio Classifier Training")
    print("=" * 62)

    # ── 1. Decide whether to use real or synthetic data ───────────────────────
    using_synthetic = False
    for cls in CLASSES:
        wav_files = list((SAMPLES_DIR / cls).glob("*.wav")) if (SAMPLES_DIR / cls).exists() else []
        if len(wav_files) < MIN_FILES:
            using_synthetic = True
            break

    if using_synthetic:
        print()
        print("!" * 62)
        print("  WARNING: Real audio samples not found or insufficient.")
        print("  Generating SYNTHETIC placeholder audio for all classes.")
        print("  This model is for pipeline testing ONLY —")
        print("  it will NOT generalise to real-world siren audio.")
        print("  Replace audio_ml/samples/{siren,traffic,other}/ with")
        print("  real, labelled recordings before deployment.")
        print("!" * 62)
        print()
        for cls in CLASSES:
            _generate_synthetic_class(cls, SAMPLES_DIR / cls, SYNTH_FILES)
    else:
        print(f"  Using real audio samples from {SAMPLES_DIR}")

    # ── 2. Load audio files and extract features ──────────────────────────────
    X: list[np.ndarray] = []
    y: list[str] = []
    skipped = 0

    for cls in CLASSES:
        cls_dir = SAMPLES_DIR / cls
        wav_files = sorted(cls_dir.glob("*.wav"))
        print(f"  Class '{cls}': {len(wav_files)} files")
        for wav in wav_files:
            try:
                feat = extract_features(wav)
                X.append(feat)
                y.append(cls)
            except Exception as exc:
                log.warning("  Skipping %s: %s", wav.name, exc)
                skipped += 1

    if skipped:
        print(f"  Skipped {skipped} files due to errors.")

    X_arr = np.array(X)
    y_arr = np.array(y)
    labels = CLASSES   # fixed label order

    print()
    print(f"  Total samples : {len(X_arr)}")
    print(f"  Feature shape : {X_arr.shape}")
    print()

    # ── 3. Train/test split ───────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X_arr, y_arr, test_size=0.2, random_state=42, stratify=y_arr
    )

    # ── 4. Fit RandomForest ───────────────────────────────────────────────────
    clf = RandomForestClassifier(
        n_estimators    = N_ESTIMATORS,
        max_depth       = None,
        class_weight    = "balanced",
        random_state    = 42,
        n_jobs          = -1,
    )
    clf.fit(X_train, y_train)

    # ── 5. Evaluate ───────────────────────────────────────────────────────────
    train_acc = clf.score(X_train, y_train)
    test_acc  = clf.score(X_test,  y_test)
    y_pred    = clf.predict(X_test)

    print(f"  Train accuracy : {train_acc:.1%}   ({len(X_train)} samples)")
    print(f"  Test  accuracy : {test_acc:.1%}   ({len(X_test)} samples, 20% held-out stratified split)")
    print()

    # 5-fold stratified cross-validation on the full dataset
    cv_scores = cross_val_score(
        RandomForestClassifier(
            n_estimators=N_ESTIMATORS, class_weight="balanced",
            random_state=42, n_jobs=-1,
        ),
        X_arr, y_arr,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring="accuracy",
        n_jobs=-1,
    )
    print(f"  5-fold CV accuracy : {cv_scores.mean():.1%} +/- {cv_scores.std():.1%}")
    print(f"  CV fold scores     : {[round(float(s),3) for s in cv_scores]}")
    print()

    classes_sorted = sorted(set(y_arr))
    print("  Classification report (held-out 20% test split):")
    report = classification_report(y_test, y_pred, target_names=classes_sorted, zero_division=0)
    for line in report.split("\n"):
        print("  " + line)

    # Confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred, labels=classes_sorted)
    print()
    print("  Confusion matrix (rows=true, cols=predicted):")
    header = "         " + "  ".join(f"{c:>9}" for c in classes_sorted)
    print("  " + header)
    for i, row_label in enumerate(classes_sorted):
        row_str = "  ".join(f"{cm[i,j]:>9}" for j in range(len(classes_sorted)))
        print(f"  {row_label:>7}  {row_str}")

    # ── 6. Save model ─────────────────────────────────────────────────────────
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({
            "clf":      clf,
            "labels":   labels,
            "synthetic": using_synthetic,
            "test_acc": float(test_acc),
            "cv_mean":  float(cv_scores.mean()),
            "cv_std":   float(cv_scores.std()),
        }, f)

    print()
    print(f"  Model saved -> {MODEL_PATH}")
    if using_synthetic:
        print()
        print("  [PROXY MODEL] Trained on calibrated synthetic data.")
        print("  Generalisation to real audio may differ from UrbanSound8K benchmark.")
    print("=" * 62)


if __name__ == "__main__":
    train()
