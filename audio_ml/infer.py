"""
audio_ml/infer.py — Siren-audio inference module.

Public API
----------
    from audio_ml.infer import predict

    result = predict("path/to/audio.wav")
    # result = {"is_siren": True, "confidence": 0.93, "label": "siren", ...}

The model (audio_ml/model.pkl) must exist before calling predict().
Run ``python -m audio_ml.train`` to create it.

If the model was trained on synthetic data, the result dict includes a
``synthetic_model`` key set to True as a reminder for demo contexts.
"""
from __future__ import annotations

import pickle
import logging
from pathlib import Path
from typing import TypedDict

import numpy as np
import librosa

log = logging.getLogger(__name__)

# ── Paths (same as train.py) ──────────────────────────────────────────────────
AUDIO_ML_DIR = Path(__file__).parent
MODEL_PATH   = AUDIO_ML_DIR / "model.pkl"

# ── Feature-extraction constants (must match train.py) ───────────────────────
SAMPLE_RATE  = 22_050
DURATION_S   = 3.0
N_MFCC       = 40

# ── Lazy-loaded model cache ───────────────────────────────────────────────────
_model_cache: dict | None = None


def _load_model() -> dict:
    """Load (and cache) model.pkl.  Thread-safe after first call."""
    global _model_cache
    if _model_cache is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. "
                "Run 'python -m audio_ml.train' first."
            )
        with open(MODEL_PATH, "rb") as f:
            _model_cache = pickle.load(f)
        log.info(
            "Loaded siren model (synthetic=%s, labels=%s)",
            _model_cache.get("synthetic"),
            _model_cache.get("labels"),
        )
    return _model_cache


def extract_features(filepath: str | Path) -> np.ndarray:
    """
    Extract an 80-dimensional MFCC feature vector from a WAV file.

    Returns np.ndarray of shape (80,): [mean_mfccs | std_mfccs]

    Also imported by train.py so both training and inference share
    the exact same feature pipeline (no train/serve skew).
    """
    y, sr = librosa.load(str(filepath), sr=SAMPLE_RATE,
                         duration=DURATION_S, mono=True)
    if len(y) == 0:
        raise ValueError(f"Audio file is empty or unreadable: {filepath}")
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    return np.concatenate([mfccs.mean(axis=1), mfccs.std(axis=1)])  # (80,)


class PredictResult(TypedDict):
    is_siren:        bool
    confidence:      float
    label:           str     # raw class label: 'siren' | 'traffic' | 'other'
    all_proba:       dict    # {label: probability} for all classes
    synthetic_model: bool    # True if model was trained on synthetic data


def predict(filepath: str | Path) -> PredictResult:
    """
    Predict whether *filepath* contains a siren sound.

    Parameters
    ----------
    filepath : str or Path
        Path to a WAV (or any librosa-compatible) audio file.

    Returns
    -------
    PredictResult dict with keys:
        is_siren        bool   — True when label == 'siren'
        confidence      float  — probability of the predicted class
        label           str    — predicted class
        all_proba       dict   — per-class probabilities
        synthetic_model bool   — model was trained on synthetic data
    """
    model_data = _load_model()
    clf:    object     = model_data["clf"]
    labels: list[str]  = model_data["labels"]

    features = extract_features(filepath).reshape(1, -1)
    proba    = clf.predict_proba(features)[0]          # (n_classes,)
    classes  = clf.classes_                            # class names in RF order

    # Build {label: prob} mapping
    proba_map: dict[str, float] = {str(c): float(p) for c, p in zip(classes, proba)}

    # Highest-probability class
    pred_idx   = int(np.argmax(proba))
    pred_label = str(classes[pred_idx])
    confidence = float(proba[pred_idx])

    log.info(
        "Predict: label=%r  confidence=%.3f  all=%s",
        pred_label, confidence,
        {k: round(v, 3) for k, v in proba_map.items()},
    )

    return PredictResult(
        is_siren        = pred_label == "siren",
        confidence      = round(confidence, 4),
        label           = pred_label,
        all_proba       = {k: round(v, 4) for k, v in proba_map.items()},
        synthetic_model = bool(model_data.get("synthetic", False)),
    )


# ── CLI convenience ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, json
    if len(sys.argv) < 2:
        print("Usage: python -m audio_ml.infer <audio_file.wav>")
        sys.exit(1)
    result = predict(sys.argv[1])
    print(json.dumps(result, indent=2))
