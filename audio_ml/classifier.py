"""
classifier.py — Siren audio classifier (training + inference).

Training uses MFCC features extracted with librosa and a
scikit-learn RandomForestClassifier.  A trained model is persisted
as a joblib file so inference is fast.

Usage (training):
    python -m audio_ml.classifier --train --data-dir /path/to/wav/files

Usage (inference):
    from audio_ml.classifier import SirenClassifier
    clf = SirenClassifier.load("models/siren_clf.joblib")
    label, confidence = clf.predict("ambulance_clip.wav")
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Tuple

import joblib
import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

MODEL_DIR = Path("models")
DEFAULT_MODEL_PATH = MODEL_DIR / "siren_clf.joblib"
DEFAULT_LABEL_PATH = MODEL_DIR / "siren_labels.joblib"
N_MFCC = 40
SAMPLE_RATE = 22_050
MAX_DURATION = 4.0  # seconds


def extract_features(audio_path: str) -> np.ndarray:
    """Return an MFCC feature vector for a wav/mp3 file."""
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, duration=MAX_DURATION, mono=True)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    return np.mean(mfccs, axis=1)  # shape: (N_MFCC,)


class SirenClassifier:
    """Thin wrapper around a scikit-learn classifier for siren audio."""

    def __init__(self, clf: RandomForestClassifier, encoder: LabelEncoder) -> None:
        self._clf = clf
        self._encoder = encoder

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        label_path: str | Path = DEFAULT_LABEL_PATH,
    ) -> None:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._clf, model_path)
        joblib.dump(self._encoder, label_path)
        logger.info("Model saved to %s", model_path)

    @classmethod
    def load(
        cls,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        label_path: str | Path = DEFAULT_LABEL_PATH,
    ) -> "SirenClassifier":
        clf = joblib.load(model_path)
        encoder = joblib.load(label_path)
        return cls(clf, encoder)

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, audio_path: str) -> Tuple[str, float]:
        """
        Predict the siren class for a given audio file.

        Returns:
            (label, confidence) where confidence is the max class probability.
        """
        features = extract_features(audio_path).reshape(1, -1)
        proba = self._clf.predict_proba(features)[0]
        idx = int(np.argmax(proba))
        label = self._encoder.inverse_transform([idx])[0]
        return label, float(proba[idx])

    # ── Training ──────────────────────────────────────────────────────────────

    @classmethod
    def train(cls, data_dir: str) -> "SirenClassifier":
        """
        Train a new classifier from a directory structured as:
            data_dir/
                ambulance/  *.wav
                fire/       *.wav
                police/     *.wav
                ...

        Returns a fitted SirenClassifier instance.
        """
        X, y = [], []
        data_path = Path(data_dir)
        for label_dir in data_path.iterdir():
            if not label_dir.is_dir():
                continue
            for audio_file in label_dir.glob("*.wav"):
                try:
                    X.append(extract_features(str(audio_file)))
                    y.append(label_dir.name)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Skipping %s: %s", audio_file, exc)

        if not X:
            raise ValueError(f"No wav files found under {data_dir}")

        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y)
        X_arr = np.array(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_arr, y_encoded, test_size=0.2, random_state=42
        )
        clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        clf.fit(X_train, y_train)

        acc = clf.score(X_test, y_test)
        logger.info("Training complete — test accuracy: %.2f", acc)
        return cls(clf, encoder)


# ── CLI entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Siren classifier CLI")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--data-dir", default="data/audio", help="Labelled audio directory")
    parser.add_argument("--predict", metavar="FILE", help="Predict class of an audio file")
    args = parser.parse_args()

    if args.train:
        model = SirenClassifier.train(args.data_dir)
        model.save()
    elif args.predict:
        model = SirenClassifier.load()
        label, conf = model.predict(args.predict)
        print(f"Predicted: {label} (confidence={conf:.2%})")
