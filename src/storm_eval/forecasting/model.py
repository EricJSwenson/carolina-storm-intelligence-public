"""Landfall-intensity classifier: predict NC landfall Saffir-Simpson category
from a storm's pre-landfall track.

A small, honest supervised model (standardize -> gradient-boosted trees). Kept
deliberately simple and interpretable; the point is a correct ML *workflow*
(leakage-safe split, calibrated expectations, real metrics), not a SOTA forecast.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from storm_eval.config import DATA_DIR
from storm_eval.forecasting.features import FEATURE_NAMES

MODEL_PATH = DATA_DIR / "models" / "landfall_intensity.pkl"


def build_model() -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", HistGradientBoostingClassifier(max_depth=3, learning_rate=0.1,
                                                max_iter=200, random_state=0)),
    ])


def train(X: np.ndarray, y: np.ndarray) -> Pipeline:
    model = build_model()
    model.fit(X, y)
    return model


def save(model: Pipeline, path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump({"model": model, "features": FEATURE_NAMES}, fh)


def load(path: Path = MODEL_PATH) -> Pipeline:
    with open(path, "rb") as fh:
        return pickle.load(fh)["model"]
