"""Apply the trained landfall-intensity model to a live storm's current track.

This is the *defensible* live-prediction piece: given an active storm's recent
positions/intensity, the model estimates the category it would come ashore at.
It is explicitly a model estimate shown beside the official NHC forecast -- NOT
an official forecast. The platform's purpose remains evaluation, not operational
prediction; this panel is a bonus for users who want it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from storm_eval.forecasting.features import FEATURE_NAMES


@dataclass
class LiveEstimate:
    category: Optional[int]
    confidence: Optional[float]
    basis: str
    disclaimer: str = ("Model estimate from the storm's recent track — NOT an official "
                       "forecast. Always follow National Hurricane Center advisories.")


def _features_from_live_track(track: List[dict]) -> Optional[List[float]]:
    """Build the model's feature row from a live storm's recent fixes.

    ``track`` items need: wind (kt), pres (mb), lat, lon, ordered oldest->newest.
    Mirrors forecasting.features so the same model applies.
    """
    pts = [t for t in track if t.get("wind") is not None]
    if len(pts) < 4:
        return None
    last = pts[-1]
    wind_24h_ago = pts[-5]["wind"] if len(pts) >= 5 else pts[0]["wind"]
    peak = max(t["wind"] for t in pts)
    fwd = abs(last["lat"] - pts[-2]["lat"]) + abs(last["lon"] - pts[-2]["lon"])
    return [
        float(last["wind"]),
        float(last.get("pres") or 1010),
        float(peak),
        float(last["wind"] - wind_24h_ago),
        float(last["lat"]),
        float(fwd),
    ]


def estimate_landfall(track: List[dict], model) -> LiveEstimate:
    feats = _features_from_live_track(track)
    if feats is None:
        return LiveEstimate(None, None, "insufficient track history (need ≥4 fixes)")
    X = np.array([feats], dtype=float)
    cat = int(model.predict(X)[0])
    conf = None
    if hasattr(model, "predict_proba"):
        try:
            conf = float(np.max(model.predict_proba(X)[0]))
        except Exception:  # noqa: BLE001
            conf = None
    basis = ", ".join(f"{n}={v:.0f}" for n, v in zip(FEATURE_NAMES, feats))
    return LiveEstimate(cat, conf, basis)
