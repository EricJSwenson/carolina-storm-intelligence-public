"""Build the model's feature table from storm tracks.

For each storm that makes an NC landfall, we use only the fixes *before* landfall
(strict lead time -- no peeking at the landfall fix) to predict the landfall
Saffir-Simpson category. The same builder runs on synthetic tracks or real
HURDAT2, since both are lists of ``TrackPoint``.

Features (all knowable before landfall):
    wind_pre        last pre-landfall max wind (kt)
    pres_pre        last pre-landfall min pressure (mb)
    peak_wind       lifetime peak wind so far (kt)
    intens_24h      wind change over the 24h before landfall (kt)
    lat_pre         latitude of the last pre-landfall fix
    fwd_speed       forward speed proxy (deg / 6h) over the last segment
Target:
    landfall_category   Saffir-Simpson category at the landfall fix (0-5)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from storm_eval.ingestion.hurdat2 import Storm, TrackPoint, saffir_simpson_category

FEATURE_NAMES = ["wind_pre", "pres_pre", "peak_wind", "intens_24h", "lat_pre", "fwd_speed"]


def _features_for_storm(points: List[TrackPoint]) -> Optional[Tuple[List[float], int]]:
    landfalls = [p for p in points if p.record_id == "L"]
    if not landfalls:
        return None
    lf = landfalls[0]
    pre = [p for p in points if p.obs_time < lf.obs_time and p.max_wind_kt is not None]
    if len(pre) < 4:
        return None
    last = pre[-1]
    wind_24h_ago = pre[-5].max_wind_kt if len(pre) >= 5 else pre[0].max_wind_kt
    peak = max(p.max_wind_kt for p in pre)
    fwd = abs(last.lat - pre[-2].lat) + abs(last.lon - pre[-2].lon)
    feats = [
        float(last.max_wind_kt),
        float(last.min_pressure_mb if last.min_pressure_mb is not None else 1010),
        float(peak),
        float(last.max_wind_kt - wind_24h_ago),
        float(last.lat),
        float(fwd),
    ]
    label = saffir_simpson_category(lf.max_wind_kt)
    return feats, int(label)


def build_training_table(storms: List[Tuple[Storm, List[TrackPoint]]]):
    """Return (X, y, years) over all storms that have a usable NC landfall."""
    X, y, years = [], [], []
    for storm, points in storms:
        row = _features_for_storm(points)
        if row is not None:
            X.append(row[0]); y.append(row[1]); years.append(storm.year)
    return np.array(X, dtype=float), np.array(y, dtype=int), np.array(years, dtype=int)
