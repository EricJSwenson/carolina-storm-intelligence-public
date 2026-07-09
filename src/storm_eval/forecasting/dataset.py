"""Synthetic Atlantic-storm generator for the landfall-intensity model.

The bundled real data is only three storms -- far too few to train a model. This
module generates many storms with a *physically-plausible statistical structure*
(intensify-then-weaken life cycle, wind/pressure inverse relationship, weakening
as the storm crosses the continental shelf) so the model learns genuine signal
while the metrics stay honest (accuracy is high-but-imperfect, never 100%).

It emits the same ``Storm`` / ``TrackPoint`` objects the HURDAT2 parser produces,
so the identical feature builder runs on synthetic and real data. For production,
point ``build_training_table`` at the full NCEI HURDAT2 download instead -- no
code change beyond the data source.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import List, Tuple

import numpy as np

from storm_eval.ingestion.hurdat2 import Storm, TrackPoint

# NC continental landfall box (matches the gold-layer spatial filter).
NC_LANDFALL = dict(lat=34.5, lon=-76.8)

# Easter egg: synthetic training storms are named after the 2025-26 Carolina
# Hurricanes roster (reigning Stanley Cup champions). These are fictional storms
# used only to train the landfall model -- they never masquerade as real data.
CANES_ROSTER = [
    "AHO", "JARVIS", "SVECHNIKOV", "STAAL", "KOTKANIEMI", "EHLERS", "STANKOVEN",
    "BLAKE", "MILLER", "MARTINOOK", "ROBINSON", "CHATFIELD", "GOSTISBEHERE",
    "JANKOWSKI", "NADEAU", "NYSTROM", "HALL", "ANDERSEN",
]


def _track_point(storm_id, t, rid, status, lat, lon, wind, pres) -> TrackPoint:
    return TrackPoint(
        storm_id=storm_id, obs_time=t, record_id=rid, status=status,
        lat=lat, lon=lon, max_wind_kt=int(round(wind)), min_pressure_mb=int(round(pres)),
        r34_ne=None, r34_se=None, r34_sw=None, r34_nw=None,
        r50_ne=None, r50_se=None, r50_sw=None, r50_nw=None,
        r64_ne=None, r64_se=None, r64_sw=None, r64_nw=None, rmw_nm=None,
    )


def _wind_to_pressure(wind: float, rng) -> float:
    # Inverse wind-pressure relationship with noise (Atlantic-ish fit).
    return 1010.0 - 0.9 * (wind - 30) + rng.normal(0, 4)


def generate_storm(idx: int, year: int, rng: np.random.Generator,
                   make_nc_landfall: bool) -> Tuple[Storm, List[TrackPoint]]:
    storm_id = f"AL{idx:02d}{year}"
    storm = Storm(storm_id, "AL", idx, year, CANES_ROSTER[idx % len(CANES_ROSTER)], 0)

    n = rng.integers(12, 28)                      # number of 6-hourly fixes
    peak = rng.uniform(45, 155)                   # lifetime peak wind (kt)
    t_peak = rng.integers(n // 2, n - 3)          # fix index of peak
    t0 = datetime(year, 9, 1, tzinfo=timezone.utc) + timedelta(days=int(rng.integers(0, 40)))

    lat, lon = rng.uniform(11, 19), rng.uniform(-58, -42)
    # Heading: move WNW, then recurve N toward the coast.
    points: List[TrackPoint] = []
    for i in range(n):
        # intensity: rise to peak, then weaken
        frac = 1 - abs(i - t_peak) / max(t_peak, n - t_peak)
        wind = max(20.0, peak * (0.45 + 0.55 * frac) + rng.normal(0, 4))
        pres = _wind_to_pressure(wind, rng)
        dlat = rng.uniform(0.6, 1.4)
        dlon = rng.uniform(0.6, 1.5) * (1 if i < t_peak else 0.4)
        lat += dlat
        lon += dlon
        status = "HU" if wind >= 64 else ("TS" if wind >= 34 else "TD")
        points.append(_track_point(storm_id, t0 + timedelta(hours=6 * i), "", status,
                                   round(lat, 1), round(lon, 1), wind, pres))

    if make_nc_landfall and len(points) >= 6:
        # Steer the last fixes toward NC and add a landfall fix; landfall wind is
        # the prior intensity reduced by shelf-weakening (the learnable target).
        pre = points[-3]
        weakening = rng.uniform(0.62, 0.95)
        lf_wind = max(25.0, pre.max_wind_kt * weakening + rng.normal(0, 5))
        lf_pres = _wind_to_pressure(lf_wind, rng)
        lf_time = pre.obs_time + timedelta(hours=12)
        lf = _track_point(storm_id, lf_time, "L",
                          "HU" if lf_wind >= 64 else "TS",
                          NC_LANDFALL["lat"], NC_LANDFALL["lon"], lf_wind, lf_pres)
        # Replace tail with an approach + landfall so pre-landfall fixes exist.
        approach = _track_point(storm_id, pre.obs_time + timedelta(hours=6), "",
                                pre.status, 33.0, -77.5,
                                (pre.max_wind_kt + lf_wind) / 2,
                                (pre.min_pressure_mb + lf_pres) / 2)
        points = points[:-1] + [approach, lf]

    storm.n_track_points = len(points)
    return storm, points


def generate_dataset(n_storms: int = 400, seed: int = 11,
                     landfall_fraction: float = 0.55) -> List[Tuple[Storm, List[TrackPoint]]]:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n_storms):
        year = int(rng.integers(1980, 2024))
        make_lf = rng.random() < landfall_fraction
        out.append(generate_storm(i % 30 + 1, year, rng, make_lf))
    return out
