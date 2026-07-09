"""Ingest NDBC buoy observations off the Carolina coast.

Stations near Morehead City: BFTN7 (Beaufort), CLKN7 (Cape Lookout),
41064 (Onslow Bay). NDBC publishes whitespace-delimited realtime files at
https://www.ndbc.noaa.gov/data/realtime2/{STATION}.txt . These numeric time
series are joined to storm tracks in the silver layer so model answers about
wind/pressure can be checked against measured values.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

from storm_eval.config import SAMPLES_DIR

REALTIME_BASE = "https://www.ndbc.noaa.gov/data/realtime2/"
CAROLINA_STATIONS = ("BFTN7", "CLKN7", "41064")


@dataclass
class BuoyObservation:
    station: str
    obs_time: str          # ISO-8601 UTC
    wind_speed_ms: Optional[float]
    gust_ms: Optional[float]
    pressure_mb: Optional[float]
    air_temp_c: Optional[float]
    water_temp_c: Optional[float]
    wave_height_m: Optional[float]


def _f(v: str) -> Optional[float]:
    v = (v or "").strip()
    if v in ("", "MM", "999", "99.0", "9999.0"):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def read_buoy(path: Path | str | None = None) -> Iterator[BuoyObservation]:
    """Read bundled sample buoy observations (offline demo path)."""
    path = Path(path) if path else SAMPLES_DIR / "buoy_sample.csv"
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            yield BuoyObservation(
                station=row["station"],
                obs_time=row["obs_time"],
                wind_speed_ms=_f(row.get("wind_speed_ms", "")),
                gust_ms=_f(row.get("gust_ms", "")),
                pressure_mb=_f(row.get("pressure_mb", "")),
                air_temp_c=_f(row.get("air_temp_c", "")),
                water_temp_c=_f(row.get("water_temp_c", "")),
                wave_height_m=_f(row.get("wave_height_m", "")),
            )


def buoy_rows(path: Path | str | None = None) -> Iterator[dict]:
    for o in read_buoy(path):
        yield o.__dict__.copy()
