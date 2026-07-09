"""Parse NOAA HURDAT2 Atlantic best-track files into structured records.

HURDAT2 is a comma-delimited text file with two kinds of lines:

  * Header line  -> one per storm:   ``AL092021, IDA, 40,``
                    basin+number+year, name, and the count of track
                    rows that follow.
  * Data line    -> one per best-track fix (mostly 6-hourly, plus
                    asynoptic landfall / intensity-peak entries):
                    ``20210829, 1655, L, HU, 29.1N, 90.2W, 130, 931, ...``

This module turns that into two flat record types -- ``Storm`` (one row per
cyclone) and ``TrackPoint`` (one row per fix) -- ready to load into Delta /
SQL tables in the bronze layer.

Format reference: NHC, "The revised Atlantic hurricane database (HURDAT2)".
The radius-of-maximum-wind column is only populated from the 2021 season on.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Iterator, Optional

# Values HURDAT2 uses to mean "missing".
_MISSING = {"-99", "-999", ""}

# Status codes, handy for filters and human-readable silver-layer columns.
STATUS_LABELS = {
    "TD": "tropical depression",
    "TS": "tropical storm",
    "HU": "hurricane",
    "EX": "extratropical cyclone",
    "SD": "subtropical depression",
    "SS": "subtropical storm",
    "LO": "low",
    "WV": "tropical wave",
    "DB": "disturbance",
}


@dataclass
class Storm:
    storm_id: str          # ATCF-style id, e.g. "AL092021" (primary key)
    basin: str             # "AL"
    cyclone_number: int    # 9
    year: int              # 2021
    name: str              # "IDA" or "UNNAMED"
    n_track_points: int    # declared row count from the header


@dataclass
class TrackPoint:
    storm_id: str                    # foreign key -> Storm.storm_id
    obs_time: datetime               # UTC
    record_id: Optional[str]         # L, P, I, ... or None at synoptic times
    status: str                      # TD / TS / HU / ...
    lat: float                       # signed decimal degrees (N +, S -)
    lon: float                       # signed decimal degrees (E +, W -)
    max_wind_kt: Optional[int]
    min_pressure_mb: Optional[int]
    r34_ne: Optional[int]; r34_se: Optional[int]; r34_sw: Optional[int]; r34_nw: Optional[int]
    r50_ne: Optional[int]; r50_se: Optional[int]; r50_sw: Optional[int]; r50_nw: Optional[int]
    r64_ne: Optional[int]; r64_se: Optional[int]; r64_sw: Optional[int]; r64_nw: Optional[int]
    rmw_nm: Optional[int]            # radius of max wind; None before 2021


def _to_int(token: str) -> Optional[int]:
    token = token.strip()
    return None if token in _MISSING else int(token)


def _parse_coord(token: str) -> float:
    """'29.1N' -> 29.1 ; '90.2W' -> -90.2."""
    token = token.strip()
    value, hemi = float(token[:-1]), token[-1].upper()
    return -value if hemi in ("S", "W") else value


def saffir_simpson_category(max_wind_kt: Optional[int]) -> Optional[int]:
    """Saffir-Simpson hurricane category (1-5) from 1-min sustained wind.

    Returns 0 for sub-hurricane systems and None when wind is missing.
    Used in the silver layer to build ground truth like "was this storm a
    Category 1 at landfall?" that the evaluator checks model answers against.
    """
    if max_wind_kt is None:
        return None
    thresholds = ((137, 5), (113, 4), (96, 3), (83, 2), (64, 1))
    for floor, category in thresholds:
        if max_wind_kt >= floor:
            return category
    return 0


def _parse_header(line: str) -> Storm:
    code, name, count, *_ = (f.strip() for f in line.split(","))
    return Storm(
        storm_id=code,
        basin=code[:2],
        cyclone_number=int(code[2:4]),
        year=int(code[4:8]),
        name=name or "UNNAMED",
        n_track_points=int(count),
    )


def _parse_data_line(storm_id: str, line: str) -> TrackPoint:
    f = [t.strip() for t in line.split(",")]
    obs_time = datetime.strptime(f[0] + f[1], "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
    r = [_to_int(x) for x in f[8:20]]               # 12 wind-radii columns
    return TrackPoint(
        storm_id=storm_id,
        obs_time=obs_time,
        record_id=f[2] or None,
        status=f[3],
        lat=_parse_coord(f[4]),
        lon=_parse_coord(f[5]),
        max_wind_kt=_to_int(f[6]),
        min_pressure_mb=_to_int(f[7]),
        r34_ne=r[0], r34_se=r[1], r34_sw=r[2], r34_nw=r[3],
        r50_ne=r[4], r50_se=r[5], r50_sw=r[6], r50_nw=r[7],
        r64_ne=r[8], r64_se=r[9], r64_sw=r[10], r64_nw=r[11],
        rmw_nm=_to_int(f[20]) if len(f) > 20 else None,   # 2021+ only
    )


def read_hurdat2(path: str) -> Iterator[tuple[Storm, list[TrackPoint]]]:
    """Stream ``(storm, track_points)`` pairs from a HURDAT2 file.

    Memory-safe for the full Atlantic file: each header declares how many
    data rows follow, so we consume exactly that many before the next header
    -- no need to guess which line is which.
    """
    with open(path, "r", encoding="utf-8") as fh:
        lines = (ln for ln in (raw.rstrip("\n") for raw in fh) if ln.strip())
        for header in lines:
            storm = _parse_header(header)
            points = [
                _parse_data_line(storm.storm_id, next(lines))
                for _ in range(storm.n_track_points)
            ]
            yield storm, points


def storm_rows(path: str) -> Iterator[dict]:
    """Flat dict rows for the ``dim_storm`` table."""
    for storm, _ in read_hurdat2(path):
        yield asdict(storm)


def track_rows(path: str) -> Iterator[dict]:
    """Flat dict rows for the ``fct_track_point`` fact table.

    ``obs_time`` is emitted as an ISO-8601 string so it loads cleanly into
    Spark / SQL regardless of session timezone.
    """
    for storm, points in read_hurdat2(path):
        for p in points:
            row = asdict(p)
            row["obs_time"] = p.obs_time.isoformat()
            row["saffir_simpson"] = saffir_simpson_category(p.max_wind_kt)
            yield row


if __name__ == "__main__":
    import sys

    src = sys.argv[1]
    n_storms = n_points = 0
    for storm, points in read_hurdat2(src):
        n_storms += 1
        n_points += len(points)
    print(f"parsed {n_storms:,} storms / {n_points:,} track points from {src}")
