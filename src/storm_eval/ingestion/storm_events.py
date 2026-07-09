"""Ingest the NOAA Storm Events Database.

The Storm Events Database is the platform's primary text corpus: 1.7M+ records
(1950-present), each with a forecaster-authored episode/event narrative. NOAA
publishes it as yearly bulk CSVs (``StormEvents_details-ftp_v1.0_dYYYY_*.csv``)
at https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/ .

In production these land in bronze via the Databricks job in
``notebooks/01_bronze_ingest.py``. For local development this module reads the
bundled sample CSV so the whole pipeline runs offline.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

from storm_eval.config import SAMPLES_DIR

BULK_BASE = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/"


@dataclass
class StormEvent:
    event_id: str
    episode_id: Optional[str]
    storm_id: Optional[str]   # link to HURDAT2 storm (set in silver join)
    state: str
    year: int
    event_type: str           # Hurricane (Typhoon), Storm Surge/Tide, ...
    begin_datetime: str
    cz_name: str              # county / zone name
    damage_property: Optional[str]
    deaths_direct: Optional[int]
    episode_narrative: str    # forecaster-written, corpus text
    event_narrative: str      # forecaster-written, corpus text

    @property
    def document(self) -> str:
        """The retrievable text unit for this event."""
        parts = [
            f"{self.event_type} in {self.cz_name}, {self.state} ({self.begin_datetime}).",
            self.episode_narrative.strip(),
            self.event_narrative.strip(),
        ]
        return "\n".join(p for p in parts if p)


def _to_int(v: str) -> Optional[int]:
    v = (v or "").strip()
    return int(v) if v.isdigit() else None


def read_storm_events(path: Path | str | None = None) -> Iterator[StormEvent]:
    """Stream StormEvent records from a Storm Events details CSV.

    Defaults to the bundled Carolina coastal-storm sample.
    """
    path = Path(path) if path else SAMPLES_DIR / "storm_events_sample.csv"
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            yield StormEvent(
                event_id=row["EVENT_ID"],
                episode_id=row.get("EPISODE_ID") or None,
                storm_id=row.get("STORM_ID") or None,
                state=row.get("STATE", ""),
                year=int(row["YEAR"]),
                event_type=row.get("EVENT_TYPE", ""),
                begin_datetime=row.get("BEGIN_DATE_TIME", ""),
                cz_name=row.get("CZ_NAME", ""),
                damage_property=row.get("DAMAGE_PROPERTY") or None,
                deaths_direct=_to_int(row.get("DEATHS_DIRECT", "")),
                episode_narrative=row.get("EPISODE_NARRATIVE", ""),
                event_narrative=row.get("EVENT_NARRATIVE", ""),
            )


def event_rows(path: Path | str | None = None) -> Iterator[dict]:
    """Flat dict rows for the bronze ``storm_events`` table."""
    for e in read_storm_events(path):
        row = e.__dict__.copy()
        row["document"] = e.document
        yield row
