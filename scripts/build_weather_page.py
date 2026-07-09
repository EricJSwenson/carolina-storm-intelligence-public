"""Build the standalone Weather Intelligence landing page.

Uses the FULL NOAA HURDAT2 Atlantic best-track record (1851-2025) when available
-- auto-downloading it on first run -- and falls back to the bundled 3-storm
sample offline. Filters to storms that affected the Carolinas (the project's
focus), so the map shows real historical volume (Helene, Matthew, Floyd, Hazel,
Hugo, ... alongside Florence/Dorian/Irene) without rendering all 2,000+ storms.
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from storm_eval.forecasting.dataset import generate_dataset
from storm_eval.forecasting.evaluate import evaluate
from storm_eval.ingestion.hurdat2 import read_hurdat2, saffir_simpson_category
from storm_eval.config import DATA_DIR, SAMPLES_DIR

# Current full Atlantic HURDAT2 file (see https://www.nhc.noaa.gov/data/hurdat/).
HURDAT2_URL = "https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2025-02272026.txt"
FULL_PATH = DATA_DIR / "hurdat2_atlantic.txt"

# Carolina-affecting region: any track fix inside this box -> include the storm.
CAROLINA_BOX = (31.0, 37.8, -82.5, -73.0)   # lat_min, lat_max, lon_min, lon_max
# NC continental landfall box (tighter) for the "NC landfall" column.
NC_BOX = (33.5, 36.8, -78.7, -75.3)
MAX_STORMS = 90                              # most recent N Carolina-affecting storms
NOTABLE = {"HELENE", "FLORENCE", "DORIAN", "IRENE", "MATTHEW", "ISABEL", "FLOYD",
           "FRAN", "HAZEL", "HUGO", "ISAIAS", "OPHELIA", "BONNIE", "ARTHUR"}
CAT_COLORS = {0: "#5b8fb0", 1: "#22d3ee", 2: "#fde047", 3: "#fb923c", 4: "#f43f5e", 5: "#c026d3"}


def _in(box, lat, lon):
    return box[0] <= lat <= box[1] and box[2] <= lon <= box[3]


def _ensure_hurdat2() -> Path:
    """Return a path to the full HURDAT2 file, downloading it if needed; else the sample."""
    if FULL_PATH.exists():
        return FULL_PATH
    try:
        FULL_PATH.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(HURDAT2_URL, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) carolina-storm-intelligence"})
        print("downloading full HURDAT2 from NHC ...")
        with urllib.request.urlopen(req, timeout=60) as r:  # noqa: S310
            FULL_PATH.write_bytes(r.read())
        print(f"  saved {FULL_PATH} ({FULL_PATH.stat().st_size // 1024} KB)")
        return FULL_PATH
    except Exception as exc:  # noqa: BLE001 - offline / blocked: use the sample
        print(f"  download unavailable ({exc}); using bundled 3-storm sample")
        return SAMPLES_DIR / "hurdat2_sample.txt"


def _select_storms(path: Path):
    selected = []
    for storm, pts in read_hurdat2(path):
        if not any(_in(CAROLINA_BOX, p.lat, p.lon) for p in pts):
            continue
        selected.append((storm, pts))
    # Most-recent first, but always keep notable named storms.
    selected.sort(key=lambda sp: sp[0].year, reverse=True)
    notable = [sp for sp in selected if sp[0].name.upper() in NOTABLE]
    rest = [sp for sp in selected if sp[0].name.upper() not in NOTABLE]
    chosen = notable + rest[: max(0, MAX_STORMS - len(notable))]
    chosen.sort(key=lambda sp: sp[0].year, reverse=True)
    return chosen


def build_data() -> dict:
    path = _ensure_hurdat2()
    chosen = _select_storms(path)
    storms = []
    for storm, pts in chosen:
        track = [{
            "lat": round(p.lat, 1), "lon": round(p.lon, 1), "wind": p.max_wind_kt,
            "pres": p.min_pressure_mb, "cat": saffir_simpson_category(p.max_wind_kt),
            "time": p.obs_time.strftime("%Y-%m-%d %HZ"),
            "landfall": p.record_id == "L",
        } for p in pts if p.max_wind_kt is not None]
        if not track:
            continue
        nc_lf = next((t for t in track if t["landfall"] and _in(NC_BOX, t["lat"], t["lon"])), None)
        storms.append({
            "id": storm.storm_id, "name": storm.name.title(), "year": storm.year,
            "peak_wind": max(t["wind"] for t in track),
            "min_pres": min((t["pres"] for t in track if t["pres"] is not None), default=None),
            "peak_cat": max(t["cat"] for t in track),
            "landfall_cat": nc_lf["cat"] if nc_lf else None,
            "track": track,
        })

    report, _ = evaluate(generate_dataset(n_storms=500))
    model = {"accuracy": round(report.accuracy, 3), "within_one": round(report.within_one, 3),
             "macro_f1": round(report.macro_f1, 3)}

    qa = [
        {"q": "Where did Florence make landfall in North Carolina?"},
        {"q": "What were Dorian's threats near the Outer Banks?"},
        {"q": "What category was Irene at Cape Lookout?"},
    ]
    # Narrative corpus for the in-browser assistant (same docs the RAG pipeline uses).
    from storm_eval.ingestion import nws_api, storm_events
    corpus = [{"id": e.event_id, "text": e.document} for e in storm_events.read_storm_events()]
    corpus += [{"id": p.product_id, "text": p.document} for p in nws_api.read_text_products()]

    return {"storms": storms, "model": model, "qa": qa, "catColors": CAT_COLORS,
            "corpus": corpus, "defaultLoc": {"lat": 34.72, "lon": -76.73, "label": "Morehead City, NC"},
            "source": "NOAA HURDAT2 1851–2025" if path == FULL_PATH else "bundled sample (offline)"}


HTML = Path(__file__).resolve().parents[1] / "dashboards" / "weather_intelligence.html"


def main() -> None:
    data = build_data()
    template = (Path(__file__).resolve().parent / "_weather_template.html").read_text(encoding="utf-8")
    HTML.write_text(template.replace("__DATA__", json.dumps(data)), encoding="utf-8")
    print(f"wrote {HTML} ({HTML.stat().st_size // 1024} KB) with {len(data['storms'])} storms")


if __name__ == "__main__":
    main()
