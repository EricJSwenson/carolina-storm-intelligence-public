# Live data & automated ingestion plan

This describes how live data flows into the platform today, and the path to a
production-grade automated pipeline. The platform's core purpose remains
**trustworthy evaluation against ground truth**; the live layer is an additive
intelligence surface, and the live model output is always an *estimate shown
beside the official NHC forecast*, never a replacement for it.

## What's built now

| Source | API (free, no key) | Module | Surface |
|---|---|---|---|
| Local forecast | api.weather.gov | `live/feeds.get_forecast` | Storm Center |
| Tides | tidesandcurrents.noaa.gov | `live/feeds.get_tides` | Storm Center |
| Buoy swell | ndbc.noaa.gov | `live/feeds.get_swell` | Storm Center |
| Active storms | nhc.noaa.gov/CurrentStorms.json | `live/feeds.get_active_storms` | Storm Center |
| Geolocation | ipapi.co / OSM Nominatim | `live/feeds.detect_location/geocode` | Storm Center |
| Live estimate | trained landfall model | `live/predict.estimate_landfall` | beside NHC bulletin |

Two delivery modes are wired:
- **On-demand** — the Streamlit Storm Center calls the feeds server-side (no
  browser CORS issues), cached for 10 minutes to be gentle on the APIs.
- **Scheduled snapshot** — `.github/workflows/live-data.yml` runs
  `scripts/fetch_live_snapshot.py` every 6 hours and commits
  `data/live/active_storms.json`, so the repo always carries recent live data.

## Why the live model output is framed as an estimate

Operational hurricane forecasting is done by the National Hurricane Center with
physics-based numerical models (GFS, ECMWF, HAFS) on supercomputers. A portfolio
ML model cannot and should not claim to replace that. So the platform:
- shows the **official NHC bulletin/advisory link** as the authoritative source, and
- shows the model's **landfall-category estimate** next to it, clearly labeled
  "model estimate — not an official forecast."

## Production-grade automated ingestion (roadmap)

**Phase 1 — scheduled pulls (done, lightweight).** GitHub Actions cron → fetch →
commit snapshot. Good enough for a portfolio and fully automated.

**Phase 2 — warehouse-backed.** Replace the committed JSON with writes to the
medallion warehouse: a scheduled Databricks job (or a cloud function) lands raw
active-storm/forecast/buoy pulls in **bronze**, conforms them in **silver**
(join live tracks to the same schema as HURDAT2), and materializes **gold**
features. The Storm Center then reads gold instead of calling APIs live.

**Phase 3 — streaming + alerting.** For active storms, poll NHC every advisory
cycle (~6 h) via a scheduled job; on each new advisory, run the landfall model,
log the estimate to MLflow, and compare it to the eventual outcome once the storm
makes landfall — feeding the **same evaluation harness** that grades the RAG
system. That closes the loop: the predictor is continuously *evaluated*

**Phase 4 — model on real history.** Retrain the landfall model on the full
HURDAT2 record (all U.S. landfalls, not just NC), with the leakage-safe
year-split, and track its accuracy over seasons in MLflow.

## Reliability notes

- Every feed degrades gracefully (returns a status string; the panel shows
  "unavailable" rather than erroring).
- Live calls can't be exercised in an offline CI sandbox; they're verified by
  running the Storm Center against the real endpoints.
- NWS requires a descriptive `User-Agent` (set in `live/feeds`).
- During quiet season, "no active storms" is the normal, correct state.
