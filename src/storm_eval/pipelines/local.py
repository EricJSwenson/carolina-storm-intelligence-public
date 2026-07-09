"""In-process medallion pipeline for local development and CI.

Mirrors the PySpark jobs in ``pipelines/{bronze,silver,gold}`` but runs entirely
in-process on the bundled samples -- no Spark, no cluster. This is what
``make demo`` executes. The production path is the Databricks notebooks.

Layers:
  bronze  -> land raw HURDAT2 / Storm Events / NWS / buoy into DuckDB
  silver  -> conform track points (+ Saffir-Simpson), link narratives to storms
  gold    -> storm_truth ground-truth table + the retrieval corpus + chunks
"""

from __future__ import annotations

from typing import List, Tuple

from storm_eval.db import connect, init_schema
from storm_eval.ingestion import ndbc_buoy, nws_api, storm_events
from storm_eval.ingestion.hurdat2 import read_hurdat2, saffir_simpson_category
from storm_eval.config import SAMPLES_DIR

# NC continental landfall bounding box (approx): the spatial filter that picks
# the Carolina landfall fix out of a storm's full Atlantic track.
NC_BOX = dict(lat_min=33.5, lat_max=36.8, lon_min=-78.7, lon_max=-75.3)


def build_bronze(con) -> None:
    storms, points = [], []
    for storm, pts in read_hurdat2(SAMPLES_DIR / "hurdat2_sample.txt"):
        storms.append((storm.storm_id, storm.basin, storm.cyclone_number,
                       storm.year, storm.name))
        for p in pts:
            points.append((p.storm_id, p.obs_time, p.record_id, p.status,
                           p.lat, p.lon, p.max_wind_kt, p.min_pressure_mb))
    con.execute("CREATE OR REPLACE TABLE bronze_hurdat_storms "
                "(storm_id VARCHAR, basin VARCHAR, cyclone_number INT, year INT, name VARCHAR)")
    con.executemany("INSERT INTO bronze_hurdat_storms VALUES (?,?,?,?,?)", storms)
    con.execute("CREATE OR REPLACE TABLE bronze_hurdat_points "
                "(storm_id VARCHAR, obs_time TIMESTAMP, record_id VARCHAR, status VARCHAR, "
                "lat DOUBLE, lon DOUBLE, max_wind_kt INT, min_pressure_mb INT)")
    con.executemany("INSERT INTO bronze_hurdat_points VALUES (?,?,?,?,?,?,?,?)", points)

    events = [(e.event_id, e.storm_id, e.state, e.year, e.event_type, e.document)
              for e in storm_events.read_storm_events()]
    con.execute("CREATE OR REPLACE TABLE bronze_storm_events "
                "(event_id VARCHAR, storm_id VARCHAR, state VARCHAR, year INT, "
                "event_type VARCHAR, document VARCHAR)")
    con.executemany("INSERT INTO bronze_storm_events VALUES (?,?,?,?,?,?)", events)

    products = [(p.product_id, p.product_type, p.office, p.issuance_time, p.document)
                for p in nws_api.read_text_products()]
    con.execute("CREATE OR REPLACE TABLE bronze_nws_products "
                "(product_id VARCHAR, product_type VARCHAR, office VARCHAR, "
                "issuance_time VARCHAR, document VARCHAR)")
    con.executemany("INSERT INTO bronze_nws_products VALUES (?,?,?,?,?)", products)

    buoy = [(o.station, o.obs_time, o.wind_speed_ms, o.pressure_mb, o.wave_height_m)
            for o in ndbc_buoy.read_buoy()]
    con.execute("CREATE OR REPLACE TABLE bronze_buoy "
                "(station VARCHAR, obs_time VARCHAR, wind_speed_ms DOUBLE, "
                "pressure_mb DOUBLE, wave_height_m DOUBLE)")
    con.executemany("INSERT INTO bronze_buoy VALUES (?,?,?,?,?)", buoy)


def build_silver(con) -> None:
    # Conform track points and derive Saffir-Simpson per fix.
    con.execute("CREATE OR REPLACE TABLE silver_track_points AS SELECT * FROM bronze_hurdat_points")
    con.execute("ALTER TABLE silver_track_points ADD COLUMN saffir_simpson INT")
    rows = con.execute(
        "SELECT rowid, max_wind_kt FROM silver_track_points"
    ).fetchall()
    for rid, wind in rows:
        con.execute("UPDATE silver_track_points SET saffir_simpson = ? WHERE rowid = ?",
                    [saffir_simpson_category(wind), rid])


def build_gold(con) -> None:
    # storm_truth: pick the NC landfall fix (record_id='L' within the NC box).
    con.execute("DELETE FROM storm_truth")
    con.execute(
        """
        INSERT INTO storm_truth
        WITH landfall AS (
            SELECT p.storm_id,
                   p.saffir_simpson AS landfall_category,
                   ROW_NUMBER() OVER (PARTITION BY p.storm_id ORDER BY p.obs_time) AS rn
            FROM silver_track_points p
            WHERE p.record_id = 'L'
              AND p.lat BETWEEN ? AND ? AND p.lon BETWEEN ? AND ?
        ),
        peak AS (
            SELECT storm_id, MAX(max_wind_kt) AS peak_wind_kt, MIN(min_pressure_mb) AS min_pressure_mb
            FROM silver_track_points GROUP BY storm_id
        )
        SELECT s.storm_id, s.name, s.year,
               lf.landfall_category, pk.peak_wind_kt, pk.min_pressure_mb
        FROM bronze_hurdat_storms s
        LEFT JOIN landfall lf ON lf.storm_id = s.storm_id AND lf.rn = 1
        LEFT JOIN peak pk ON pk.storm_id = s.storm_id
        """,
        [NC_BOX["lat_min"], NC_BOX["lat_max"], NC_BOX["lon_min"], NC_BOX["lon_max"]],
    )


def gold_corpus(con) -> List[Tuple[str, str]]:
    """The retrieval corpus: storm-event narratives + NWS products."""
    docs = con.execute(
        "SELECT event_id, document FROM bronze_storm_events "
        "UNION ALL SELECT product_id, document FROM bronze_nws_products"
    ).fetchall()
    return [(str(d[0]), d[1]) for d in docs if d[1]]


def run_medallion(con=None):
    con = con or connect()
    init_schema(con)
    build_bronze(con)
    build_silver(con)
    build_gold(con)
    return con
