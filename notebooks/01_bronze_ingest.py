# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Bronze ingest
# MAGIC Land raw NOAA sources into Delta. Storm Events + HURDAT2 are the core;
# MAGIC NWS products and NDBC buoys are appended incrementally.

# COMMAND ----------
from storm_eval.pipelines.bronze import ingest_storm_events, ingest_hurdat2

ingest_storm_events.ingest(spark, years=range(1996, 2027))
ingest_hurdat2.ingest(spark, "/Volumes/storm/raw/hurdat2/hurdat2-atlantic.txt")

# COMMAND ----------
display(spark.table("storm.bronze_hurdat_storms").limit(10))
