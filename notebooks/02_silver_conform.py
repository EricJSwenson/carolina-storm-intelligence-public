# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Silver conform
# MAGIC Derive Saffir-Simpson per track fix and link narratives to storm tracks.

# COMMAND ----------
from storm_eval.pipelines.silver import conform_hurdat2
conform_hurdat2.conform(spark)

# COMMAND ----------
display(spark.table("storm.silver_track_points")
        .select("storm_id", "obs_time", "status", "max_wind_kt", "saffir_simpson").limit(20))
