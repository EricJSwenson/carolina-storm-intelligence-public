"""Gold: build the storm-truth table and the retrieval corpus (Databricks).

storm_truth is the evaluator's source of ground truth: the NC-landfall Saffir-
Simpson category, peak winds, and minimum pressure per storm. The corpus table
feeds the embedding/index job.
"""

from pyspark.sql import SparkSession, functions as F, Window

NC_BOX = (33.5, 36.8, -78.7, -75.3)  # lat_min, lat_max, lon_min, lon_max


def build(spark: SparkSession) -> None:
    pts = spark.table("storm.silver_track_points")
    lat_min, lat_max, lon_min, lon_max = NC_BOX
    landfall = (pts.filter(
        (F.col("record_id") == "L")
        & F.col("lat").between(lat_min, lat_max)
        & F.col("lon").between(lon_min, lon_max))
        .withColumn("rn", F.row_number().over(
            Window.partitionBy("storm_id").orderBy("obs_time")))
        .filter(F.col("rn") == 1)
        .select("storm_id", F.col("saffir_simpson").alias("landfall_category")))
    peak = pts.groupBy("storm_id").agg(
        F.max("max_wind_kt").alias("peak_wind_kt"),
        F.min("min_pressure_mb").alias("min_pressure_mb"))
    storms = spark.table("storm.bronze_hurdat_storms")
    truth = storms.join(landfall, "storm_id", "left").join(peak, "storm_id", "left")
    truth.write.format("delta").mode("overwrite").saveAsTable("storm.gold_storm_truth")


if __name__ == "__main__":
    build(SparkSession.builder.getOrCreate())
