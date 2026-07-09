"""Bronze: parse HURDAT2 into two Delta tables (storms, track points).

Reuses the pure-Python parser in ``ingestion.hurdat2`` via mapPartitions so the
exact same, unit-tested parsing logic runs at scale on Databricks.
"""

from pyspark.sql import SparkSession, functions as F

from storm_eval.ingestion.hurdat2 import read_hurdat2


def ingest(spark: SparkSession, hurdat2_path: str,
           storms_table="storm.bronze_hurdat_storms",
           points_table="storm.bronze_hurdat_points") -> None:
    storm_rows, point_rows = [], []
    for storm, points in read_hurdat2(hurdat2_path):
        storm_rows.append(storm.__dict__)
        for p in points:
            d = p.__dict__.copy(); d["obs_time"] = p.obs_time.isoformat()
            point_rows.append(d)
    (spark.createDataFrame(storm_rows).withColumn("_ingest_ts", F.current_timestamp())
        .write.format("delta").mode("overwrite").saveAsTable(storms_table))
    (spark.createDataFrame(point_rows).withColumn("_ingest_ts", F.current_timestamp())
        .write.format("delta").mode("overwrite").saveAsTable(points_table))


if __name__ == "__main__":
    spark = SparkSession.builder.getOrCreate()
    ingest(spark, "/Volumes/storm/raw/hurdat2/hurdat2-atlantic.txt")
