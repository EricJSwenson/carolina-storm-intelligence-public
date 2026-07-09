"""Bronze: ingest the NOAA Storm Events bulk CSVs into a Delta table (Databricks).

Production counterpart to ``pipelines/local.build_bronze``. Reads the yearly
gzip CSVs straight from NCEI into a raw Delta table partitioned by year, with
ingest metadata. Schedule as a Databricks job; backfill by widening YEARS.
"""

from pyspark.sql import SparkSession, functions as F

NCEI_BASE = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles"


def ingest(spark: SparkSession, years, out_table: str = "storm.bronze_storm_events") -> None:
    paths = [f"/Volumes/storm/raw/storm_events/details_{y}.csv.gz" for y in years]
    df = (
        spark.read.option("header", True).option("inferSchema", True).csv(paths)
        .withColumn("_ingest_ts", F.current_timestamp())
        .withColumn("_source_file", F.input_file_name())
    )
    (df.write.format("delta").mode("overwrite")
       .option("overwriteSchema", "true").partitionBy("YEAR")
       .saveAsTable(out_table))


if __name__ == "__main__":
    spark = SparkSession.builder.getOrCreate()
    ingest(spark, years=range(1996, 2027))
