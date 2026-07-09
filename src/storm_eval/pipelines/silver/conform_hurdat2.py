"""Silver: conform track points and join narratives to storm tracks.

Derives Saffir-Simpson per fix, then performs the keystone join of the project:
each Storm Events narrative is linked to its HURDAT2 track (and, downstream, to
buoy observations) so an answer's claims can be checked against measured truth.
"""

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import IntegerType

from storm_eval.ingestion.hurdat2 import saffir_simpson_category

ss_udf = F.udf(saffir_simpson_category, IntegerType())


def conform(spark: SparkSession) -> None:
    points = spark.table("storm.bronze_hurdat_points").withColumn(
        "saffir_simpson", ss_udf(F.col("max_wind_kt"))
    )
    points.write.format("delta").mode("overwrite").option("overwriteSchema", "true") \
        .saveAsTable("storm.silver_track_points")

    # Link narratives to storms by name+year (illustrative; production also uses
    # spatial/temporal proximity of the event to the track).
    events = spark.table("storm.bronze_storm_events")
    storms = spark.table("storm.bronze_hurdat_storms")
    linked = (events.join(
        storms,
        (F.upper(events.EVENT_NARRATIVE).contains(F.col("name"))) & (events.YEAR == storms.year),
        "left",
    ).select(events["*"], storms.storm_id))
    linked.write.format("delta").mode("overwrite").option("overwriteSchema", "true") \
        .saveAsTable("storm.silver_event_narratives")


if __name__ == "__main__":
    conform(SparkSession.builder.getOrCreate())
