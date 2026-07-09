# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Gold storm-truth
# MAGIC Build the evaluator's ground-truth table: NC-landfall category, peak wind,
# MAGIC min pressure per storm.

# COMMAND ----------
from storm_eval.pipelines.gold import build_storm_truth
build_storm_truth.build(spark)

# COMMAND ----------
display(spark.table("storm.gold_storm_truth").orderBy("year"))
