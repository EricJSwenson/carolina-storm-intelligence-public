# Databricks notebook source
# MAGIC %md
# MAGIC # 04 · Index + evaluate
# MAGIC Chunk + embed the gold corpus, then benchmark each registered model and
# MAGIC log metrics to MLflow. Promotion is gated on hallucination rate.

# COMMAND ----------
from storm_eval.db import connect, init_schema
from storm_eval.evaluation.benchmark import benchmark_all
from storm_eval.experiments.runner import build_store
from storm_eval.pipelines.local import gold_corpus, run_medallion

con = connect(); init_schema(con); run_medallion(con)
store = build_store(gold_corpus(con), chunk_size=320, embedding_backend="local")
runs = benchmark_all(["mock-grounded", "mock-naive"], store, warehouse=con)
print(runs)

# COMMAND ----------
from storm_eval.evaluation.ab_test import compare_runs
print(compare_runs(con, "reward", "mock-naive", "mock-grounded").summary())
