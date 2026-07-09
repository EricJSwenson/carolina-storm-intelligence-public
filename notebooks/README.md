# Databricks notebooks

Production orchestration (source format, importable into a Databricks Repo). The
medallion notebooks run on a cluster against Unity Catalog Delta tables; the
index/evaluate notebook reuses the same `storm_eval` package the local demo
uses, so logic is identical across environments.

| Notebook | Stage |
|---|---|
| `01_bronze_ingest` | Raw NOAA sources -> Delta |
| `02_silver_conform` | Saffir-Simpson + narrative/track join |
| `03_gold_truth` | Ground-truth + corpus tables |
| `04_index_and_evaluate` | Embed, benchmark, A/B, MLflow |

Wire them into a Databricks Workflow (Jobs) with a nightly schedule; the
eval-regression step mirrors `.github/workflows/eval-regression.yml`.
