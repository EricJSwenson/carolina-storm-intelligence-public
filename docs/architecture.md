# Architecture

## Layers

```
Sources            Ingestion          Medallion (Delta/DuckDB)        RAG + Eval            Serving
─────────          ─────────          ────────────────────────       ──────────            ───────
Storm Events  ─┐                      bronze: raw landed tables       retrieve (embed,      FastAPI
HURDAT2        ├─► ingestion/  ──────► silver: conformed track pts  ─► vector search,    ─► /query
NWS (KMHX)     │   parsers              + narrative↔track join          rerank)              /evaluate
NDBC buoys    ─┘                       gold:  storm_truth + corpus     generate (LLM)
                                                                       evaluate (metrics +   Streamlit
                                                                       HURDAT2 ground truth) dashboard
                                                                       │
                                                       MLflow ◄────────┘  A/B test, preference pairs
```

## Key design decisions

- **Swappable backends behind one interface.** Embeddings, vector store,
  reranker, and generator each have a `local`/`mock` offline implementation and
  a production adapter (OpenAI, Chroma/Pinecone, cross-encoder, Ollama). The
  experiments layer A/B tests exactly these swaps.
- **The same parser runs everywhere.** `ingestion/hurdat2.py` is pure Python and
  unit-tested; the PySpark bronze job calls it via `mapPartitions`, so parsing
  logic is identical local and at scale.
- **Structured ground truth is a first-class table.** `gold_storm_truth` is the
  evaluator's oracle; hallucinations are *verifiable*, not vibes.
- **Local = production, minus the cluster.** DuckDB stands in for Databricks SQL
  (same DDL), pandas stands in for Spark, deterministic mocks stand in for paid
  models — so the project runs and tests in CI with zero credentials.

## The keystone join (silver)

Each Storm Events narrative is linked to its HURDAT2 track (by name/year and, in
production, spatial-temporal proximity), and tracks are linked to NDBC buoy
observations. This three-way join is what lets the evaluator check a textual
claim against measured truth.

## Repository map

- `src/storm_eval/ingestion` — source parsers (HURDAT2, Storm Events, NWS, buoys)
- `src/storm_eval/pipelines` — `local.py` (runnable demo) + PySpark bronze/silver/gold
- `src/storm_eval/rag` — embeddings, chunking, vector store, reranker, retriever, generator, pipeline
- `src/storm_eval/evaluation` — metrics, judges, ground truth, benchmark, A/B
- `src/storm_eval/forecasting` — landfall-intensity model: synthetic+real dataset, features, model, leakage-safe eval
- `src/storm_eval/experiments` — config sweeps
- `src/storm_eval/preference` — preference pairs + RLHF-style simulation
- `src/storm_eval/serving` — FastAPI app
- `src/storm_eval/tracking` — MLflow utils + registry
- `sql/`, `dbt/`, `notebooks/`, `dashboards/`, `docker/`, `.github/workflows/`
