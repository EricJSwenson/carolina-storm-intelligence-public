"""Ensure the warehouse + vector store + model exist before a dashboard renders.

Local runs populate these with `make demo`. On a hosted deploy (e.g. Streamlit
Community Cloud) the filesystem starts empty, so this builds them once on first
load -- the app is self-contained and needs no pre-seeded data.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def ensure_data() -> None:
    from storm_eval.config import settings
    from storm_eval.db import connect, init_schema

    if Path(settings.warehouse_path).exists():
        con = connect()
        init_schema(con)
        has_rows = con.execute("SELECT COUNT(*) FROM evaluations").fetchone()[0]
        con.close()
        if has_rows:
            return

    con = connect()
    init_schema(con)

    from storm_eval.evaluation.benchmark import load_eval_set, run_benchmark
    from storm_eval.experiments.runner import build_store
    from storm_eval.forecasting.dataset import generate_dataset
    from storm_eval.forecasting.evaluate import persist_predictions
    from storm_eval.forecasting.features import build_training_table
    from storm_eval.forecasting.model import train as fc_train
    from storm_eval.pipelines.local import gold_corpus, run_medallion
    from storm_eval.preference.collect_pairs import build_preference_pairs

    run_medallion(con)
    store = build_store(gold_corpus(con), chunk_size=320, embedding_backend="local")
    store.persist()
    cases = load_eval_set()
    for model in ("mock-grounded", "mock-naive"):
        run_benchmark(model, store, con, cases)
    build_preference_pairs(con)

    storms = generate_dataset(n_storms=500)
    X, y, _ = build_training_table(storms)
    persist_predictions(con, storms, fc_train(X, y))
    con.close()
