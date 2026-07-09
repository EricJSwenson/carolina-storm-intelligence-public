"""End-to-end local demo: ingest -> medallion -> RAG -> evaluate -> A/B -> RLHF.

Runs the entire platform offline on the bundled Carolina-storm samples, then
prints a summary. After this, the warehouse and vector store are populated so
the dashboard and the FastAPI service have real data to serve.

    python scripts/run_local_pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from storm_eval.db import connect, init_schema
from storm_eval.evaluation.ab_test import compare_runs
from storm_eval.evaluation.benchmark import load_eval_set, run_benchmark
from storm_eval.experiments.runner import build_store
from storm_eval.pipelines.local import gold_corpus, run_medallion
from storm_eval.preference.collect_pairs import build_preference_pairs
from storm_eval.preference.rlhf_sim import reward_pool_from_evals, simulate

MODELS = ["mock-grounded", "mock-naive"]


def main() -> None:
    con = connect()
    init_schema(con)

    print("1) Medallion pipeline (bronze -> silver -> gold)")
    run_medallion(con)
    truth = con.execute("SELECT name, year, landfall_category, peak_wind_kt FROM storm_truth "
                        "ORDER BY year").fetchall()
    for name, year, cat, peak in truth:
        print(f"   ground truth: {name} ({year}) -> NC landfall Cat {cat}, peak {peak} kt")

    print("\n2) Build vector store from gold corpus")
    corpus = gold_corpus(con)
    store = build_store(corpus, chunk_size=320, embedding_backend="local")
    store.persist()
    print(f"   indexed {len(store.ids)} chunks from {len(corpus)} documents")

    print("\n3) Benchmark models")
    cases = load_eval_set()
    for m in MODELS:
        run_benchmark(m, store, con, cases)
    for m in MODELS:
        row = con.execute(
            "SELECT AVG(groundedness), AVG(answer_relevance), "
            "AVG(CASE WHEN hallucinated THEN 1.0 ELSE 0.0 END) "
            "FROM evaluations WHERE model=? AND hallucinated IS NOT NULL", [m]
        ).fetchone()
        g_all = con.execute("SELECT AVG(groundedness) FROM evaluations WHERE model=?", [m]).fetchone()[0]
        print(f"   {m:<14} groundedness={g_all:.3f}  "
              f"hallucination_rate={row[2]:.2f} (on checked questions)")

    print("\n4) Statistical A/B test")
    print("   note: embedding-groundedness rates both models ~equally because the")
    print("   naive answer is lexically near-identical to the truth -- which is why")
    print("   the structured HURDAT2 check above is the metric that actually catches it.")
    ab = compare_runs(con, "reward", "mock-naive", "mock-grounded")
    print("   " + ab.summary())

    print("\n5) Build preference pairs (RLHF dataset)")
    n_pairs = build_preference_pairs(con)
    print(f"   wrote {n_pairs} chosen/rejected pairs to preference_pairs")

    print("\n6) Simulate RLHF-style best-of-N improvement")
    pool = reward_pool_from_evals(con)
    for it in simulate(pool):
        print(f"   best-of-{it.n_candidates:<2} mean reward = {it.mean_reward:+.3f}")

    print("\n7) Train + evaluate the landfall-intensity model")
    from storm_eval.forecasting.dataset import generate_dataset
    from storm_eval.forecasting.evaluate import evaluate as fc_evaluate, persist_predictions
    from storm_eval.forecasting.features import build_training_table
    from storm_eval.forecasting.model import save, train as fc_train
    storms = generate_dataset(n_storms=500)
    report, _ = fc_evaluate(storms)
    print(f"   held-out: exact={report.accuracy:.3f}  within-1={report.within_one:.3f}  "
          f"macroF1={report.macro_f1:.3f}  (test storms={report.n_test})")
    Xf, yf, _ = build_training_table(storms)
    final = fc_train(Xf, yf)
    save(final)
    persist_predictions(con, storms, final)

    print("\nDone. Warehouse: data/warehouse.duckdb | Vector store: data/vectorstore/")
    print("Next: `python scripts/build_weather_page.py` then open dashboards/weather_intelligence.html")


if __name__ == "__main__":
    main()
