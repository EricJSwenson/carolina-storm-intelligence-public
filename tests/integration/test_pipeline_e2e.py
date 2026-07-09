"""End-to-end: medallion -> index -> benchmark on the bundled samples."""
import duckdb
import pytest

from storm_eval.db import init_schema
from storm_eval.evaluation.benchmark import load_eval_set, run_benchmark
from storm_eval.experiments.runner import build_store
from storm_eval.pipelines.local import gold_corpus, run_medallion


@pytest.fixture
def con(tmp_path):
    c = duckdb.connect(str(tmp_path / "wh.duckdb"))
    init_schema(c)
    run_medallion(c)
    return c


def test_storm_truth_built(con):
    rows = con.execute("SELECT storm_id, landfall_category FROM storm_truth ORDER BY storm_id").fetchall()
    truth = dict(rows)
    assert truth["AL062018"] == 1 and truth["AL052019"] == 1 and truth["AL092011"] == 1


def test_grounded_beats_naive_on_hallucination(con):
    store = build_store(gold_corpus(con), chunk_size=320, embedding_backend="local")
    cases = load_eval_set()
    run_benchmark("mock-grounded", store, con, cases)
    run_benchmark("mock-naive", store, con, cases)

    def hrate(model):
        r = con.execute(
            "SELECT AVG(CASE WHEN hallucinated THEN 1.0 ELSE 0.0 END) FROM evaluations "
            "WHERE model=? AND hallucinated IS NOT NULL", [model]).fetchone()[0]
        return r

    assert hrate("mock-grounded") == 0.0
    assert hrate("mock-naive") == 1.0
