"""CI gate: benchmark the candidate model and fail if it regresses.

Exits non-zero when the grounded model's hallucination rate exceeds
MAX_HALLUCINATION_RATE. This is the continuous-improvement guardrail: a prompt
or retrieval change that increases hallucinations cannot be merged.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from storm_eval.db import connect, init_schema
from storm_eval.evaluation.benchmark import load_eval_set, run_benchmark
from storm_eval.experiments.runner import build_store
from storm_eval.pipelines.local import gold_corpus, run_medallion

MODEL = os.getenv("CANDIDATE_MODEL", "mock-grounded")
THRESHOLD = float(os.getenv("MAX_HALLUCINATION_RATE", "0.10"))

con = connect()
init_schema(con)
run_medallion(con)
store = build_store(gold_corpus(con), chunk_size=320, embedding_backend="local")
run_benchmark(MODEL, store, con, load_eval_set())
rate = con.execute(
    "SELECT AVG(CASE WHEN hallucinated THEN 1.0 ELSE 0.0 END) FROM evaluations "
    "WHERE model=? AND hallucinated IS NOT NULL", [MODEL]).fetchone()[0] or 0.0

print(f"{MODEL} hallucination_rate={rate:.3f} (threshold {THRESHOLD:.3f})")
if rate > THRESHOLD:
    print("FAIL: hallucination rate regressed above threshold")
    sys.exit(1)
print("PASS")
