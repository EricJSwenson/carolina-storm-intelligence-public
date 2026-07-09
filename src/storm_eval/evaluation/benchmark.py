"""Benchmark a model over the evaluation set and persist per-question metrics.

For every (model, question) it runs the RAG pipeline, scores the answer on each
metric, applies the structured ground-truth check, and writes a row to the
``evaluations`` table. Runs are tagged with a ``run_id`` and logged to MLflow.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from storm_eval.config import SAMPLES_DIR
from storm_eval.db import connect, init_schema
from storm_eval.evaluation.ground_truth import hallucination_flag
from storm_eval.evaluation.judges import get_judge
from storm_eval.evaluation.metrics import (
    answer_relevance,
    context_relevance,
    cost_usd,
    faithfulness,
    groundedness,
)
from storm_eval.rag.pipeline import RAGPipeline
from storm_eval.rag.vectorstore import LocalVectorStore
from storm_eval.tracking.mlflow_utils import log_run


@dataclass
class EvalCase:
    question_id: str
    storm_id: Optional[str]
    question: str


def load_eval_set(path: Path | str | None = None) -> List[EvalCase]:
    path = Path(path) if path else SAMPLES_DIR / "eval_set.jsonl"
    cases = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            cases.append(EvalCase(d["question_id"], d.get("storm_id"), d["question"]))
    return cases


def _truth_lookup(con) -> Dict[str, dict]:
    rows = con.execute(
        "SELECT storm_id, landfall_category, peak_wind_kt FROM storm_truth"
    ).fetchall()
    return {r[0]: {"landfall_category": r[1], "peak_wind_kt": r[2]} for r in rows}


def run_benchmark(model: str, store: LocalVectorStore, con,
                  cases: List[EvalCase], judge_name: str = "heuristic",
                  top_k: int = 4, rerank: str = "lexical",
                  template: str = "rag_v1.jinja") -> str:
    """Run one model across the eval set; returns the run_id."""
    run_id = uuid.uuid4().hex[:12]
    run_ts = datetime.now(timezone.utc)
    pipeline = RAGPipeline(store, model=model, top_k=top_k, rerank_strategy=rerank)
    judge = get_judge(judge_name)
    truth = _truth_lookup(con)

    rows, per_q = [], []
    for case in cases:
        res = pipeline.answer(case.question)
        contexts = [h.text for h in res.hits]
        gnd = groundedness(res.answer, contexts)
        fth = faithfulness(res.answer, contexts)
        cr = context_relevance(case.question, contexts)
        ar = answer_relevance(case.question, res.answer)
        js = judge.score(case.question, res.answer, contexts)
        hl = hallucination_flag(res.answer, truth.get(case.storm_id, {})) if case.storm_id else None
        cost = cost_usd(res.model, res.prompt_tokens, res.completion_tokens)
        reward = gnd - (1.0 if hl else 0.0)
        rows.append((run_id, run_ts, res.model, case.question_id, case.storm_id,
                     case.question, res.answer, gnd, fth, cr, ar, js, hl,
                     res.latency_ms, cost, res.prompt_tokens, res.completion_tokens, reward))
        per_q.append({"groundedness": gnd, "faithfulness": fth, "context_relevance": cr,
                      "answer_relevance": ar, "judge": js, "latency_ms": res.latency_ms,
                      "cost_usd": cost, "hallucinated": hl})

    con.executemany(
        "INSERT INTO evaluations VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows
    )

    summary = _summarize(per_q)
    log_run(model=model, params={"top_k": top_k, "rerank": rerank, "template": template,
                                 "judge": judge_name}, metrics=summary, run_id=run_id)
    return run_id


def _summarize(per_q: List[dict]) -> Dict[str, float]:
    n = len(per_q) or 1
    checked = [r for r in per_q if r["hallucinated"] is not None]
    return {
        "mean_groundedness": sum(r["groundedness"] for r in per_q) / n,
        "mean_faithfulness": sum(r["faithfulness"] for r in per_q) / n,
        "mean_context_relevance": sum(r["context_relevance"] for r in per_q) / n,
        "mean_answer_relevance": sum(r["answer_relevance"] for r in per_q) / n,
        "mean_judge": sum(r["judge"] for r in per_q) / n,
        "hallucination_rate": (sum(1 for r in checked if r["hallucinated"]) / len(checked))
        if checked else 0.0,
        "p50_latency_ms": sorted(r["latency_ms"] for r in per_q)[len(per_q) // 2],
        "total_cost_usd": sum(r["cost_usd"] for r in per_q),
    }


def benchmark_all(models: List[str], store: LocalVectorStore,
                  warehouse=None, **kw) -> Dict[str, str]:
    """Benchmark several models; returns {model: run_id}."""
    con = warehouse or connect()
    init_schema(con)
    cases = load_eval_set()
    return {m: run_benchmark(m, store, con, cases, **kw) for m in models}
