"""Experiment Tracking — configuration sweeps and metric movement across runs."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import duckdb
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from _bootstrap import ensure_data  # noqa: E402
ensure_data()
WAREHOUSE = ROOT / "data" / "warehouse.duckdb"
MLRUNS = ROOT / "data" / "mlruns" / "local_runs.jsonl"

st.set_page_config(page_title="Experiment Tracking", page_icon="🧪", layout="wide")
st.title("Experiment Tracking")
st.caption("Each run logs params (prompt, retrieval, embedding, chunk size) and metrics. "
           "Compare configurations and watch performance move.")

if not WAREHOUSE.exists():
    st.warning("Run `make demo` first.")
    st.stop()

con = duckdb.connect(str(WAREHOUSE), read_only=True)

# Runs from the warehouse (one row per run × model).
runs = con.execute("""
    SELECT run_id, model, MIN(run_ts) AS run_ts,
           AVG(groundedness) AS groundedness,
           AVG(answer_relevance) AS answer_relevance,
           AVG(reward) AS reward,
           AVG(latency_ms) AS latency_ms,
           COUNT(*) AS n
    FROM evaluations GROUP BY run_id, model ORDER BY run_ts
""").fetchdf()

st.subheader("All runs")
st.dataframe(runs.style.format({
    "groundedness": "{:.3f}", "answer_relevance": "{:.3f}",
    "reward": "{:+.3f}", "latency_ms": "{:.2f}"}), use_container_width=True)

st.subheader("Reward across runs")
runs2 = runs.copy()
runs2["run"] = range(1, len(runs2) + 1)
st.plotly_chart(px.line(runs2, x="run", y="reward", color="model", markers=True,
                        title="Reward by run"), use_container_width=True)

# Params logged to the MLflow JSONL fallback (prompt, retrieval, chunk size, embedding).
st.subheader("Logged configurations (MLflow)")
if MLRUNS.exists():
    recs = [json.loads(line) for line in MLRUNS.read_text().splitlines() if line.strip()]
    if recs:
        rows = []
        for r in recs:
            row = {"model": r["model"], "run_id": r["run_id"]}
            row.update(r.get("params", {}))
            row.update({k: round(v, 3) for k, v in r.get("metrics", {}).items()
                        if k in ("mean_groundedness", "hallucination_rate", "p50_latency_ms")})
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info("No MLflow runs logged yet.")
else:
    st.info("No MLflow run log found. Run an experiment: "
            "`python scripts/run_experiment.py chunk_size 160 320 640`")

st.caption("In production these runs live in the MLflow tracking server; the model registry "
           "promotes the winning configuration once it clears the eval-regression gate.")
