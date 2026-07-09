"""Model Evaluation — RAG leaderboard + hallucination + A/B + landfall model."""
from __future__ import annotations

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

st.set_page_config(page_title="Model Evaluation", page_icon="📊", layout="wide")
st.title("Model Evaluation")

if not WAREHOUSE.exists():
    st.warning("Run `make demo` first.")
    st.stop()

con = duckdb.connect(str(WAREHOUSE), read_only=True)


@st.cache_data
def q(sql):
    return con.execute(sql).fetchdf()


# ---------------- RAG models ----------------
st.header("RAG systems")
latest = q("""
    WITH r AS (SELECT *, DENSE_RANK() OVER (PARTITION BY model ORDER BY run_ts DESC) rk
               FROM evaluations)
    SELECT * FROM r WHERE rk = 1
""")
board = (latest.assign(hallu=latest["hallucinated"])
         .groupby("model")
         .agg(questions=("question_id", "count"),
              groundedness=("groundedness", "mean"),
              answer_relevance=("answer_relevance", "mean"),
              reward=("reward", "mean"),
              hallucination_rate=("hallu", lambda s: s.dropna().mean() if s.dropna().size else 0.0),
              latency_ms=("latency_ms", "mean"))
         .reset_index().sort_values("reward", ascending=False))

st.dataframe(board.style.format({
    "groundedness": "{:.3f}", "answer_relevance": "{:.3f}", "reward": "{:+.3f}",
    "hallucination_rate": "{:.0%}", "latency_ms": "{:.2f}"}), use_container_width=True)

a, b = st.columns(2)
a.plotly_chart(px.bar(board, x="model", y="reward", color="model", range_y=[0, 1.05],
                      title="Reward by model"), use_container_width=True)
b.plotly_chart(px.bar(board, x="model", y="hallucination_rate", color="model", range_y=[0, 1.05],
                      title="Hallucination rate (checked questions)"), use_container_width=True)

st.subheader("A/B test (reward)")
try:
    from storm_eval.evaluation.ab_test import compare_runs

    models = board["model"].tolist()
    if len(models) >= 2:
        ab = compare_runs(con, "reward", models[-1], models[0])
        verdict = "✅ significant" if ab.significant else "⚪ not significant"
        st.metric(f"{ab.model_b} − {ab.model_a}", f"{ab.diff:+.3f}",
                  help=f"95% CI [{ab.ci_low:+.3f}, {ab.ci_high:+.3f}], p={ab.p_value:.4f}")
        st.write(f"**{verdict}** — {ab.summary()}")
except Exception as exc:  # noqa: BLE001
    st.info(f"A/B unavailable: {exc}")

with st.expander("Per-question detail"):
    pick = st.selectbox("Model", board["model"].tolist())
    st.dataframe(latest[latest["model"] == pick][
        ["question_id", "question", "answer", "groundedness", "reward", "hallucinated"]],
        use_container_width=True)

# ---------------- Landfall-intensity model ----------------
st.divider()
st.header("Landfall-intensity model")
try:
    fc = q("SELECT * FROM forecast_evaluations")
except Exception:
    fc = pd.DataFrame()

if fc.empty:
    st.info("Run `python scripts/train_forecaster.py` to populate forecast metrics.")
else:
    acc = (fc["predicted_category"] == fc["actual_category"]).mean()
    within1 = (fc["predicted_category"].sub(fc["actual_category"]).abs() <= 1).mean()
    m1, m2, m3 = st.columns(3)
    m1.metric("Exact accuracy", f"{acc:.0%}")
    m2.metric("Within 1 category", f"{within1:.0%}")
    m3.metric("Storms scored", f"{len(fc)}")

    cm = (fc.groupby(["actual_category", "predicted_category"]).size()
          .reset_index(name="n"))
    fig = px.density_heatmap(cm, x="predicted_category", y="actual_category", z="n",
                             color_continuous_scale="Blues", title="Confusion matrix",
                             labels={"predicted_category": "predicted", "actual_category": "actual"})
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Predicts NC landfall Saffir-Simpson category from a storm's pre-landfall track. "
               "Held-out by year (no leakage). Demo trained on synthetic best-tracks.")
