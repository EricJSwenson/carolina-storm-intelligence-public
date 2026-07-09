"""Storm Intelligence — technical dashboards (Streamlit multipage app).

    streamlit run dashboards/Home.py

Pages:
  1 · Model Evaluation     RAG leaderboard, hallucination, A/B + the landfall model
  2 · Experiment Tracking  config sweeps and performance over runs

The recruiter-facing Weather Intelligence view is the standalone
dashboards/weather_intelligence.html (open it directly in a browser).
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from _bootstrap import ensure_data  # noqa: E402
ensure_data()

st.set_page_config(page_title="Storm Intelligence", page_icon="🌀", layout="wide")
st.title("Storm Intelligence — Platform Console")
st.caption("RAG over NOAA storm narratives + a landfall-intensity model, all graded against HURDAT2 ground truth.")

warehouse = ROOT / "data" / "warehouse.duckdb"
if not warehouse.exists():
    st.warning("No warehouse yet. Run `make demo` and `python scripts/train_forecaster.py` first.")
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("What this is", "Weather intelligence + LLM eval")
c2.metric("Ground truth", "NOAA HURDAT2")
c3.metric("Models tracked", "RAG answers + landfall forecaster")

st.markdown(
    """
**Use the pages in the sidebar:**

- **Model Evaluation** — how the RAG models and the landfall-intensity model score
  against ground truth: groundedness, hallucination rate, A/B significance, and the
  forecaster's accuracy and confusion matrix.
- **Experiment Tracking** — configuration sweeps (chunk size, retrieval, embeddings,
  prompts) and how metrics move across runs.

The **Weather Intelligence** landing page (`dashboards/weather_intelligence.html`)
is a standalone file — open it in a browser for the interactive storm-track map.
"""
)
