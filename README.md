# Carolina Storm Intelligence

**An AI weather-intelligence platform built on 175 years of NOAA hurricane data — with a trustworthy LLM evaluation system at its core.**

The thesis: don't grade AI answers on vibes. Grade them against **verified ground truth**. This platform answers hurricane questions with a RAG system, then checks every answer against the **HURDAT2** best-track record — so a hallucination is a *measurable factual error*, not another model's opinion.

🔗 **Live weather map:** https://ericjswenson.github.io/carolina-storm-intelligence/
🔗 **Evaluation + Storm Center console:** https://carolina-storm-intelligence.streamlit.app/
💻 **Code:** https://github.com/EricjSwenson/carolina-storm-intelligence

> Runs fully offline with no API keys (deterministic embedder, mock LLMs, synthetic data, DuckDB). The PySpark / dbt / MLflow / Databricks path carries the production story.

---

## What it does

**1. Weather intelligence.** Every Atlantic storm from 1851–2025 on an interactive map, filtered to the storms that affected the Carolinas. Filter by year, category, and name; inspect tracks, intensity history, and a searchable storm database.

**2. A landfall-intensity model.** A gradient-boosted classifier estimates a storm's NC landfall Saffir-Simpson category from its pre-landfall track, with leakage-safe (year-split) evaluation. On live storms its estimate is shown **beside the official NHC forecast — clearly labeled a model estimate, not an official forecast.**

**3. A RAG assistant with a real evaluation harness.** Answers come from NOAA storm narratives and are scored against HURDAT2 ground truth for groundedness, with A/B testing (paired bootstrap + significance) and cost tracking across configurations.

## Live Storm Center

A server-side Streamlit page adds live, location-aware NOAA data (server-side so it can call the feeds cleanly, which a static page can't):

- **Local forecast** (api.weather.gov), **tides** (NOAA CO-OPS), and **buoy swell** (NDBC), for an auto-detected or manually entered location.
- **Active tropical cyclones** from the NHC, with the official advisory bulletin shown next to the model's landfall-category estimate.
- A **RAG search bar** answering storm questions, checked against HURDAT2.

The static landing page also carries a no-backend **in-browser assistant**: it answers from the embedded NOAA narratives, falls back to a hurricane-scoped Wikipedia lookup (with citations) for broader questions like damage and fatalities, handles greetings, and honestly says "I don't know" for anything off-topic.

## Tech stack

`Python` · `DuckDB` (medallion warehouse) · `scikit-learn` · `Streamlit` + `Plotly` · `FastAPI` · a from-scratch `RAG` pipeline (hashing embedder, vector store, reranker, grounded generator) · `pytest` + `ruff` in CI · deployed on **GitHub Pages** (static map) and **Streamlit Community Cloud** (console).

Production-path: PySpark bronze/silver/gold jobs, dbt models, MLflow tracking + registry, Databricks notebooks.

## Quickstart

```bash
git clone https://github.com/EricjSwenson/carolina-storm-intelligence.git
cd carolina-storm-intelligence
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

make demo            # runs the full pipeline: RAG eval + landfall model
pytest -q            # 19 tests
streamlit run dashboards/Home.py     # the dashboards + Storm Center
```

Build the static weather map (auto-downloads the full HURDAT2 record, falls back to a bundled sample offline):

```bash
python scripts/build_weather_page.py
```


## How the evaluation works

The model's answers are checked against structured HURDAT2 facts (landfall category, position, intensity), not against another LLM's judgment. A naive generator and a grounded generator can both score ~1.0 on embedding-similarity groundedness, yet the structured ground-truth check still catches the naive one's fabricated category — which is the entire point. See [docs/evaluation_methodology.md](docs/evaluation_methodology.md).

## Architecture & docs

[Architecture](docs/architecture.md) · [Data sources](docs/data_sources.md) · [Evaluation methodology](docs/evaluation_methodology.md) · [Live data plan](docs/live_data_plan.md) · [Deployment](docs/deployment.md) 

## Notes

- The live model estimate is **not an official forecast** — always follow the National Hurricane Center.
- 🏒 Easter egg: the synthetic storms the landfall model trains on are named after the 2025–26 Carolina Hurricanes roster (2026 Stanley Cup champions).

## License

MIT — see [LICENSE](LICENSE).
