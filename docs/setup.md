# Setup

## Requirements
- Python 3.10+
- No API keys or cloud account needed for the offline demo.

## Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Run the full pipeline (offline)
```bash
make demo          # full pipeline: RAG eval + A/B + preference + landfall model
```
This populates `data/warehouse.duckdb` and `data/vectorstore/`.

## Explore
```bash
make weather       # build the Weather Intelligence landing page (HTML)
make forecaster    # train + evaluate the landfall-intensity model
make test          # 17 unit + integration tests
make dashboard     # Streamlit console: Model Evaluation + Experiment Tracking
make api           # FastAPI service (localhost:8000/docs)
make experiments   # chunk-size sweep
```

## Query the API
```bash
curl -s localhost:8000/query -H 'content-type: application/json' \
  -d '{"question":"What category was Florence at NC landfall?","model":"mock-grounded"}'
```

## Switch to production backends
Copy `.env.example` to `.env` and set, e.g., `STORM_LLM_BACKEND=openai` and
`OPENAI_API_KEY=...`. The PySpark jobs in `src/storm_eval/pipelines/` and the
notebooks in `notebooks/` run on Databricks against Unity Catalog Delta tables.

## Docker
```bash
docker compose -f docker/docker-compose.yml up   # api + dashboard
```
