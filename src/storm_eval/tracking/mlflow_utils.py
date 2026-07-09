"""MLflow experiment tracking, with a local JSONL fallback.

In production every benchmark/experiment run logs params + metrics to MLflow so
model performance is comparable over time and models can be promoted via the
registry. If MLflow isn't installed (or in minimal CI), runs append to
``data/mlruns/local_runs.jsonl`` instead, so nothing silently disappears.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

from storm_eval.config import settings


def _fallback(model: str, params: Dict, metrics: Dict, run_id: str) -> None:
    path = Path(settings.mlflow_uri)
    path.mkdir(parents=True, exist_ok=True)
    rec = {"run_id": run_id, "model": model, "ts": datetime.now(timezone.utc).isoformat(),
           "params": params, "metrics": metrics}
    with open(path / "local_runs.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")


def log_run(model: str, params: Dict, metrics: Dict, run_id: str) -> None:
    try:
        import mlflow  # heavy; imported lazily

        mlflow.set_tracking_uri(settings.mlflow_uri)
        mlflow.set_experiment(settings.experiment_name)
        with mlflow.start_run(run_name=f"{model}-{run_id}"):
            mlflow.log_param("model", model)
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
    except Exception:  # noqa: BLE001 - fall back so runs are never lost
        _fallback(model, params, metrics, run_id)
