"""Thin model-registry helpers (promote/compare) over MLflow.

Used to promote the winning model from a benchmark to a stage ('staging' ->
'production') once its metrics beat the incumbent. No-ops gracefully without
MLflow so the rest of the platform stays importable.
"""

from __future__ import annotations

from typing import Optional


def promote(model_name: str, version: str, stage: str = "Staging") -> Optional[str]:
    try:
        from mlflow.tracking import MlflowClient

        client = MlflowClient()
        client.transition_model_version_stage(model_name, version, stage)
        return f"{model_name} v{version} -> {stage}"
    except Exception:  # noqa: BLE001
        return None
