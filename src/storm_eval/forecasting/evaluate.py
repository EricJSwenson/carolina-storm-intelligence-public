"""Evaluate the landfall-intensity model with a leakage-safe, year-based split.

Storms are split by *year* so no storm leaks between train and test, and metrics
include exact accuracy, within-one-category accuracy (a forecaster cares whether
you're close), and macro-F1. Predictions are written to the warehouse so the
evaluation dashboard tracks the predictive model alongside the LLM systems --
the same harness, two model types.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from storm_eval.forecasting.features import build_training_table
from storm_eval.forecasting.model import train


@dataclass
class ForecastReport:
    n_train: int
    n_test: int
    accuracy: float
    within_one: float
    macro_f1: float
    confusion: list


def year_split(years: np.ndarray, test_frac: float = 0.25, seed: int = 3):
    uniq = np.unique(years)
    rng = np.random.default_rng(seed)
    test_years = set(rng.choice(uniq, size=max(1, int(len(uniq) * test_frac)), replace=False))
    test_mask = np.array([y in test_years for y in years])
    return ~test_mask, test_mask


def evaluate(storms) -> tuple[ForecastReport, object]:
    X, y, years = build_training_table(storms)
    tr, te = year_split(years)
    model = train(X[tr], y[tr])
    pred = model.predict(X[te])
    report = ForecastReport(
        n_train=int(tr.sum()), n_test=int(te.sum()),
        accuracy=float(accuracy_score(y[te], pred)),
        within_one=float(np.mean(np.abs(y[te] - pred) <= 1)),
        macro_f1=float(f1_score(y[te], pred, average="macro", zero_division=0)),
        confusion=confusion_matrix(y[te], pred, labels=[0, 1, 2, 3, 4, 5]).tolist(),
    )
    return report, model


def persist_predictions(con, storms, model) -> None:
    """Write per-storm predicted vs actual landfall category to the warehouse."""
    con.execute("""
        CREATE TABLE IF NOT EXISTS forecast_evaluations (
            run_ts TIMESTAMP, storm_year INTEGER,
            predicted_category INTEGER, actual_category INTEGER, correct BOOLEAN
        )""")
    X, y, years = build_training_table(storms)
    preds = model.predict(X)
    ts = datetime.now(timezone.utc)
    rows = [(ts, int(years[i]), int(preds[i]), int(y[i]), bool(preds[i] == y[i]))
            for i in range(len(y))]
    con.execute("DELETE FROM forecast_evaluations")
    con.executemany("INSERT INTO forecast_evaluations VALUES (?,?,?,?,?)", rows)
