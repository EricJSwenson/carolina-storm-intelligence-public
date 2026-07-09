"""Train + evaluate the landfall-intensity model and persist it.

    python scripts/train_forecaster.py

Trains on synthetic storms by default (for an offline, runnable demo). Point
``generate_dataset`` at a parsed real HURDAT2 file to train on the full record.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from storm_eval.db import connect, init_schema
from storm_eval.forecasting.dataset import generate_dataset
from storm_eval.forecasting.evaluate import evaluate, persist_predictions
from storm_eval.forecasting.model import save


def main() -> None:
    storms = generate_dataset(n_storms=500)
    report, model = evaluate(storms)
    print("Landfall-intensity model (held-out, year-split):")
    print(f"  train storms        {report.n_train}")
    print(f"  test storms         {report.n_test}")
    print(f"  exact accuracy      {report.accuracy:.3f}")
    print(f"  within-1 category   {report.within_one:.3f}")
    print(f"  macro F1            {report.macro_f1:.3f}")

    # Refit on all data for the saved artifact and write predictions to warehouse.
    from storm_eval.forecasting.features import build_training_table
    from storm_eval.forecasting.model import train
    X, y, _ = build_training_table(storms)
    final = train(X, y)
    save(final)
    con = connect(); init_schema(con)
    persist_predictions(con, storms, final)
    print(f"  saved model + wrote {len(y)} predictions to forecast_evaluations")


if __name__ == "__main__":
    main()
