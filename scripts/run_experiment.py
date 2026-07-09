"""Run a one-dimensional experiment sweep and print the per-cell metrics.

    python scripts/run_experiment.py chunk_size 160 320 640
    python scripts/run_experiment.py rerank none lexical
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from storm_eval.db import connect
from storm_eval.experiments.runner import run_sweep


def _cast(v: str):
    try:
        return int(v)
    except ValueError:
        return v


def main() -> None:
    if len(sys.argv) < 3:
        print("usage: run_experiment.py <param> <value> [value ...]")
        raise SystemExit(1)
    param, values = sys.argv[1], [_cast(v) for v in sys.argv[2:]]
    print(f"Sweeping {param} over {values}\n")
    runs = run_sweep(param, values)
    con = connect()
    for label, run_id in runs.items():
        row = con.execute(
            "SELECT AVG(groundedness), AVG(answer_relevance), AVG(reward), AVG(latency_ms) "
            "FROM evaluations WHERE run_id=?", [run_id]).fetchone()
        print(f"{label:<24} groundedness={row[0]:.3f}  relevance={row[1]:.3f}  "
              f"reward={row[2]:+.3f}  latency={row[3]:.1f}ms  (run {run_id})")


if __name__ == "__main__":
    main()
