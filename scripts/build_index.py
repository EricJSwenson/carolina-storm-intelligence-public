"""Rebuild the persisted vector store from the gold corpus (for serving)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from storm_eval.db import connect, init_schema
from storm_eval.experiments.runner import build_store
from storm_eval.pipelines.local import gold_corpus, run_medallion

con = connect(); init_schema(con); run_medallion(con)
store = build_store(gold_corpus(con), chunk_size=320, embedding_backend="local")
store.persist()
print(f"indexed {len(store.ids)} chunks -> {store.path}")
