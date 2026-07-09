"""Post-training / configuration experiment runner.

Sweeps one config dimension (chunk size, embedding backend, rerank strategy,
prompt template), rebuilding the vector store when retrieval inputs change, and
benchmarks each cell so the experiments can be compared with the same A/B test
used for models. Each cell logs to MLflow.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from storm_eval.db import connect, init_schema
from storm_eval.evaluation.benchmark import load_eval_set, run_benchmark
from storm_eval.pipelines.local import gold_corpus, run_medallion
from storm_eval.rag.chunking import chunk_text
from storm_eval.rag.embeddings import get_embedder
from storm_eval.rag.vectorstore import LocalVectorStore


def build_store(corpus: List, chunk_size: int, embedding_backend: str) -> LocalVectorStore:
    embedder = get_embedder(embedding_backend)
    ids, doc_ids, texts = [], [], []
    for doc_id, text in corpus:
        for ch in chunk_text(doc_id, text, size=chunk_size):
            ids.append(ch.chunk_id); doc_ids.append(doc_id); texts.append(ch.text)
    store = LocalVectorStore()
    store.add(ids, doc_ids, texts, embedder.embed(texts))
    return store


@dataclass
class Cell:
    label: str
    run_id: str


def run_sweep(param: str, values: List[Any], model: str = "mock-grounded",
              base_chunk: int = 320, base_embed: str = "local") -> Dict[str, str]:
    """Run a one-dimensional sweep; returns {value_label: run_id}."""
    con = connect(); init_schema(con)
    run_medallion(con)
    corpus = gold_corpus(con)
    cases = load_eval_set()
    results: Dict[str, str] = {}
    for v in values:
        chunk = v if param == "chunk_size" else base_chunk
        embed = v if param == "embedding_backend" else base_embed
        rerank = v if param == "rerank" else "lexical"
        template = v if param == "template" else "rag_v1.jinja"
        store = build_store(corpus, chunk, embed)
        run_id = run_benchmark(model, store, con, cases, top_k=4,
                               rerank=rerank, template=template)
        results[f"{param}={v}"] = run_id
    return results
