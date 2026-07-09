"""Retriever: embed query -> vector search -> optional rerank."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from storm_eval.config import settings
from storm_eval.rag.embeddings import Embedder, get_embedder
from storm_eval.rag.reranker import rerank
from storm_eval.rag.vectorstore import Hit, VectorStore


@dataclass
class Retriever:
    store: VectorStore
    embedder: Embedder
    top_k: int = settings.top_k
    rerank_strategy: str = "lexical"

    @classmethod
    def build(cls, store: VectorStore, **kw) -> "Retriever":
        return cls(store=store, embedder=get_embedder(), **kw)

    def retrieve(self, query: str, k: int | None = None) -> List[Hit]:
        k = k or self.top_k
        qv = self.embedder.embed([query])[0]
        hits = self.store.search(qv, k=max(k, self.top_k))
        hits = rerank(query, hits, self.rerank_strategy)
        return hits[:k]
