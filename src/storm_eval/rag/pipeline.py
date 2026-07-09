"""End-to-end RAG pipeline: retrieve -> generate.

Every component (embedder, vector store, reranker, generator) is injected, so a
single ``RAGPipeline`` can be reconfigured for any experiment cell.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from storm_eval.rag.generator import Generation, get_generator
from storm_eval.rag.retriever import Retriever
from storm_eval.rag.vectorstore import Hit, VectorStore


@dataclass
class RAGResult:
    query: str
    answer: str
    model: str
    hits: List[Hit]
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int


class RAGPipeline:
    def __init__(self, store: VectorStore, model: str = "mock-grounded",
                 top_k: int = 4, rerank_strategy: str = "lexical"):
        self.retriever = Retriever.build(store, top_k=top_k, rerank_strategy=rerank_strategy)
        self.generator = get_generator(model)

    def answer(self, query: str) -> RAGResult:
        hits = self.retriever.retrieve(query)
        gen: Generation = self.generator.generate(query, hits)
        return RAGResult(
            query=query, answer=gen.text, model=gen.model, hits=hits,
            latency_ms=gen.latency_ms, prompt_tokens=gen.prompt_tokens,
            completion_tokens=gen.completion_tokens,
        )
