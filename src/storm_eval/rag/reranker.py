"""Rerankers behind one interface.

``none`` keeps retrieval order. ``lexical`` re-scores by query/term overlap on
top of the dense score -- a cheap, dependency-free reranker that measurably
shifts results, so the experiments layer can A/B it against ``none`` and a
production cross-encoder.
"""

from __future__ import annotations

import re
from typing import List

from storm_eval.rag.vectorstore import Hit

_TOKEN = re.compile(r"[a-z0-9]+")


def _overlap(query: str, text: str) -> float:
    q = set(_TOKEN.findall(query.lower()))
    t = set(_TOKEN.findall(text.lower()))
    return len(q & t) / (len(q) + 1e-9)


def rerank(query: str, hits: List[Hit], strategy: str = "lexical") -> List[Hit]:
    if strategy == "none" or not hits:
        return hits
    if strategy == "lexical":
        scored = [Hit(h.chunk_id, h.doc_id, h.text, 0.5 * h.score + 0.5 * _overlap(query, h.text))
                  for h in hits]
        return sorted(scored, key=lambda h: -h.score)
    if strategy == "cross-encoder":
        from sentence_transformers import CrossEncoder  # lazy, production only

        model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        scores = model.predict([(query, h.text) for h in hits])
        ranked = sorted(zip(hits, scores), key=lambda x: -x[1])
        return [Hit(h.chunk_id, h.doc_id, h.text, float(s)) for h, s in ranked]
    raise ValueError(f"unknown rerank strategy: {strategy}")
