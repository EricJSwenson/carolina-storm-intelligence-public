"""Context- and answer-relevance metrics."""
from __future__ import annotations

from typing import List

from storm_eval.evaluation.metrics._text import cosine


def context_relevance(query: str, contexts: List[str]) -> float:
    """Mean similarity of retrieved chunks to the query (retrieval quality)."""
    if not contexts:
        return 0.0
    return sum(cosine(query, c) for c in contexts) / len(contexts)


def answer_relevance(query: str, answer: str) -> float:
    """Does the answer actually address the question?"""
    return cosine(query, answer)
