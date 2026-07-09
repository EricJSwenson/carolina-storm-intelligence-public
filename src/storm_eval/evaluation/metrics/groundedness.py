"""Groundedness / faithfulness: is the answer supported by retrieved context?"""
from __future__ import annotations

from typing import List

from storm_eval.evaluation.metrics._text import max_support, sentences

SUPPORT_THRESHOLD = 0.35


def groundedness(answer: str, contexts: List[str], threshold: float = SUPPORT_THRESHOLD) -> float:
    """Fraction of answer sentences supported by some context sentence [0,1]."""
    claims = sentences(answer)
    if not claims:
        return 0.0
    supported = sum(1 for c in claims if max_support(c, contexts) >= threshold)
    return supported / len(claims)


def faithfulness(answer: str, contexts: List[str]) -> float:
    """Stricter variant: penalizes any single unsupported claim."""
    claims = sentences(answer)
    if not claims:
        return 0.0
    return min(max_support(c, contexts) for c in claims)
