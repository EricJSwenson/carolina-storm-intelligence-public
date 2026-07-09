"""Shared text helpers for metric computation."""
from __future__ import annotations

import re
from functools import lru_cache
from typing import List

import numpy as np

from storm_eval.rag.embeddings import get_embedder

_SENT = re.compile(r"[^.!?]+[.!?]")


@lru_cache(maxsize=1)
def _embedder():
    return get_embedder()


def sentences(text: str) -> List[str]:
    out = [s.strip() for s in _SENT.findall(text) if s.strip()]
    return out or ([text.strip()] if text.strip() else [])


def cosine(a: str, b: str) -> float:
    va, vb = _embedder().embed([a, b])
    denom = (np.linalg.norm(va) * np.linalg.norm(vb)) + 1e-9
    return float(np.dot(va, vb) / denom)


def max_support(claim: str, contexts: List[str]) -> float:
    ctx_sents = [s for c in contexts for s in sentences(c)]
    if not ctx_sents:
        return 0.0
    return max(cosine(claim, cs) for cs in ctx_sents)
