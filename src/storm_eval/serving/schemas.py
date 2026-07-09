"""Pydantic request/response models for the serving API."""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel


class QueryRequest(BaseModel):
    question: str
    model: str = "mock-grounded"
    top_k: int = 4


class Source(BaseModel):
    doc_id: str
    score: float
    text: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    model: str
    latency_ms: float
    sources: List[Source]


class EvaluateRequest(BaseModel):
    question: str
    answer: str
    contexts: List[str]
    expected_landfall_category: Optional[int] = None


class EvaluateResponse(BaseModel):
    groundedness: float
    faithfulness: float
    answer_relevance: float
    hallucinated: Optional[bool]
