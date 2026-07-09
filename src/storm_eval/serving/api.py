"""FastAPI app exposing the RAG system and the evaluator.

Run:  uvicorn storm_eval.serving.api:app --reload
Docs: http://localhost:8000/docs
"""
from __future__ import annotations

from fastapi import FastAPI

from storm_eval.serving.routes import evaluate, query

app = FastAPI(title="Storm LLM Eval Platform", version="0.1.0")
app.include_router(query.router, tags=["rag"])
app.include_router(evaluate.router, tags=["evaluation"])


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
