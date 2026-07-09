from __future__ import annotations

from fastapi import APIRouter

from storm_eval.rag.pipeline import RAGPipeline
from storm_eval.rag.vectorstore import LocalVectorStore
from storm_eval.serving.schemas import QueryRequest, QueryResponse, Source

router = APIRouter()


def _store() -> LocalVectorStore:
    # Loads the persisted vector store built by the gold layer / demo.
    return LocalVectorStore().load()


@router.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    pipeline = RAGPipeline(_store(), model=req.model, top_k=req.top_k)
    res = pipeline.answer(req.question)
    return QueryResponse(
        question=res.query, answer=res.answer, model=res.model,
        latency_ms=res.latency_ms,
        sources=[Source(doc_id=h.doc_id, score=h.score, text=h.text) for h in res.hits],
    )
