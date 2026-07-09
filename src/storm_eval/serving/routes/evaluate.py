from __future__ import annotations

from fastapi import APIRouter

from storm_eval.evaluation.ground_truth import hallucination_flag
from storm_eval.evaluation.metrics import answer_relevance, faithfulness, groundedness
from storm_eval.serving.schemas import EvaluateRequest, EvaluateResponse

router = APIRouter()


@router.post("/evaluate", response_model=EvaluateResponse)
def evaluate(req: EvaluateRequest) -> EvaluateResponse:
    truth = {"landfall_category": req.expected_landfall_category}
    return EvaluateResponse(
        groundedness=groundedness(req.answer, req.contexts),
        faithfulness=faithfulness(req.answer, req.contexts),
        answer_relevance=answer_relevance(req.question, req.answer),
        hallucinated=hallucination_flag(req.answer, truth),
    )
