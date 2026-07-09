-- Cleaned evaluation rows: one per (model, question, run).
select
    run_id,
    run_ts,
    model,
    question_id,
    storm_id,
    groundedness,
    faithfulness,
    context_relevance,
    answer_relevance,
    judge_score,
    hallucinated,
    reward,
    latency_ms,
    cost_usd
from evaluations
