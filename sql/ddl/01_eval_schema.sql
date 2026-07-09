-- Core evaluation store. One row per (model, question, run).
CREATE TABLE IF NOT EXISTS evaluations (
    run_id            VARCHAR,
    run_ts            TIMESTAMP,
    model             VARCHAR,
    question_id       VARCHAR,
    storm_id          VARCHAR,
    question          VARCHAR,
    answer            VARCHAR,
    groundedness      DOUBLE,
    faithfulness      DOUBLE,
    context_relevance DOUBLE,
    answer_relevance  DOUBLE,
    judge_score       DOUBLE,
    hallucinated      BOOLEAN,   -- NULL when no structured check applies
    latency_ms        DOUBLE,
    cost_usd          DOUBLE,
    prompt_tokens     INTEGER,
    completion_tokens INTEGER,
    reward            DOUBLE     -- groundedness minus hallucination penalty
);
