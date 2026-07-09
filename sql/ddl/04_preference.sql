-- Preference pairs for RLHF-style workflows: chosen vs rejected answers.
CREATE TABLE IF NOT EXISTS preference_pairs (
    pair_id     VARCHAR,
    question_id VARCHAR,
    question    VARCHAR,
    chosen      VARCHAR,
    rejected    VARCHAR,
    chosen_model   VARCHAR,
    rejected_model VARCHAR,
    margin      DOUBLE,          -- reward-signal difference
    created_ts  TIMESTAMP
);
