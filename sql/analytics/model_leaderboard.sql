-- Model leaderboard across the latest run of each model.
WITH latest AS (
    SELECT *, DENSE_RANK() OVER (PARTITION BY model ORDER BY run_ts DESC) AS r
    FROM evaluations
)
SELECT
    model,
    COUNT(*)                                                   AS n_questions,
    ROUND(AVG(groundedness), 3)                                AS groundedness,
    ROUND(AVG(answer_relevance), 3)                            AS answer_relevance,
    ROUND(AVG(reward), 3)                                      AS reward,
    ROUND(SUM(CASE WHEN hallucinated THEN 1.0 ELSE 0 END)
          / NULLIF(SUM(CASE WHEN hallucinated IS NOT NULL THEN 1 ELSE 0 END), 0), 3)
                                                               AS hallucination_rate,
    ROUND(AVG(latency_ms), 1)                                  AS latency_ms,
    ROUND(SUM(cost_usd), 4)                                    AS total_cost_usd
FROM latest
WHERE r = 1
GROUP BY model
ORDER BY reward DESC;
