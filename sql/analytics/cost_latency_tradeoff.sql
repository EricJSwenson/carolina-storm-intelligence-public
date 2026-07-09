-- Cost / latency / quality trade-off for model selection.
SELECT
    model,
    ROUND(AVG(reward), 3)        AS mean_reward,
    ROUND(AVG(latency_ms), 1)    AS mean_latency_ms,
    ROUND(AVG(cost_usd), 6)      AS mean_cost_per_query
FROM evaluations
GROUP BY model
ORDER BY mean_reward DESC;
