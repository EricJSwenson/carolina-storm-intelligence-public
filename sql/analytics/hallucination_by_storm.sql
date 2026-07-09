-- Where do models hallucinate? Hallucination rate per storm (checked questions).
SELECT
    e.storm_id,
    t.name,
    e.model,
    COUNT(*)                                              AS checked_questions,
    SUM(CASE WHEN e.hallucinated THEN 1 ELSE 0 END)       AS hallucinations
FROM evaluations e
LEFT JOIN storm_truth t ON t.storm_id = e.storm_id
WHERE e.hallucinated IS NOT NULL
GROUP BY e.storm_id, t.name, e.model
ORDER BY hallucinations DESC, e.storm_id;
