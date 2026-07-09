-- Model leaderboard: the table the dashboard and promotion logic read.
select
    model,
    count(*)                                              as n_questions,
    round(avg(groundedness), 3)                           as mean_groundedness,
    round(avg(answer_relevance), 3)                       as mean_answer_relevance,
    round(avg(reward), 3)                                 as mean_reward,
    round(sum(is_hallucination) * 1.0
          / nullif(sum(is_checked), 0), 3)                as hallucination_rate,
    round(avg(latency_ms), 1)                             as mean_latency_ms,
    round(sum(cost_usd), 4)                               as total_cost_usd
from {{ ref('fct_evaluations') }}
group by model
order by mean_reward desc
