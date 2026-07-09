-- Fact table: latest run per model, with a non-hallucinated flag for rollups.
with ranked as (
    select *,
           dense_rank() over (partition by model order by run_ts desc) as run_rank
    from {{ ref('stg_evaluations') }}
)
select
    model,
    question_id,
    storm_id,
    groundedness,
    answer_relevance,
    reward,
    case when hallucinated then 1 else 0 end as is_hallucination,
    case when hallucinated is not null then 1 else 0 end as is_checked,
    latency_ms,
    cost_usd,
    run_ts
from ranked
where run_rank = 1
