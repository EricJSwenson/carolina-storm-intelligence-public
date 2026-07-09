# Evaluation methodology

## Metrics

| Metric | Question it answers | How |
|---|---|---|
| Context relevance | Did retrieval surface the right chunks? | Mean embedding similarity of query↔retrieved chunks |
| Answer relevance | Does the answer address the question? | Embedding similarity of query↔answer |
| Groundedness / faithfulness | Is each claim supported by retrieved context? | Fraction of answer sentences with a supporting context sentence |
| Judge score | LLM-as-judge groundedness verdict | Heuristic offline; OpenAI JSON verdict in prod |
| **Hallucination (ground truth)** | Does the answer contradict HURDAT2? | Parse the claimed category, compare to best-track truth |
| Latency / cost | Operational quality | Measured wall-clock; token-based cost |
| Reward | Optimization signal | `groundedness − hallucination_penalty` |


