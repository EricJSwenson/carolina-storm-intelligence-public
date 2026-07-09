"""LLM-as-judge groundedness scoring.

Offline, ``HeuristicJudge`` reuses the embedding-overlap groundedness so the
pipeline runs with no API. ``OpenAIJudge`` is the production judge that parses a
JSON verdict from the model. Both expose ``score(query, answer, contexts)``.
"""

from __future__ import annotations

import json
from typing import List

from storm_eval.config import settings
from storm_eval.evaluation.metrics.groundedness import groundedness


class HeuristicJudge:
    name = "heuristic"

    def score(self, query: str, answer: str, contexts: List[str]) -> float:
        return groundedness(answer, contexts)


class OpenAIJudge:
    name = "openai-judge"

    def __init__(self, model: str | None = None):
        from openai import OpenAI  # lazy

        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = model or settings.openai_model

    def score(self, query: str, answer: str, contexts: List[str]) -> float:
        from storm_eval.rag.prompts import render_judge_prompt

        prompt = render_judge_prompt(query=query, answer=answer, context="\n\n".join(contexts))
        resp = self.client.chat.completions.create(
            model=self.model, messages=[{"role": "user", "content": prompt}],
            temperature=0, response_format={"type": "json_object"},
        )
        verdict = json.loads(resp.choices[0].message.content)
        return 1.0 if verdict.get("grounded") else 0.0


def get_judge(name: str = "heuristic"):
    return OpenAIJudge() if name in ("openai", "openai-judge") else HeuristicJudge()
