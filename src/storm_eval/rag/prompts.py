"""Prompt rendering. Templates live in conf/prompts/*.jinja so the experiments
layer can A/B prompt variants without touching code."""

from __future__ import annotations

from functools import lru_cache

from jinja2 import Template

from storm_eval.config import ROOT

PROMPT_DIR = ROOT / "conf" / "prompts"


@lru_cache(maxsize=None)
def _load(name: str) -> Template:
    return Template((PROMPT_DIR / name).read_text(encoding="utf-8"))


def render_rag_prompt(query: str, context: str, template: str = "rag_v1.jinja") -> str:
    return _load(template).render(query=query, context=context)


def render_judge_prompt(query: str, answer: str, context: str,
                        template: str = "judge_groundedness.jinja") -> str:
    return _load(template).render(query=query, answer=answer, context=context)
