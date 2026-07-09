"""Generation backends behind one interface.

Two deterministic mock models let the platform run and be tested with no API
keys, and -- importantly -- they behave *differently*, so the evaluation harness
measures a genuine quality gap rather than a contrived one:

  * ``mock-grounded`` answers strictly from retrieved context (faithful).
  * ``mock-naive``    answers from a small built-in prior that contains real
                      errors and ignores context when it "already knows" --
                      a stand-in for an ungrounded model that hallucinates.

``openai`` and ``ollama`` are production backends with the same interface.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import List

from storm_eval.config import settings
from storm_eval.rag.vectorstore import Hit

_SENT = re.compile(r"[^.!?]+[.!?]")
_TOKEN = re.compile(r"[a-z0-9]+")


@dataclass
class Generation:
    text: str
    model: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    contexts: List[str] = field(default_factory=list)


def _approx_tokens(s: str) -> int:
    return max(1, len(s) // 4)


def _overlap(a: str, b: str) -> float:
    sa, sb = set(_TOKEN.findall(a.lower())), set(_TOKEN.findall(b.lower()))
    return len(sa & sb) / (len(sa) + 1e-9)


class MockGroundedLLM:
    """Returns the context sentence most relevant to the query."""

    name = "mock-grounded"

    @staticmethod
    def _score(query: str, sentence: str) -> float:
        base = _overlap(query, sentence)
        ql, sl = query.lower(), sentence.lower()
        # A grounded reader focuses on landfall sentences for a landfall question
        # rather than being distracted by peak-intensity prose.
        if any(c in ql for c in ("landfall", "ashore", "come ashore")):
            if "landfall" in sl or "ashore" in sl:
                base += 0.5
        return base

    def generate(self, query: str, hits: List[Hit]) -> Generation:
        t0 = time.perf_counter()
        context = " ".join(h.text for h in hits)
        sentences = [s.strip() for s in _SENT.findall(context)] or [context]
        best = max(sentences, key=lambda s: self._score(query, s)) if sentences else ""
        answer = best if best else "The retrieved context does not contain that information."
        latency = (time.perf_counter() - t0) * 1000
        prompt = query + context
        return Generation(answer, self.name, latency, _approx_tokens(prompt),
                          _approx_tokens(answer), [h.text for h in hits])


class MockNaiveLLM:
    """Answers from a small, partially-wrong prior; ignores context when it
    thinks it already knows. Produces checkable hallucinations."""

    name = "mock-naive"
    # Deliberately seeded with errors (real landfall categories differ).
    _PRIOR = {
        "florence": "Hurricane Florence made landfall in North Carolina as a Category 3 hurricane.",
        "dorian": "Hurricane Dorian struck North Carolina as a Category 4 hurricane.",
        "irene": "Hurricane Irene made landfall in North Carolina as a Category 3 hurricane.",
    }

    def generate(self, query: str, hits: List[Hit]) -> Generation:
        t0 = time.perf_counter()
        q = query.lower()
        answer = next((v for k, v in self._PRIOR.items() if k in q), None)
        if answer is None:  # falls back to context when it has no prior
            context = " ".join(h.text for h in hits)
            sents = [s.strip() for s in _SENT.findall(context)] or [context]
            answer = max(sents, key=lambda s: _overlap(query, s)) if sents else ""
        latency = (time.perf_counter() - t0) * 1000
        return Generation(answer, self.name, latency, _approx_tokens(query),
                          _approx_tokens(answer), [h.text for h in hits])


class OpenAIGenerator:
    name = "openai"

    def __init__(self, model: str | None = None):
        from openai import OpenAI  # lazy

        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = model or settings.openai_model

    def generate(self, query: str, hits: List[Hit]) -> Generation:
        from storm_eval.rag.prompts import render_rag_prompt

        context = "\n\n".join(f"[{i+1}] {h.text}" for i, h in enumerate(hits))
        prompt = render_rag_prompt(query=query, context=context)
        t0 = time.perf_counter()
        resp = self.client.chat.completions.create(
            model=self.model, messages=[{"role": "user", "content": prompt}], temperature=0,
        )
        latency = (time.perf_counter() - t0) * 1000
        u = resp.usage
        return Generation(resp.choices[0].message.content, f"openai:{self.model}", latency,
                          u.prompt_tokens, u.completion_tokens, [h.text for h in hits])


class OllamaGenerator:
    name = "ollama"

    def __init__(self, model: str = "llama3.1"):
        self.model = model

    def generate(self, query: str, hits: List[Hit]) -> Generation:
        import json
        import urllib.request

        from storm_eval.rag.prompts import render_rag_prompt

        context = "\n\n".join(f"[{i+1}] {h.text}" for i, h in enumerate(hits))
        prompt = render_rag_prompt(query=query, context=context)
        body = json.dumps({"model": self.model, "prompt": prompt, "stream": False}).encode()
        t0 = time.perf_counter()
        req = urllib.request.Request(f"{settings.ollama_host}/api/generate", data=body)
        with urllib.request.urlopen(req, timeout=120) as r:  # noqa: S310
            out = json.load(r)
        latency = (time.perf_counter() - t0) * 1000
        answer = out["response"]
        return Generation(answer, f"ollama:{self.model}", latency,
                          _approx_tokens(prompt), _approx_tokens(answer), [h.text for h in hits])


_REGISTRY = {
    "mock-grounded": MockGroundedLLM,
    "mock-naive": MockNaiveLLM,
    "openai": OpenAIGenerator,
    "ollama": OllamaGenerator,
}


def get_generator(name: str | None = None):
    name = name or settings.llm_backend
    if name in ("mock", "mock-grounded"):
        return MockGroundedLLM()
    cls = _REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"unknown llm backend: {name}")
    return cls()
