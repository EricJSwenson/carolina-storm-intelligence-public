"""Token-based cost estimate (USD per 1K tokens). Local/mock models are free."""
from __future__ import annotations

PRICING = {  # (prompt, completion) USD per 1K tokens
    "mock-grounded": (0.0, 0.0),
    "mock-naive": (0.0, 0.0),
    "openai:gpt-4o-mini": (0.00015, 0.0006),
    "openai:gpt-4o": (0.0025, 0.01),
}


def cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    p, c = PRICING.get(model, (0.0, 0.0))
    return round((prompt_tokens / 1000) * p + (completion_tokens / 1000) * c, 6)
