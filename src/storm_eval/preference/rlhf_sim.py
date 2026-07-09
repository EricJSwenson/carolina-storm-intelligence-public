"""Simulate an RLHF-style improvement loop (illustrative, no GPU training).

Given a reward signal (groundedness minus a hallucination penalty), each
iteration applies best-of-N selection over candidate answers -- the same
principle as rejection-sampling / RLHF preference optimization -- and tracks how
mean reward improves as N grows. Demonstrates the *workflow* and lets you talk
about reward modeling and preference optimization concretely.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class RLHFIteration:
    n_candidates: int
    mean_reward: float


def best_of_n(candidates: List[float], n: int, rng: np.random.Generator) -> float:
    sample = rng.choice(candidates, size=n, replace=True)
    return float(np.max(sample))


def simulate(reward_pool: List[float], schedule=(1, 2, 4, 8, 16),
             trials: int = 200, seed: int = 13) -> List[RLHFIteration]:
    rng = np.random.default_rng(seed)
    out = []
    for n in schedule:
        rewards = [best_of_n(reward_pool, n, rng) for _ in range(trials)]
        out.append(RLHFIteration(n, float(np.mean(rewards))))
    return out


def reward_pool_from_evals(con) -> List[float]:
    rows = con.execute(
        "SELECT groundedness, hallucinated FROM evaluations"
    ).fetchall()
    return [g - (1.0 if h else 0.0) for g, h in rows]
