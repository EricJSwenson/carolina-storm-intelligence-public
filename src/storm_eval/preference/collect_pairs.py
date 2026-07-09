"""Build preference-ranking pairs from benchmarked answers.

For each question where two models answered, the higher-reward answer becomes
``chosen`` and the lower becomes ``rejected``. Reward favors grounded, non-
hallucinated answers. The resulting pairs are the dataset a reward model / DPO
run would consume -- the post-training experimentation artifact.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import List, Tuple


def _reward(groundedness: float, hallucinated) -> float:
    penalty = 1.0 if hallucinated else 0.0
    return groundedness - penalty


def build_preference_pairs(con) -> int:
    """Populate preference_pairs from the evaluations table. Returns pair count."""
    rows = con.execute(
        "SELECT question_id, question, model, answer, groundedness, hallucinated "
        "FROM evaluations"
    ).fetchall()
    by_q: dict[str, List[Tuple]] = {}
    for qid, question, model, answer, gnd, hl in rows:
        by_q.setdefault(qid, []).append((question, model, answer, gnd, hl))

    con.execute("DELETE FROM preference_pairs")
    pairs = []
    for qid, answers in by_q.items():
        if len(answers) < 2:
            continue
        ranked = sorted(answers, key=lambda a: _reward(a[3], a[4]), reverse=True)
        best, worst = ranked[0], ranked[-1]
        margin = _reward(best[3], best[4]) - _reward(worst[3], worst[4])
        if margin <= 0:
            continue
        pairs.append((uuid.uuid4().hex[:12], qid, best[0], best[2], worst[2],
                      best[1], worst[1], margin, datetime.now(timezone.utc)))
    con.executemany("INSERT INTO preference_pairs VALUES (?,?,?,?,?,?,?,?,?)", pairs)
    return len(pairs)
