"""Statistical A/B comparison between two models' per-question metric scores.

Uses a paired bootstrap (no distributional assumptions) plus a paired t-test for
a parametric cross-check. Returns the mean difference, 95% CI, and p-value so
reports can claim *statistically significant* improvements rather than eyeballed
ones.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from scipy import stats


@dataclass
class ABResult:
    metric: str
    model_a: str
    model_b: str
    mean_a: float
    mean_b: float
    diff: float            # mean_b - mean_a
    ci_low: float
    ci_high: float
    p_value: float
    significant: bool

    def summary(self) -> str:
        verdict = "significant" if self.significant else "not significant"
        return (f"{self.metric}: {self.model_b} - {self.model_a} = {self.diff:+.3f} "
                f"(95% CI [{self.ci_low:+.3f}, {self.ci_high:+.3f}], p={self.p_value:.4f}, {verdict})")


def paired_ab(metric: str, model_a: str, scores_a: List[float],
              model_b: str, scores_b: List[float],
              alpha: float = 0.05, n_boot: int = 10000, seed: int = 7) -> ABResult:
    a, b = np.asarray(scores_a, float), np.asarray(scores_b, float)
    if len(a) != len(b):
        raise ValueError("paired A/B requires equal-length, aligned score vectors")
    diffs = b - a
    rng = np.random.default_rng(seed)
    boot = rng.choice(diffs, size=(n_boot, len(diffs)), replace=True).mean(axis=1)
    ci_low, ci_high = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    if np.allclose(diffs, 0):
        p = 1.0
    else:
        p = float(stats.ttest_rel(b, a).pvalue)
    return ABResult(metric, model_a, model_b, float(a.mean()), float(b.mean()),
                    float(diffs.mean()), float(ci_low), float(ci_high), p,
                    significant=(p < alpha and ci_low * ci_high > 0))


def compare_runs(con, metric: str, model_a: str, model_b: str, **kw) -> ABResult:
    """Pull aligned per-question scores for two models and run the A/B test."""
    q = ("SELECT question_id, {m} FROM evaluations WHERE model = ? "
         "ORDER BY question_id").format(m=metric)
    rows_a = dict(con.execute(q, [model_a]).fetchall())
    rows_b = dict(con.execute(q, [model_b]).fetchall())
    keys = sorted(set(rows_a) & set(rows_b))
    return paired_ab(metric, model_a, [rows_a[k] for k in keys],
                     model_b, [rows_b[k] for k in keys], **kw)
