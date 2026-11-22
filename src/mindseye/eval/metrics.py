"""Scoring and statistics.

Per-example scores are in [0, 1]: exact match for multiple choice, mean relative
accuracy (MRA) for numerical answers, both as defined by VSI-Bench.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Sequence

import numpy as np

# Thresholds used by the VSI-Bench reference scorer:
# ``np.linspace(0.5, 0.95, int((0.95 - 0.5) / 0.05 + 2))`` gives these 10 values.
MRA_THRESHOLDS = np.linspace(0.5, 0.95, 10)

REL_DIRECTION = (
    "object_rel_direction_easy",
    "object_rel_direction_medium",
    "object_rel_direction_hard",
)


def mean_relative_accuracy(pred: float | None, target: float) -> float:
    """Fraction of thresholds ``c`` for which ``|pred - target| / target <= 1 - c``."""
    if pred is None or not math.isfinite(pred):
        return 0.0
    rel_err = abs(pred - target) / target
    return float(np.mean(rel_err <= 1 - MRA_THRESHOLDS))


def mean_by_type(scores: Sequence[float], types: Sequence[str]) -> dict[str, float]:
    sums: dict[str, list[float]] = defaultdict(list)
    for s, t in zip(scores, types, strict=True):
        sums[t].append(s)
    return {t: float(np.mean(v)) for t, v in sorted(sums.items())}


def vsibench_overall(scores: Sequence[float], types: Sequence[str]) -> float:
    """Official VSI-Bench aggregate: mean over tasks, with the three
    relative-direction difficulties first averaged into one task."""
    per_type = mean_by_type(scores, types)
    present = [per_type.pop(t) for t in REL_DIRECTION if t in per_type]
    if present:
        if len(present) != len(REL_DIRECTION):
            raise ValueError("relative-direction subtypes are incomplete; overall is undefined")
        per_type["object_rel_direction"] = sum(present) / len(present)
    return float(np.mean(list(per_type.values())))


def bootstrap_ci(
    statistic: Callable[[np.ndarray], float],
    n: int,
    *,
    clusters: Sequence | None = None,
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap interval for ``statistic(indices)``.

    With ``clusters`` (e.g. the scene of each question) whole clusters are
    resampled, which respects the dependence between questions on one video.
    A resample on which the statistic is undefined is skipped.
    """
    rng = np.random.default_rng(seed)
    if clusters is None:
        members = [np.array([i]) for i in range(n)]
    else:
        by_cluster: dict = defaultdict(list)
        for i, c in enumerate(clusters):
            by_cluster[c].append(i)
        members = [np.array(v) for _, v in sorted(by_cluster.items(), key=lambda kv: str(kv[0]))]
    values = []
    for _ in range(n_resamples):
        picks = rng.integers(0, len(members), len(members))
        idx = np.concatenate([members[p] for p in picks])
        try:
            values.append(statistic(idx))
        except ValueError:
            continue
    if len(values) < 0.9 * n_resamples:
        raise ValueError("statistic undefined on more than 10% of resamples")
    lo, hi = np.quantile(values, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def mcnemar_exact(a_correct: Sequence[bool], b_correct: Sequence[bool]) -> dict[str, float]:
    """Exact two-sided McNemar test on paired binary outcomes."""
    a = np.asarray(a_correct, dtype=bool)
    b = np.asarray(b_correct, dtype=bool)
    if a.shape != b.shape:
        raise ValueError("paired outcomes must have the same length")
    only_a = int(np.sum(a & ~b))
    only_b = int(np.sum(~a & b))
    n = only_a + only_b
    if n == 0:
        p = 1.0
    else:
        k = min(only_a, only_b)
        tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
        p = min(1.0, 2 * tail)
    return {"only_a": only_a, "only_b": only_b, "p_value": p}
