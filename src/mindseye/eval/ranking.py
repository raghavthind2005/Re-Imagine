"""Threshold-free metrics for a score that ranks claims (higher = more likely positive).

Ties are handled exactly: tied scores are flagged or kept together, never split.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _arrays(labels: Sequence[bool], scores: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(labels, dtype=bool)
    s = np.asarray(scores, dtype=float)
    if y.shape != s.shape:
        raise ValueError("labels and scores differ in length")
    return y, s


def auroc(labels: Sequence[bool], scores: Sequence[float]) -> float:
    """Probability that a random positive outscores a random negative (ties count half)."""
    y, s = _arrays(labels, scores)
    pos, neg = s[y], s[~y]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    sorted_s = s[order]
    i = 0
    while i < len(s):  # average ranks over ties
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    u = ranks[y].sum() - len(pos) * (len(pos) + 1) / 2
    return float(u / (len(pos) * len(neg)))


def operating_points(labels: Sequence[bool], scores: Sequence[float]) -> list[dict[str, float]]:
    """One point per distinct threshold: flag every claim scoring at least that much."""
    y, s = _arrays(labels, scores)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    points = []
    for t in sorted(set(s.tolist()), reverse=True):
        flagged = s >= t
        tp, fp = int((flagged & y).sum()), int((flagged & ~y).sum())
        points.append(
            {
                "threshold": t,
                "recall": tp / n_pos if n_pos else float("nan"),
                "precision": tp / (tp + fp) if tp + fp else float("nan"),
                "false_flag": fp / n_neg if n_neg else float("nan"),
            }
        )
    return points


def average_precision(labels: Sequence[bool], scores: Sequence[float]) -> float:
    """Area under the step precision–recall curve (sum of precision × recall increments)."""
    points = operating_points(labels, scores)
    if not points or np.isnan(points[0]["recall"]):
        return float("nan")
    ap, prev_recall = 0.0, 0.0
    for p in points:
        ap += (p["recall"] - prev_recall) * p["precision"]
        prev_recall = p["recall"]
    return float(ap)


def recall_at_false_flag(labels: Sequence[bool], scores: Sequence[float], max_rate: float) -> float:
    """Highest recall among thresholds whose false-flag rate is at most ``max_rate``."""
    best = 0.0
    for p in operating_points(labels, scores):
        if p["false_flag"] <= max_rate + 1e-12:
            best = max(best, p["recall"])
    return float(best)
