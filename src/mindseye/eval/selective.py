"""Selective-prediction metrics: how wrong is the system when it claims to be sure?"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def confident_error_rate(correct: Sequence[bool], high_confidence: Sequence[bool]) -> float:
    """Fraction of high-confidence answers that are wrong. NaN if none are high-confidence."""
    c = np.asarray(correct, dtype=bool)
    h = np.asarray(high_confidence, dtype=bool)
    if c.shape != h.shape:
        raise ValueError("inputs must have the same length")
    return float((~c[h]).mean()) if h.any() else float("nan")


def coverage(answered: Sequence[bool]) -> float:
    """Fraction of questions answered rather than abstained on."""
    return float(np.asarray(answered, dtype=bool).mean())


def selective_risk(correct: Sequence[bool], answered: Sequence[bool]) -> float:
    """Error rate over the answered questions only. NaN if nothing was answered."""
    c = np.asarray(correct, dtype=bool)
    a = np.asarray(answered, dtype=bool)
    return float((~c[a]).mean()) if a.any() else float("nan")


def risk_coverage_curve(
    correct: Sequence[bool], confidence: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    """Risk at each coverage level when answering the most confident questions first.

    Returns ``(coverage, risk)`` arrays of length n: entry k-1 is the error rate
    over the k most confident answers. Ties in confidence are broken by treating
    tied items as a block (their average), so the curve does not depend on input
    order.
    """
    c = np.asarray(correct, dtype=float)
    conf = np.asarray(confidence, dtype=float)
    if c.shape != conf.shape or c.ndim != 1 or len(c) == 0:
        raise ValueError("need two non-empty 1-D arrays of the same length")
    order = np.argsort(-conf, kind="stable")
    err = 1.0 - c[order]
    sorted_conf = conf[order]
    # Replace each tied block's errors with the block mean.
    start = 0
    for end in range(1, len(err) + 1):
        if end == len(err) or sorted_conf[end] != sorted_conf[start]:
            err[start:end] = err[start:end].mean()
            start = end
    k = np.arange(1, len(err) + 1)
    return k / len(err), np.cumsum(err) / k


def aurc(correct: Sequence[bool], confidence: Sequence[float]) -> float:
    """Area under the risk-coverage curve (mean risk over all coverage levels). Lower is better."""
    _, risk = risk_coverage_curve(correct, confidence)
    return float(risk.mean())


def paired_bootstrap_diff(
    a: Sequence[float],
    b: Sequence[float],
    *,
    clusters: Sequence | None = None,
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    """Mean of ``a - b`` on the same examples, with a percentile bootstrap interval.

    With ``clusters`` whole clusters (e.g. scenes) are resampled together.
    """
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    if clusters is None:
        groups = [np.array([i]) for i in range(len(d))]
    else:
        by: dict = {}
        for i, cl in enumerate(clusters):
            by.setdefault(cl, []).append(i)
        groups = [np.array(v) for _, v in sorted(by.items(), key=lambda kv: str(kv[0]))]
    means = np.empty(n_resamples)
    for r in range(n_resamples):
        picks = rng.integers(0, len(groups), len(groups))
        means[r] = d[np.concatenate([groups[p] for p in picks])].mean()
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {"diff": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi), "n": len(d)}
