"""Object-level comparison of an imagined view with the real view of the same camera.

Both views are described by per-label instance detections. A label counts as
*present* in a view if at least one instance is detected. Comparing presence:

- matched   present in both
- invented  present in the imagined view only (the world model made it up)
- missing   present in the real view only (the world model left it out)

For matched labels with exactly one instance in each view, the position check
compares mask centroids on a coarse grid.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Obj:
    """Minimal detection record: label and centroid in pixels."""

    label: str
    cx: float
    cy: float


@dataclass
class ViewComparison:
    matched: list[str] = field(default_factory=list)
    invented: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    position_checked: int = 0
    position_agree: int = 0


def grid_cell(cx: float, cy: float, width: int, height: int, grid: int = 3) -> tuple[int, int]:
    return min(grid - 1, int(cx * grid / width)), min(grid - 1, int(cy * grid / height))


def compare_view(
    real: list[Obj], imagined: list[Obj], width: int, height: int, grid: int = 3
) -> ViewComparison:
    by_real: dict[str, list[Obj]] = defaultdict(list)
    by_imag: dict[str, list[Obj]] = defaultdict(list)
    for o in real:
        by_real[o.label].append(o)
    for o in imagined:
        by_imag[o.label].append(o)
    out = ViewComparison()
    for label in sorted(set(by_real) | set(by_imag)):
        if label in by_real and label in by_imag:
            out.matched.append(label)
            if len(by_real[label]) == 1 and len(by_imag[label]) == 1:
                r, i = by_real[label][0], by_imag[label][0]
                out.position_checked += 1
                same = grid_cell(r.cx, r.cy, width, height, grid) == grid_cell(
                    i.cx, i.cy, width, height, grid
                )
                out.position_agree += int(same)
        elif label in by_imag:
            out.invented.append(label)
        else:
            out.missing.append(label)
    return out


def rates(comparisons: list[ViewComparison]) -> dict[str, float]:
    """Pooled rates over views.

    ``invented_rate``  of the object labels the imagined views show, the share not in the real view
    ``missing_rate``   of the object labels the real views show, the share absent from the imagined
    ``position_agreement``  of matched single-instance labels, the share in the same grid cell
    """
    n_matched = sum(len(c.matched) for c in comparisons)
    n_invented = sum(len(c.invented) for c in comparisons)
    n_missing = sum(len(c.missing) for c in comparisons)
    checked = sum(c.position_checked for c in comparisons)
    agree = sum(c.position_agree for c in comparisons)

    def div(a: int, b: int) -> float:
        return a / b if b else float("nan")

    return {
        "n_views": len(comparisons),
        "n_imagined_labels": n_matched + n_invented,
        "n_real_labels": n_matched + n_missing,
        "invented_rate": div(n_invented, n_matched + n_invented),
        "missing_rate": div(n_missing, n_matched + n_missing),
        "position_agreement": div(agree, checked),
        "n_position_checked": checked,
    }


def flag_metrics(invented: list[bool], flagged: list[bool]) -> dict[str, float]:
    """Precision, recall and F1 of a method that flags claims as invented.

    Undefined ratios (nothing flagged, or nothing invented) are NaN.
    """
    if len(invented) != len(flagged):
        raise ValueError("length mismatch")
    tp = sum(i and f for i, f in zip(invented, flagged, strict=True))
    fp = sum((not i) and f for i, f in zip(invented, flagged, strict=True))
    fn = sum(i and (not f) for i, f in zip(invented, flagged, strict=True))
    tn = len(invented) - tp - fp - fn
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else float("nan")
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def cohen_kappa(a: list[bool], b: list[bool]) -> float:
    """Chance-corrected agreement between two binary raters."""
    if len(a) != len(b) or not a:
        raise ValueError("need two equal-length, non-empty lists")
    n = len(a)
    agree = sum(x == y for x, y in zip(a, b, strict=True)) / n
    pa, pb = sum(a) / n, sum(b) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    return float("nan") if expected == 1 else (agree - expected) / (1 - expected)
