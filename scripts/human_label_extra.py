"""Exploratory baselines on the human-labelled test set, and dev tuning of a combined rule.

    python scripts/human_label_extra.py --out results/hallucination_human_extra.md

Not pre-registered (requested 2026-10-14 after the pre-registered test result). Thresholds
are chosen on the dev human labels only and applied unchanged to test.

Population weights: the dev sample has 25 claims per (SAM 3 invented/real x view coverage)
cell, so each dev claim is weighted by its cell's claim count / 25; the test sample weights
are those pre-registered (1 for SAM 3-invented claims, 265/40 for SAM 3-real claims).
"""

# ruff: noqa: E501
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from hallucination_curves import attach_scores
from hallucination_headtohead import load_claims
from human_label_report import clopper_pearson

from mindseye.eval.metrics import mcnemar_exact

ROOT = Path(__file__).resolve().parents[1]
COVERAGE_GRID = [round(0.1 * i, 1) for i in range(1, 10)]
TAU_GRID = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
EXIT_RECALL, EXIT_FALSE_FLAG = 0.70, 0.10


def cov_bin(x: float) -> str:
    return "low" if x < 0.2 else ("mid" if x < 0.6 else "high")


def labelled(records: Path, sample_path: Path, labels_path: Path, split: str) -> list[dict]:
    claims, _ = load_claims(records)
    claims = attach_scores(claims, records / "scores.jsonl")
    by_key = {f"{c['scene']}|{c['target']}|{c['label']}": c for c in claims}
    with open(labels_path, newline="") as f:
        labels = {r["key"]: r for r in csv.DictReader(f)}
    sample = json.loads(sample_path.read_text())["split"]["claims"]
    if split == "dev":
        cells = Counter(
            f"{'real' if c['true_in_real_view'] else 'invented'}/{cov_bin(c['view_coverage'])}"
            for c in claims
        )
        weight = {s["key"]: cells[s["stratum"]] / 25 for s in sample}
    else:
        weight = {s["key"]: s["weight"] for s in sample if s["group"] != "dev_repeat"}
    rows = []
    for key, w in weight.items():
        lab = labels.get(key, {})
        real, imag = lab.get("visible_in_real"), lab.get("visible_in_imagined")
        if real not in ("yes", "no") or imag not in ("yes", "no"):
            continue
        cat = (
            "detector_error"
            if imag == "no"
            else ("hallucination" if real == "no" else "real_object")
        )
        rows.append({**by_key[key], "weight": w, "category": cat, "absent_in_real": real == "no"})
    return rows


def wsum(rows, pred) -> float:
    return float(sum(r["weight"] for r in rows if pred(r)))


def scene_ci(rows, stat, n_boot=2000, seed=0):
    by = defaultdict(list)
    for r in rows:
        by[r["scene"]].append(r)
    groups = [by[k] for k in sorted(by)]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        v = stat([x for g in rng.integers(0, len(groups), len(groups)) for x in groups[g]])
        if not np.isnan(v):
            vals.append(v)
    p = stat(rows)
    if np.isnan(p) or len(vals) < 0.9 * n_boot:
        return p, None
    return p, tuple(np.quantile(vals, [0.025, 0.975]))


def pc(est) -> str:
    p, ci = est
    if np.isnan(p):
        return "n/a"
    return f"{100 * p:.0f}%" + (
        f" ({100 * ci[0]:.0f}–{100 * ci[1]:.0f})" if ci else " (interval n/a)"
    )


def recall(flag):
    return lambda rs: (lambda h: float(np.mean([flag(r) for r in h])) if h else float("nan"))(
        [r for r in rs if r["category"] == "hallucination"]
    )


def false_flag(flag):
    def stat(rs):
        real = [r for r in rs if r["category"] == "real_object"]
        return wsum(real, flag) / wsum(real, lambda r: True) if real else float("nan")

    return stat


def precision_genuine(flag):
    """Of flagged genuine imagined objects (weighted), the share that are hallucinations."""

    def stat(rs):
        g = [r for r in rs if r["category"] != "detector_error" and flag(r)]
        return (
            wsum(g, lambda r: r["category"] == "hallucination") / wsum(g, lambda r: True)
            if g
            else float("nan")
        )

    return stat


def precision_absent(flag):
    """Of all flagged claims (weighted), the share whose object is absent from the real frame."""

    def stat(rs):
        f = [r for r in rs if flag(r)]
        return (
            wsum(f, lambda r: r["absent_in_real"]) / wsum(f, lambda r: True) if f else float("nan")
        )

    return stat


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = labelled(
        ROOT / "results/hallucination_v2",
        ROOT / "data/labels/claim_sample.json",
        ROOT / "data/labels/human_labels.csv",
        "dev",
    )
    test = labelled(
        ROOT / "results/hallucination_test",
        ROOT / "data/labels/test_claim_sample.json",
        ROOT / "data/labels/human_labels_test.csv",
        "test",
    )

    # Dev: coverage-only threshold (best weighted F1 for hallucinations among genuine claims).
    genuine_dev = [r for r in dev if r["category"] != "detector_error"]

    def wf1(flag, rs):
        tp = wsum(rs, lambda r: r["category"] == "hallucination" and flag(r))
        fp = wsum(rs, lambda r: r["category"] == "real_object" and flag(r))
        fn = wsum(rs, lambda r: r["category"] == "hallucination" and not flag(r))
        return 2 * tp / (2 * tp + fp + fn) if tp else 0.0

    f1s = {t: wf1(lambda r, t=t: r["region_covered"] < t, genuine_dev) for t in COVERAGE_GRID}
    t_cov = max(COVERAGE_GRID, key=lambda t: (f1s[t], -t))

    def coverage_only(r, t=t_cov):
        return r["region_covered"] < t

    # Dev: combined rule grid.
    grid = []
    for tau in TAU_GRID:

        def comb(r, tau=tau):
            return r["verdict"] != "supported" and r["p_no_context"] >= tau

        grid.append((tau, recall(comb)(dev), false_flag(comb)(dev)))
    ok = [g for g in grid if g[1] >= EXIT_RECALL and g[2] <= EXIT_FALSE_FLAG]
    chosen = max(ok, key=lambda g: (g[1], -g[2])) if ok else None

    L = [
        "<!-- Generated by scripts/human_label_extra.py. Do not edit by hand. -->",
        "",
        "**Exploratory, not pre-registered.** Thresholds chosen on dev human labels only.",
        "",
        "## Dev: choices made before looking at test",
        "",
        f"Coverage-only baseline: flag a claim whose own image region the map covers less than *t*; "
        f"*t* = {t_cov} (best weighted F1 over {COVERAGE_GRID}; F1 by *t*: "
        + ", ".join(f"{t}: {100 * f:.0f}" for t, f in f1s.items())
        + ").",
        "",
        f"Combined rule: flag only when the map does not confirm the object AND Qwen (given the observed frames) "
        f'gives p("no") ≥ τ. Exit rule: some τ must keep recall ≥ {100 * EXIT_RECALL:.0f}% with weighted false-flag '
        f"rate ≤ {100 * EXIT_FALSE_FLAG:.0f}% on dev. Dev: {sum(r['category'] == 'hallucination' for r in dev)} hallucinations, "
        f"{sum(r['category'] == 'real_object' for r in dev)} real objects.",
        "",
        "| τ | Dev recall | Dev false-flag rate (weighted) | Meets both |",
        "|---|---|---|---|",
    ]
    for tau, rec, ffr in grid:
        L.append(
            f"| {tau} | {100 * rec:.0f}% | {100 * ffr:.0f}% | {'yes' if (rec >= EXIT_RECALL and ffr <= EXIT_FALSE_FLAG) else 'no'} |"
        )
    L += [
        "",
        (
            f"**Chosen τ: {chosen[0]}.**"
            if chosen
            else "**No τ meets the exit rule; the combined rule is not taken to test (stop, as instructed).**"
        ),
        "",
    ]

    methods = [
        ("(b) Map check (ours)", lambda r: r["verdict"] != "supported"),
        (
            "(c′) Map where covered, VLM with observed frames elsewhere (main baseline)",
            lambda r: r["verdict"] == "contradicted"
            or (r["verdict"] == "no_coverage" and r["p_no_context"] > 0.5),
        ),
        (f"Coverage-only (region covered < {t_cov})", coverage_only),
        ("Flag all", lambda r: True),
        (
            "(d) ViSA-style self-check (reference; cannot see that a drawn object is absent)",
            lambda r: r["p_no_imagined"] > 0.5,
        ),
    ]
    hal = [r for r in test if r["category"] == "hallucination"]
    L += [
        "## Test (human labels): baselines and precision",
        "",
        f"Genuine hallucinations: {len(hal)}; real objects: {sum(r['category'] == 'real_object' for r in test)}; "
        f"detector errors: {sum(r['category'] == 'detector_error' for r in test)}. Weighted rates use the pre-registered "
        "sampling weights; intervals are 95% scene-level bootstrap (exact binomial for recall).",
        "",
        "| Method | Recall on hallucinations | False-flag rate on real objects | Precision on genuine imagined objects | Precision against the real frame (incl. detector errors) |",
        "|---|---|---|---|---|",
    ]
    for name, flag in methods:
        k = sum(flag(r) for r in hal)
        lo, hi = clopper_pearson(k, len(hal))
        L.append(
            f"| {name} | {k} of {len(hal)}, {100 * k / len(hal):.0f}% ({100 * lo:.0f}–{100 * hi:.0f}) "
            f"| {pc(scene_ci(test, false_flag(flag)))} | {pc(scene_ci(test, precision_genuine(flag)))} | {pc(scene_ci(test, precision_absent(flag)))} |"
        )
    mc = mcnemar_exact([methods[0][1](r) for r in hal], [coverage_only(r) for r in hal])
    L += [
        "",
        f"Paired on the {len(hal)} hallucinations, (b) vs coverage-only: caught only by (b) {mc['only_a']}, only by coverage-only "
        f"{mc['only_b']}; exact McNemar p = {mc['p_value']:.3g}.",
        "",
        "Precision on genuine imagined objects: of the flagged claims whose object really is drawn in the imagined view, the "
        "weighted share that are hallucinations. Precision against the real frame: of all flagged claims, the weighted share "
        "whose object is absent from the real frame (a flagged detector error counts as correct here).",
    ]
    Path(args.out).write_text("\n".join(L) + "\n")
    json.dump(
        {"coverage_threshold": t_cov, "combined_tau": chosen[0] if chosen else None},
        open(ROOT / "results/human_label_extra_choices.json", "w"),
    )
    print("\n".join(L))


if __name__ == "__main__":
    main()
