"""The README's results table, generated from raw records and human labels.

    python scripts/make_summary.py          # rewrites the block between the summary markers

Every row says how much weight it can bear: "confirmed on test (pre-registered)",
"dev only", "exploratory" or "not run". Writes `results/summary.md` as well.
"""

# ruff: noqa: E501
from __future__ import annotations

import argparse
import json
from pathlib import Path

from hallucination_headtohead import load_claims, methods, metric
from hallucination_report import pct, ratio, scene_boot
from human_label_extra import false_flag, labelled, scene_ci
from human_label_report import clopper_pearson

from mindseye.eval.metrics import mcnemar_exact
from mindseye.eval.selective import paired_bootstrap_diff

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results/raw"
CONFIRMED, DEV, EXPLORATORY, NOT_RUN = (
    "confirmed on test (pre-registered)",
    "dev only",
    "exploratory",
    "not run",
)


def preds(name: str) -> dict:
    return {
        json.loads(x)["id"]: json.loads(x)
        for x in (RAW / name / "predictions.jsonl").read_text().splitlines()
    }


def paired(run: str, base: str, clustered: bool) -> str:
    r, b = preds(run), preds(base)
    ids = sorted(r)
    d = paired_bootstrap_diff(
        [r[i]["score"] for i in ids],
        [b[i]["score"] for i in ids],
        clusters=[r[i]["scene"] for i in ids] if clustered else None,
    )
    return f"{100 * d['diff']:+.1f} points (95% CI {100 * d['ci_low']:+.1f} to {100 * d['ci_high']:+.1f}), n = {len(ids)}"


def frac(k: int, n: int) -> str:
    lo, hi = clopper_pearson(k, n)
    return f"{k}/{n} ({100 * lo:.0f}–{100 * hi:.0f}%)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--readme", default=str(ROOT / "README.md"))
    args = ap.parse_args()
    rows = []

    # Imagination: imagine vs vlm_only.
    base_v, base_s = (
        "20261013-195423_p2_vlm_only_vsi_sub200_sparse8_448_qwen8b_v1",
        "20261013-182549_p2_vlm_only_sat_dev_circular_qwen8b_v1",
    )
    rows.append(
        (
            DEV,
            "`imagine` − `vlm_only`, VSI-Bench, MindJourney's evidence rule (helpfulness ≥ 8)",
            paired("p2_imagine_vsi_sub200_sparse8_qwen8b_v1-n100", base_v, True),
        )
    )
    rows.append(
        (
            DEV,
            "`imagine` − `vlm_only`, VSI-Bench, two imagined views always",
            paired("p2_imagine_vsi_sub200_sparse8_top2_qwen8b_v1-n100", base_v, True),
        )
    )
    rows.append(
        (
            DEV,
            "`imagine` − `vlm_only`, SAT perspective, strict circular, two imagined views always",
            paired("p2_imagine_sat_persp_circular_top2_qwen8b_v1", base_s, False),
        )
    )

    # Where the world model invents objects (SAM 3 count).
    for split, recs in (("test", ROOT / "results/hallucination_test"),):
        vs = []
        for f in sorted(recs.glob("views*.jsonl")):
            vs += [json.loads(line) for line in f.read_text().splitlines()]
        for v in vs:
            v["n_inv"], v["n_imag"] = len(v["invented"]), len(v["imagined_labels"])
        held = [v for v in vs if not v["is_control"]]
        lo_cov = scene_boot([v for v in held if v["geo_coverage"] < 0.2], ratio("n_inv", "n_imag"))
        hi_cov = scene_boot([v for v in held if v["geo_coverage"] >= 0.6], ratio("n_inv", "n_imag"))
        floor = ratio("n_inv", "n_imag")([v for v in vs if v["is_control"]])
        rows.append((CONFIRMED, f"Imagined objects SAM 3 finds absent from the real frame ({split}): views the map covers < 20% vs ≥ 60%",
                     f"{pct(lo_cov)} vs {pct(hi_cov)}; control floor {100 * floor:.0f}% (count includes detector errors)"))  # fmt: skip

    # Human labels.
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
    for split, rs, status in (("test", test, CONFIRMED), ("dev", dev, DEV)):
        inv = [r for r in rs if not r["true_in_real_view"]]
        k = sum(r["category"] == "detector_error" for r in inv)
        h = sum(r["category"] == "hallucination" for r in inv)
        rows.append((status, f"Claims SAM 3 marks invented that are detector errors / genuine hallucinations ({split}, human labels)",
                     f"{k} / {h} of {len(inv)}"))  # fmt: skip
    t_cov = json.loads((ROOT / "results/human_label_extra_choices.json").read_text())[
        "coverage_threshold"
    ]
    hal = [r for r in test if r["category"] == "hallucination"]
    flags = {
        "b": lambda r: r["verdict"] != "supported",
        "c2": lambda r: r["verdict"] == "contradicted"
        or (r["verdict"] == "no_coverage" and r["p_no_context"] > 0.5),
        "d": lambda r: r["p_no_imagined"] > 0.5,
        "cov": lambda r: r["region_covered"] < t_cov,
    }
    caught = {k: [f(r) for r in hal] for k, f in flags.items()}
    p_bd = mcnemar_exact(caught["b"], caught["d"])["p_value"]
    p_bc = mcnemar_exact(caught["b"], caught["c2"])["p_value"]
    p_bcov = mcnemar_exact(caught["b"], caught["cov"])["p_value"]
    n = len(hal)
    rows.append((CONFIRMED, "Hallucinations caught: VLM looking at the imagined frame only (ViSA-style, (d)) vs map check (b)",
                 f"{frac(sum(caught['d']), n)} vs {frac(sum(caught['b']), n)}; exact McNemar p = {p_bd:.3f}"))  # fmt: skip
    rows.append((CONFIRMED, "Hallucinations caught: VLM with the observed frames where the map is blind ((c′), main baseline) vs map check (b)",
                 f"{frac(sum(caught['c2']), n)} vs {frac(sum(caught['b']), n)}; exact McNemar p = {p_bc:.3f}"))  # fmt: skip
    ff = {k: scene_ci(test, false_flag(f)) for k, f in flags.items()}
    rows.append((CONFIRMED, "Real objects falsely flagged (weighted): (b) / (c′) / (d)",
                 " / ".join(f"{100 * p:.0f}% ({100 * ci[0]:.0f}–{100 * ci[1]:.0f})" for p, ci in (ff["b"], ff["c2"], ff["d"]))))  # fmt: skip
    rows.append((EXPLORATORY, f"Coverage test alone (flag objects whose region the map covers < {100 * t_cov:.0f}%; threshold from dev): hallucinations caught, false flags",
                 f"{frac(sum(caught['cov']), n)}, {100 * ff['cov'][0]:.0f}% ({100 * ff['cov'][1][0]:.0f}–{100 * ff['cov'][1][1]:.0f}); vs (b) paired p = {p_bcov:.2f}"))  # fmt: skip

    # Automatic-truth head-to-head (pre-registered, not met).
    claims, _ = load_claims(ROOT / "results/hallucination_test")
    sub = [
        c for c in claims if c["vlm_real"] is not None and c["vlm_real"] == c["true_in_real_view"]
    ]

    def inv(c):
        return not c["true_in_real_view"]

    m = dict(methods(0.3))
    d = scene_boot(
        sub,
        lambda r: metric(m["(b) Geometry + no-coverage rule"], "f1", inv)(r)
        - metric(m["(d) ViSA-style self-check"], "f1", inv)(r),
    )
    rows.append(
        (
            CONFIRMED + ": **not met**",
            "F1 of (b) − (d) against automatic ground truth (SAM 3 and Qwen agree; dominated by detector errors)",
            f"{pct(d, True)} points; criterion needed the interval to exclude 0",
        )
    )
    rows.append(
        (
            DEV,
            "Combined rule ((b) and VLM doubt) to cut false flags: some threshold with recall ≥ 70% and false flags ≤ 10%",
            "none found; not taken to test",
        )
    )
    down = ROOT / "results/downstream.md"
    rows.append(
        (
            EXPLORATORY if down.exists() else NOT_RUN,
            "Downstream: answer accuracy with imagined views filtered by (b)",
            "see `results/downstream.md`" if down.exists() else "optional run not done",
        )
    )

    L = [
        "<!-- Generated by scripts/make_summary.py. Do not edit by hand. -->",
        "",
        "| Status | Result | Value |",
        "|---|---|---|",
    ]
    L += [f"| {s} | {r} | {v} |" for s, r, v in rows]
    text = "\n".join(L) + "\n"
    (ROOT / "results/summary.md").write_text(text)
    readme = Path(args.readme)
    if readme.exists():
        s = readme.read_text()
        a, b = "<!-- summary:start -->", "<!-- summary:end -->"
        if a in s and b in s:
            readme.write_text(s[: s.index(a) + len(a)] + "\n" + text + s[s.index(b) :])
    print(text)


if __name__ == "__main__":
    main()
