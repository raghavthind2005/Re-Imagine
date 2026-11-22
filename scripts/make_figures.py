"""Figures for the README, generated from raw records and human labels.

    python scripts/make_figures.py            # all figures, into results/figures/

1. `example_panel.png` — imagined view vs real held-out frame for one dev ARKitScenes view,
   with the map's coverage shaded and each claimed object marked flagged or kept by the map
   check. The view is chosen by rule: the dev ARKitScenes view with the most human-confirmed
   hallucinations. Boxes come from re-running SAM 3 (CPU) on the claimed labels only.
   Needs the dev views (`runs/_kaggle/hallucination_v2/...`) and SAM 3 weights; skipped otherwise.
2. `hallucination_by_coverage.png` — left: share of imagined objects SAM 3 finds absent from
   the real frame, by map coverage of the view (dev and test; includes detector errors);
   right: where human-confirmed hallucinations and real objects sit, by how much of the
   object's own region the map covers (dev + test human labels, sampling-weighted).
3. `method_comparison.png` — test, human labels: hallucinations caught (exact 95% CI) and
   weighted false-flag rate on real objects (scene bootstrap 95% CI) per method.

Only ARKitScenes imagery is drawn (non-commercial licence, see docs/LICENSES.md).
"""

# ruff: noqa: E501
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from hallucination_report import ratio, scene_boot
from human_label_extra import false_flag, labelled, scene_ci
from human_label_report import clopper_pearson

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results/figures"
COLORS = {"b": "#1d4ed8", "c2": "#b45309", "cov": "#0f766e", "d": "#6b7280"}


def plt_setup():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    return plt


def method_flags(t_cov: float):
    return [
        ("ViSA-style self-check (d)", "d", lambda r: r["p_no_imagined"] > 0.5),
        (
            "Map + VLM with observed frames (c′)",
            "c2",
            lambda r: r["verdict"] == "contradicted"
            or (r["verdict"] == "no_coverage" and r["p_no_context"] > 0.5),
        ),
        (
            f"Coverage only, region < {t_cov:g} (exploratory)",
            "cov",
            lambda r: r["region_covered"] < t_cov,
        ),
        ("Map check (b)", "b", lambda r: r["verdict"] != "supported"),
    ]


def views(records: Path) -> list[dict]:
    out = []
    for f in sorted(records.glob("views*.jsonl")):
        out += [json.loads(line) for line in f.read_text().splitlines()]
    for v in out:
        v["n_inv"], v["n_imag"] = len(v["invented"]), len(v["imagined_labels"])
    return out


def fig_coverage(dev, test) -> None:
    plt = plt_setup()
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.5, 3.4), dpi=150)
    bins = [
        ("< 20%", lambda v: v["geo_coverage"] < 0.2),
        ("20–60%", lambda v: 0.2 <= v["geo_coverage"] < 0.6),
        ("≥ 60%", lambda v: v["geo_coverage"] >= 0.6),
    ]
    for k, (name, recs, color) in enumerate(
        (
            ("dev", ROOT / "results/hallucination_v2", "#64748b"),
            ("test", ROOT / "results/hallucination_test", "#1d4ed8"),
        )
    ):
        vs = views(recs)
        held = [v for v in vs if not v["is_control"]]
        pts = [scene_boot([v for v in held if f(v)], ratio("n_inv", "n_imag")) for _, f in bins]
        x = np.arange(len(bins)) + (k - 0.5) * 0.3
        y = [100 * p for p, _ in pts]
        err = np.array([[100 * (p - ci[0]), 100 * (ci[1] - p)] for p, ci in pts]).T
        a.bar(x, y, 0.3, yerr=err, color=color, capsize=3, label=f"{name}, held-out cameras")
        floor = 100 * ratio("n_inv", "n_imag")([v for v in vs if v["is_control"]])
        a.axhline(
            floor, color=color, ls=":", lw=1, label=f"{name}, control cameras (floor) {floor:.0f}%"
        )
    a.set_xticks(range(len(bins)), [n for n, _ in bins])
    a.set_xlabel("Share of the view the map of observed frames covers")
    a.set_ylabel("Imagined objects absent from real frame (%)")
    a.set_title("SAM 3 count (includes detector errors)", fontsize=9)
    a.legend(fontsize=7, frameon=False)

    pool = [r for r in dev + test if r["category"] in ("hallucination", "real_object")]
    rbins = [
        ("< 10%", lambda r: r["region_covered"] < 0.1),
        ("10–50%", lambda r: 0.1 <= r["region_covered"] < 0.5),
        ("≥ 50%", lambda r: r["region_covered"] >= 0.5),
    ]
    for k, (cat, label, color) in enumerate(
        (("hallucination", "Hallucinations", "#dc2626"), ("real_object", "Real objects", "#16a34a"))
    ):
        rs = [r for r in pool if r["category"] == cat]
        w = sum(r["weight"] for r in rs)
        y = [100 * sum(r["weight"] for r in rs if f(r)) / w for _, f in rbins]
        b.bar(
            np.arange(len(rbins)) + (k - 0.5) * 0.3,
            y,
            0.3,
            color=color,
            label=f"{label} (n = {len(rs)})",
        )
    b.set_xticks(range(len(rbins)), [n for n, _ in rbins])
    b.set_xlabel("Share of the object's own region the map covers")
    b.set_ylabel("Share of objects (%, sampling-weighted)")
    b.set_title("Human labels, dev + test (objects drawn in the imagined view)", fontsize=9)
    b.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "hallucination_by_coverage.png")


def fig_methods(test, t_cov) -> None:
    plt = plt_setup()
    hal = [r for r in test if r["category"] == "hallucination"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.5, 3.0), dpi=150)
    names = []
    for i, (name, key, flag) in enumerate(method_flags(t_cov)):
        k = sum(flag(r) for r in hal)
        lo, hi = clopper_pearson(k, len(hal))
        a.barh(
            i,
            100 * k / len(hal),
            xerr=[[100 * (k / len(hal) - lo)], [100 * (hi - k / len(hal))]],
            color=COLORS[key],
            capsize=3,
        )
        a.text(100 * hi + 2, i, f"{k}/{len(hal)}", va="center", fontsize=8)
        p, ci = scene_ci(test, false_flag(flag))
        b.barh(
            i,
            100 * p,
            xerr=[[100 * (p - ci[0])], [100 * (ci[1] - p)]],
            color=COLORS[key],
            capsize=3,
        )
        names.append(name)
    for ax in (a, b):
        ax.set_yticks(range(len(names)), names if ax is a else [""] * len(names))
        ax.set_xlim(0, 105)
    a.set_xlabel("Hallucinations caught (%)")
    b.set_xlabel("Real objects falsely flagged (%, weighted)")
    a.set_title(f"Test, human labels: {len(hal)} genuine hallucinations", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "method_comparison.png")


def fig_example(dev, views_dir: Path, weights: str) -> None:
    plt = plt_setup()
    import matplotlib.patches as patches

    from mindseye.geometry.sam3_model import SAM3Runner

    per_view = defaultdict(list)
    for r in dev:
        per_view[(r["scene"], r["target"])].append(r)
    ark = [k for k in sorted(per_view) if k[0].startswith("arkitscenes")]
    scene, t = max(
        ark,
        key=lambda k: (
            sum(r["category"] == "hallucination" for r in per_view[k]),
            [-ord(c) for c in k[0]],
        ),
    )
    rows = sorted(per_view[(scene, t)], key=lambda r: r["label"])
    v = np.load(views_dir / scene / "views.npz")
    imag, real, mask = v["imagined"][t], v["real"][t], v["geo_mask"][t].astype(bool)
    sam = SAM3Runner(weights, device="cpu", dtype="float32")
    dets = sam.detect(imag, [r["label"] for r in rows], threshold=0.5)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), dpi=150)
    shaded = imag.astype(float) / 255
    shaded[~mask] = 0.35 * shaded[~mask] + 0.65 * np.array([0.10, 0.10, 0.12])
    for ax, img, title in zip(
        axes,
        (imag, shaded, real),
        (
            "Imagined view (Stable Virtual Camera)",
            "Dark: region the observed frames never saw",
            "Real held-out frame (ground truth only)",
        ),
        strict=True,
    ):
        ax.imshow(img)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    names = {
        "hallucination": "hallucination",
        "real_object": "real",
        "detector_error": "detector error",
    }
    for k, r in enumerate(rows):
        flagged = r["verdict"] != "supported"
        color = "#f59e0b" if flagged else "#16a34a"
        for d in (d for d in dets if d.label == r["label"]):
            x0, y0, x1, y1 = d.box
            axes[0].add_patch(
                patches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=color, lw=1.5)
            )
        # One tag per object type, stacked in the top-left corner so none overlap.
        axes[0].text(6, 18 + 22 * k, f"{r['label']}: {'flagged' if flagged else 'kept'} ({names[r['category']]})",
                     color="black", fontsize=6.5, bbox={"facecolor": color, "pad": 1, "lw": 0})  # fmt: skip
    fig.suptitle(
        f"{scene.replace('__', '/')}, held-out camera {t}; box tags: map-check verdict (human label); amber = flagged, green = kept",
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(FIG / "example_panel.png")
    print("example:", scene, t, [(r["label"], r["category"], r["verdict"]) for r in rows])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dev-views", default=str(ROOT / "runs/_kaggle/hallucination_v2/runs/heldout_imagine")
    )
    ap.add_argument("--sam3-weights", default=str(ROOT / "cache/sam3_weights"))
    args = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
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
    t_cov = json.loads((ROOT / "results/human_label_extra_choices.json").read_text())[
        "coverage_threshold"
    ]
    fig_coverage(dev, test)
    fig_methods(test, t_cov)
    if Path(args.dev_views).exists() and Path(args.sam3_weights).exists():
        fig_example(dev, Path(args.dev_views), args.sam3_weights)
    else:
        print("dev views or SAM 3 weights not found: example panel not regenerated")


if __name__ == "__main__":
    main()
