"""Pre-registered human-label sample on test (docs/PREREGISTRATION.md, section 4) and its sheets.

    python scripts/make_test_label_sheet.py --test-records results/hallucination_test \\
        --test-views runs/_kaggle/hallucination_test/runs/heldout_imagine \\
        --dev-views runs/_kaggle/hallucination_v2/runs/heldout_imagine

Draws the sample once (seed 20261016) into `data/labels/test_claim_sample.json`:
every claim SAM 3 marks invented, 40 random claims it marks real, and 30 random dev-sheet
claims as hidden repeats; plus a 50-claim subset of the test claims for a second labeller.
Writes two blind sheets under `labelling/` (gitignored: they hold benchmark frames).
"""

# ruff: noqa: E501
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np
from make_label_sheet import PAGE, jpeg_b64, radios

from mindseye.data.splits import save_split

SEED = 20261016
N_REAL, N_REPEAT, N_SECOND = 40, 30, 50


def test_claims(records: Path) -> list[dict]:
    claims = []
    for f in sorted(records.glob("views*.jsonl")):
        for line in f.read_text().splitlines():
            v = json.loads(line)
            if v["is_control"]:
                continue
            for c in v["claims"]:
                claims.append(
                    {
                        "key": f"{v['scene']}|{v['target']}|{c['label']}",
                        "sam3_invented": not c["true_in_real_view"],
                    }
                )
    return sorted(claims, key=lambda c: c["key"])


def draw(records: Path, dev_sample: Path) -> dict:
    rng = np.random.default_rng(SEED)
    claims = test_claims(records)
    invented = [c for c in claims if c["sam3_invented"]]
    real = [c for c in claims if not c["sam3_invented"]]
    real_pick = [real[i] for i in sorted(rng.permutation(len(real))[:N_REAL])]
    test = [{**c, "group": "sam3_invented_census", "weight": 1.0} for c in invented]
    test += [{**c, "group": "sam3_real_sample", "weight": len(real) / N_REAL} for c in real_pick]
    dev = sorted(c["key"] for c in json.loads(dev_sample.read_text())["split"]["claims"])
    repeats = [
        {"key": dev[i], "group": "dev_repeat", "weight": 0.0}
        for i in sorted(rng.permutation(len(dev))[:N_REPEAT])
    ]
    sheet = test + repeats
    order = [sheet[i]["key"] for i in rng.permutation(len(sheet))]
    second = [test[i]["key"] for i in rng.permutation(len(test))[:N_SECOND]]
    return {
        "claims": sheet,
        "sheet_order": order,
        "second_labeller_order": second,
        "counts": {
            "sam3_invented": len(invented),
            "sam3_real": len(real),
            "sam3_real_sampled": N_REAL,
        },
    }


def write_sheet(
    keys: list[str], views: dict[str, Path], out: Path, store: str, filename: str
) -> None:
    cache: dict[str, dict] = {}
    items = []
    for i, key in enumerate(keys, 1):
        scene, target, label = key.split("|")
        if scene not in cache:
            cache.clear()
            cache[scene] = dict(np.load(views[scene] / scene / "views.npz"))
        v, t = cache[scene], int(target)
        items.append(
            f'<div class="item"><div><strong>{i}.</strong> Object: <strong>{html.escape(label)}</strong></div>'
            f'<div class="imgs"><figure><img alt="imagined view" src="data:image/jpeg;base64,{jpeg_b64(v["imagined"][t])}">'
            f'<figcaption>Imagined view</figcaption></figure><figure><img alt="real photo" '
            f'src="data:image/jpeg;base64,{jpeg_b64(v["real"][t])}"><figcaption>Real photo, same camera</figcaption></figure></div>'
            f'<div class="q">Visible in the <strong>real</strong> photo? {radios(key, "real")}</div>'
            f'<div class="q">Visible in the <strong>imagined</strong> view? {radios(key, "imagined")}</div></div>'
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        PAGE.format(items="\n".join(items), keys=json.dumps(keys), store=store, filename=filename)
    )
    print(f"{len(keys)} claims -> {out} ({out.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-records", required=True)
    ap.add_argument("--test-views", required=True)
    ap.add_argument("--dev-views", required=True)
    ap.add_argument("--dev-sample", default="data/labels/claim_sample.json")
    ap.add_argument("--sample", default="data/labels/test_claim_sample.json")
    ap.add_argument("--out-dir", default="labelling")
    args = ap.parse_args()
    path = Path(args.sample)
    if not path.exists():
        save_split(path, draw(Path(args.test_records), Path(args.dev_sample)),
                   {"seed": SEED, "preregistered": "docs/PREREGISTRATION.md, section 4"})  # fmt: skip
    sample = json.loads(path.read_text())["split"]
    test_scenes = {p.name for p in Path(args.test_views).iterdir() if p.is_dir()}
    views = {}
    for c in sample["claims"]:
        scene = c["key"].split("|")[0]
        views[scene] = Path(args.test_views) if scene in test_scenes else Path(args.dev_views)
    out = Path(args.out_dir)
    write_sheet(
        sample["sheet_order"],
        views,
        out / "test_claims_sheet.html",
        "test-claim-labels",
        "human_labels_test.csv",
    )
    write_sheet(
        sample["second_labeller_order"],
        views,
        out / "second_labeller_sheet.html",
        "second-labeller-labels",
        "human_labels_second.csv",
    )
    print(
        sample["counts"],
        "groups:",
        {
            g: sum(c["group"] == g for c in sample["claims"])
            for g in ("sam3_invented_census", "sam3_real_sample", "dev_repeat")
        },
    )


if __name__ == "__main__":
    main()
