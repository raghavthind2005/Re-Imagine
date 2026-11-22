"""A fresh blind sheet for an independent labeller: the 14 test hallucinations + 40 others.

    python scripts/make_independent_sheet.py --test-views runs/_kaggle/hallucination_test/runs/heldout_imagine

The 14 claims the primary labeller judged genuine hallucinations, plus 40 claims drawn at
random (seed 20261017) from the other test claims on the primary sheet, shuffled. No answers,
model outputs or groups are shown. The sample is saved once to
`data/labels/independent_labeller_sample.json`; the sheet goes to `labelling/` (gitignored).
"""

# ruff: noqa: E501
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from make_test_label_sheet import write_sheet

from mindseye.data.splits import save_split

ROOT = Path(__file__).resolve().parents[1]
SEED, N_OTHER = 20261017, 40


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-views", required=True)
    ap.add_argument("--sample", default=str(ROOT / "data/labels/independent_labeller_sample.json"))
    ap.add_argument("--out", default=str(ROOT / "labelling/independent_labeller_sheet.html"))
    args = ap.parse_args()
    path = Path(args.sample)
    if not path.exists():
        test = json.loads((ROOT / "data/labels/test_claim_sample.json").read_text())["split"]
        keys = sorted(c["key"] for c in test["claims"] if c["group"] != "dev_repeat")
        with open(ROOT / "data/labels/human_labels_test.csv", newline="") as f:
            lab = {r["key"]: r for r in csv.DictReader(f)}
        hal = [
            k for k in keys
            if lab[k]["visible_in_imagined"] == "yes" and lab[k]["visible_in_real"] == "no"
        ]  # fmt: skip
        rest = [k for k in keys if k not in hal]
        rng = np.random.default_rng(SEED)
        others = [rest[i] for i in sorted(rng.permutation(len(rest))[:N_OTHER])]
        order = [(hal + others)[i] for i in rng.permutation(len(hal) + len(others))]
        save_split(path, {"order": order, "hallucinations": hal, "others": others},
                   {"seed": SEED, "source": "primary test labels, data/labels/human_labels_test.csv"})  # fmt: skip
    order = json.loads(path.read_text())["split"]["order"]
    views = {k.split("|")[0]: Path(args.test_views) for k in order}
    write_sheet(
        order, views, Path(args.out), "independent-labeller-labels", "human_labels_independent.csv"
    )


if __name__ == "__main__":
    main()
