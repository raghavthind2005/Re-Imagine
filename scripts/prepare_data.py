"""Create the fixed dev/test splits. Run once; the outputs are committed.

    python scripts/prepare_data.py --sat-val cache/hf/SAT/SAT_val.parquet \
        --vsi-jsonl cache/hf/VSI-Bench/test.jsonl

SAT:      dev  = 500 rows of SAT val (synthesized), stratified by question type
          test = all 150 rows of SAT test (SAT-Real)
VSI-Bench: scenes are split, not questions, so no video is on both sides;
          stratified by source dataset. dev = 25% of scenes, test = 75%.
"""

from __future__ import annotations

import argparse
from collections import Counter

from mindseye.data.sat import read_question_types
from mindseye.data.splits import save_split, split_groups, stratified_sample
from mindseye.data.vsibench import read_rows, scene_key

SEED = 20261013
SAT_DEV_SIZE = 500
VSI_DEV_FRACTION = 0.25


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sat-val", required=True)
    ap.add_argument("--vsi-jsonl", required=True)
    ap.add_argument("--out-dir", default="data/splits")
    args = ap.parse_args()

    types = read_question_types(args.sat_val)
    dev_rows = sorted(stratified_sample(list(range(len(types))), types, SAT_DEV_SIZE, SEED))
    save_split(
        f"{args.out_dir}/sat.json",
        {"dev_val_rows": dev_rows},
        {
            "seed": SEED,
            "source": "array/SAT SAT_val.parquet (dev), SAT_test.parquet all rows (test)",
            "dev_by_type": dict(Counter(types[i] for i in dev_rows)),
        },
    )

    rows = read_rows(args.vsi_jsonl)
    scenes = [scene_key(r["dataset"], r["scene_name"]) for r in rows]
    dev, test = split_groups(scenes, [r["dataset"] for r in rows], VSI_DEV_FRACTION, SEED)
    dev_set = set(dev)
    save_split(
        f"{args.out_dir}/vsibench.json",
        {"dev_scenes": dev, "test_scenes": test},
        {
            "seed": SEED,
            "source": "nyu-visionx/VSI-Bench test.jsonl",
            "dev_fraction_of_scenes": VSI_DEV_FRACTION,
            "n_dev_questions": sum(s in dev_set for s in scenes),
            "n_test_questions": sum(s not in dev_set for s in scenes),
        },
    )
    print("wrote", args.out_dir)


if __name__ == "__main__":
    main()
