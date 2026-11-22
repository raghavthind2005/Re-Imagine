"""Create the fixed dev subsets used by every imagination-based mode. Run once.

    python scripts/prepare_subsets.py

VSI-Bench: 200 dev questions. Every task type is present; the four tasks where
the sparse baseline is weakest get 30 questions each, the other six share 80.
SAT: all dev perspective questions (the MindJourney target task).
The file is committed and the writer refuses to overwrite it.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from mindseye.data.sat import read_question_types
from mindseye.data.splits import load_split, save_split
from mindseye.data.vsibench import load_vsibench

SEED = 20261013
WEAK = [
    "obj_appearance_order",
    "route_planning",
    "object_abs_distance",
    "object_rel_direction_hard",
]
N_WEAK_EACH = 30
N_TOTAL = 200


def main() -> None:
    scenes = load_split("data/splits/vsibench.json")["dev_scenes"]
    dev = load_vsibench("cache/hf/VSI-Bench/test.jsonl", "dev", "cache/vsibench_videos", scenes)
    by_type: dict[str, list[str]] = {}
    for ex in dev:
        by_type.setdefault(ex.meta["question_type"], []).append(ex.id)
    others = sorted(t for t in by_type if t not in WEAK)
    n_other = N_TOTAL - N_WEAK_EACH * len(WEAK)
    quota = {t: N_WEAK_EACH for t in WEAK}
    for i, t in enumerate(others):
        quota[t] = n_other // len(others) + (1 if i < n_other % len(others) else 0)
    rng = np.random.default_rng(SEED)
    chosen: list[str] = []
    for t in sorted(quota):
        ids = sorted(by_type[t])
        chosen += [ids[i] for i in sorted(rng.permutation(len(ids))[: quota[t]])]

    sat_rows = load_split("data/splits/sat.json")["dev_val_rows"]
    types = read_question_types("cache/hf/SAT/SAT_val.parquet")
    perspective = [f"sat-val-{r}" for r in sat_rows if types[r] == "perspective"]

    id_to_type = {ex.id: ex.meta["question_type"] for ex in dev}
    id_to_scene = {ex.id: ex.meta["scene"] for ex in dev}
    save_split(
        "data/splits/imagine_subset.json",
        {"vsi_ids": chosen, "sat_perspective_ids": perspective},
        {
            "seed": SEED,
            "vsi_by_type": dict(Counter(id_to_type[i] for i in chosen)),
            "vsi_n_scenes": len({id_to_scene[i] for i in chosen}),
            "n_sat_perspective": len(perspective),
        },
    )
    print("wrote data/splits/imagine_subset.json")


if __name__ == "__main__":
    main()
