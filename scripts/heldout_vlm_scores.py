"""Continuous VLM scores for the hallucination-flagging methods (d) and (c′).

    python scripts/heldout_vlm_scores.py --views runs/heldout_imagine \\
        --objects runs/heldout_objects --out runs/heldout_vlm

For every object claim on a held-out view (each label the imagined view shows), Qwen's
probability of answering "no" to the same prompts as `heldout_vlm_judge.py`:

- `p_no_imagined`  the imagined frame alone (ViSA-style self-check, method (d))
- `p_no_context`   the observed real frames plus the imagined frame (used by (c′))

Writes `scores.jsonl`. Prompts are unchanged (`presence-v1`), so the greedy answers in
`judgements.jsonl` / `context.jsonl` should match `p_no > 0.5` almost always.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mindseye.utils.cache import DiskCache
from mindseye.utils.runlog import append_jsonl, completed_ids
from mindseye.vlm.prompts import presence
from mindseye.vlm.qwen_local import QwenLocal


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", required=True)
    ap.add_argument("--objects", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-id", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--cache-dir", default="cache/vlm")
    args = ap.parse_args()

    rows = []
    for f in sorted(Path(args.objects).glob("views*.jsonl")):
        rows += [json.loads(line) for line in f.read_text().splitlines()]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    out_file = out / "scores.jsonl"
    done = completed_ids(out_file, key="key")
    vlm = QwenLocal(args.model_id, cache=DiskCache(args.cache_dir))
    current, v = None, None
    for r in rows:
        if r["is_control"]:
            continue
        if r["scene"] != current:
            current, v = r["scene"], dict(np.load(Path(args.views) / r["scene"] / "views.npz"))
        t = r["target"]
        observed = list(v["observed"])
        for label in r["imagined_labels"]:
            key = f"{r['scene']}|{t}|{label}"
            if key in done:
                continue
            imag = vlm.yes_no(
                [v["imagined"][t]], presence.single(label), prompt_version=presence.VERSION
            )
            ctx = vlm.yes_no(
                observed + [v["imagined"][t]],
                presence.with_context(label, len(observed)),
                prompt_version=presence.VERSION,
            )
            record = {
                "key": key,
                "scene": r["scene"],
                "target": t,
                "label": label,
                "p_no_imagined": imag["p_no"],
                "mass_imagined": imag["mass"],
                "p_no_context": ctx["p_no"],
                "mass_context": ctx["mass"],
            }
            append_jsonl(out_file, record)
            print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
