"""VLM judgements of object presence on real and imagined held-out views.

    python scripts/heldout_vlm_judge.py --views runs/heldout_imagine \\
        --objects runs/heldout_objects --out runs/heldout_vlm

For every view and every object label that SAM 3 found in the real or the
imagined view, one row with:

- `vlm_real`      the VLM's answer on the real frame (a second ground truth)
- `vlm_imagined`  the VLM's answer on the imagined frame alone (ViSA-style
                  self-verification of the claim "this object is visible here")

A second pass writes `context.jsonl`: for held-out views and the labels the
imagined view shows, the VLM's answer given the observed real frames plus the
imagined frame.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mindseye.utils.cache import DiskCache
from mindseye.utils.runlog import append_jsonl
from mindseye.vlm.prompts import presence
from mindseye.vlm.qwen_local import QwenLocal


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", required=True)
    ap.add_argument("--objects", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-id", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--cache-dir", default="cache/vlm")
    ap.add_argument("--no-context", action="store_true")
    args = ap.parse_args()

    rows = []
    for f in sorted(Path(args.objects).glob("views*.jsonl")):
        rows += [json.loads(line) for line in f.read_text().splitlines()]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def done_keys(path: Path) -> set:
        if not path.exists():
            return set()
        recs = [json.loads(line) for line in path.read_text().splitlines()]
        return {(r["scene"], r["target"], r["label"]) for r in recs}

    vlm = QwenLocal(args.model_id, cache=DiskCache(args.cache_dir))

    def ask(images: list[np.ndarray], prompt: str) -> tuple[bool | None, str]:
        resp = vlm.generate(images, prompt, prompt_version=presence.VERSION, max_tokens=4)
        return presence.parse_yes_no(resp.text), resp.text

    def scene_views(scene: str, cache: dict) -> dict:
        if scene not in cache:
            cache.clear()
            cache[scene] = dict(np.load(Path(args.views) / scene / "views.npz"))
        return cache[scene]

    # Pass 1: single-image judgements on the real and the imagined frame.
    out_file, cache = out / "judgements.jsonl", {}
    done = done_keys(out_file)
    for r in rows:
        v, t = scene_views(r["scene"], cache), r["target"]
        for label in sorted(set(r["real_labels"]) | set(r["imagined_labels"])):
            if (r["scene"], t, label) in done:
                continue
            real, real_text = ask([v["real"][t]], presence.single(label))
            imag, imag_text = ask([v["imagined"][t]], presence.single(label))
            record = {
                "scene": r["scene"],
                "target": t,
                "is_control": r["is_control"],
                "label": label,
                "sam_real": label in r["real_labels"],
                "sam_imagined": label in r["imagined_labels"],
                "vlm_real": real,
                "vlm_imagined": imag,
                "text": {"real": real_text, "imagined": imag_text},
            }
            append_jsonl(out_file, record)
            print(json.dumps({k: record[k] for k in record if k != "text"}), flush=True)
    if args.no_context:
        return

    # Pass 2: the imagined frame judged with the observed real frames as context.
    ctx_file = out / "context.jsonl"
    done = done_keys(ctx_file)
    for r in rows:
        if r["is_control"]:
            continue
        v, t = scene_views(r["scene"], cache), r["target"]
        observed = list(v["observed"])
        for label in r["imagined_labels"]:
            if (r["scene"], t, label) in done:
                continue
            ctx, text = ask(
                observed + [v["imagined"][t]], presence.with_context(label, len(observed))
            )
            record = {
                "scene": r["scene"],
                "target": t,
                "label": label,
                "vlm_context": ctx,
                "text": text,
            }
            append_jsonl(ctx_file, record)
            print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
