"""Run one experiment config.

    python scripts/run_eval.py --config configs/xxx.yaml [--limit 10]

Writes runs/<run_id>/{config.yaml, env.txt, log.txt, predictions.jsonl}. Re-running
with the same --run-id resumes from the last completed example.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import yaml

from mindseye.agent import vlm_only
from mindseye.data.sat import load_sat
from mindseye.data.schema import Example
from mindseye.data.splits import load_split
from mindseye.data.vsibench import load_vsibench
from mindseye.utils.cache import DiskCache
from mindseye.utils.runlog import (
    append_jsonl,
    completed_ids,
    get_logger,
    init_run,
    make_run_id,
    utc_now,
)
from mindseye.utils.seeding import set_seed
from mindseye.vlm.base import MockVLM, VLMBackend


def load_examples(cfg: dict) -> list[Example]:
    data = cfg["data"]
    split = data["split"]
    if split == "test" and not cfg.get("final", False):
        raise SystemExit("test split is only evaluated by configs marked `final: true`")
    if data["benchmark"] == "sat":
        if split == "dev":
            rows = load_split(data["split_file"])["dev_val_rows"]
            return load_sat(data["val_parquet"], "val", "dev", data["image_dir"], rows)
        return load_sat(data["test_parquet"], "test", "test", data["image_dir"])
    if data["benchmark"] == "vsibench":
        scenes = load_split(data["split_file"])[f"{split}_scenes"]
        return load_vsibench(data["jsonl"], split, data["video_root"], scenes)
    raise SystemExit(f"unknown benchmark: {data['benchmark']}")


def restrict_to_ids(examples: list[Example], data: dict) -> list[Example]:
    """Keep only the examples named in a fixed id file, if the config gives one."""
    if "id_file" not in data:
        return examples
    wanted = set(load_split(data["id_file"])[data["id_key"]])
    kept = [e for e in examples if e.id in wanted]
    if len(kept) != len(wanted):
        raise SystemExit(f"id file names {len(wanted)} examples but {len(kept)} were found")
    return kept


def subsample(examples: list[Example], limit: int | None, seed: int) -> list[Example]:
    """A fixed pseudo-random subset, so smoke runs cover every question type."""
    if limit is None or limit >= len(examples):
        return examples

    def rank(e: Example) -> str:
        return hashlib.sha256(f"{seed}:{e.id}".encode()).hexdigest()

    return sorted(examples, key=rank)[:limit]


def build_backend(cfg: dict, cache: DiskCache) -> VLMBackend:
    vlm = cfg["vlm"]
    if vlm["backend"] == "mock":
        return MockVLM(vlm.get("reply", "A"), cache=cache)
    if vlm["backend"] == "qwen_local":
        from mindseye.vlm.qwen_local import QwenLocal

        return QwenLocal(
            vlm["model_id"],
            dtype=vlm.get("dtype", "float16"),
            load_in_4bit=vlm.get("load_in_4bit", False),
            max_pixels=vlm.get("max_pixels"),
            max_memory=vlm.get("max_memory"),
            cache=cache,
        )
    raise SystemExit(f"unknown backend: {vlm['backend']}")


def peak_vram_gb() -> list[float]:
    try:
        import torch
    except ImportError:
        return []
    return [
        round(torch.cuda.max_memory_allocated(i) / 2**30, 2)
        for i in range(torch.cuda.device_count())
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--cache-dir", default="cache/vlm")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    cfg["limit"] = args.limit
    cfg["config_file"] = args.config
    seed = cfg["seed"]
    set_seed(seed)

    suffix = f"-n{args.limit}" if args.limit else ""
    run_id = args.run_id or make_run_id(cfg["name"] + suffix)
    run_dir = init_run(args.runs_root, run_id, cfg, seed)
    log = get_logger(run_dir)
    pred_path = run_dir / "predictions.jsonl"

    examples = subsample(restrict_to_ids(load_examples(cfg), cfg["data"]), args.limit, seed)
    done = completed_ids(pred_path)
    todo = [e for e in examples if e.id not in done]
    log.info("run %s: %d examples, %d already done", run_id, len(examples), len(done))

    backend = build_backend(cfg, DiskCache(args.cache_dir))
    gen_cfg = {**cfg["generation"], "seed": seed}
    start = time.perf_counter()
    for i, ex in enumerate(todo, 1):
        append_jsonl(pred_path, vlm_only.answer(ex, backend, cfg["view"], gen_cfg))
        if i % 25 == 0 or i == len(todo):
            log.info("%d/%d done, %.1fs elapsed", i, len(todo), time.perf_counter() - start)

    summary = {
        "run_id": run_id,
        "end_time": utc_now(),
        "n_examples": len(examples),
        "n_new": len(todo),
        "wall_seconds_this_session": round(time.perf_counter() - start, 1),
        "peak_vram_gb_per_gpu": peak_vram_gb(),
    }
    append_jsonl(run_dir / "sessions.jsonl", summary)
    log.info("finished: %s", json.dumps(summary))


if __name__ == "__main__":
    main()
