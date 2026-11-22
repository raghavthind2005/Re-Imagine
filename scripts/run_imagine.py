"""Run the ``imagine`` mode in stages, one GPU model at a time.

    python scripts/run_imagine.py --config configs/xxx.yaml --run-id <id> --stage geom
    python scripts/run_imagine.py ... --stage vlm      # pick start frames
    python scripts/run_imagine.py ... --stage svc      # render what the search asked for
    python scripts/run_imagine.py ... --stage vlm      # score views, then answer

Qwen3-VL (both T4s), Stable Virtual Camera and VGGT do not fit in memory together.
Each stage replays every unfinished question from the caches and stops, for that
question, at the first result that belongs to a model it has not loaded. A search
of n steps needs: geom, then (vlm, svc) n times, then vlm. Stages are safe to
repeat. `--shard i/n` splits the questions so two GPUs can render at once.

Writes runs/<run-id>/{config.yaml, env.txt, log.txt, predictions.jsonl, stages.jsonl}.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml
from run_eval import build_backend, load_examples, peak_vram_gb, restrict_to_ids, subsample

from mindseye.agent import imagine
from mindseye.agent.vlm_only import select_media
from mindseye.data.schema import Example
from mindseye.imagine.svc_model import SVCCacheOnly, SVCSettings
from mindseye.imagine.svc_world_model import SVCWorldModel
from mindseye.utils.cache import DiskCache, Pending
from mindseye.utils.runlog import append_jsonl, completed_ids, get_logger, init_run, utc_now
from mindseye.utils.seeding import set_seed
from mindseye.vlm.base import CacheOnlyVLM


def geometry_key(example: Example, view_cfg: dict) -> str:
    """One reconstruction per scene and frame count; per question for still images."""
    if view_cfg["mode"] == "images":
        return example.id
    return f"{example.meta['scene'].replace('/', '__')}_k{view_cfg['n_observed']}"


def load_images(media: list) -> list[np.ndarray]:
    from PIL import Image

    return [
        m if isinstance(m, np.ndarray) else np.asarray(Image.open(m).convert("RGB")) for m in media
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", required=True, choices=["geom", "vlm", "svc"])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--svc-repo")
    ap.add_argument("--svc-weights")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--sheets", action="store_true", help="save a contact sheet per question")
    ap.add_argument("--max-seconds", type=float, help="stop starting new questions after this")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    cfg["limit"], cfg["config_file"] = args.limit, args.config
    seed = cfg["seed"]
    set_seed(seed)
    run_dir = init_run(args.runs_root, args.run_id, cfg, seed)
    log = get_logger(run_dir)
    pred_path = run_dir / "predictions.jsonl"
    cache_root = Path(args.cache_dir)
    geom_dir = cache_root / "geometry"
    geom_dir.mkdir(parents=True, exist_ok=True)
    svc_cache, vlm_cache = DiskCache(cache_root / "svc"), DiskCache(cache_root / "vlm")
    settings = SVCSettings(**cfg.get("svc", {}))
    view_cfg, icfg = cfg["view"], cfg["imagine"]

    examples = subsample(restrict_to_ids(load_examples(cfg), cfg["data"]), args.limit, seed)
    shard, n_shards = (int(x) for x in args.shard.split("/"))
    done = completed_ids(pred_path)
    todo = [e for e in examples if e.id not in done][shard::n_shards]
    log.info("stage %s: %d examples, %d to do in this shard", args.stage, len(examples), len(todo))
    start = time.perf_counter()
    counts = {"finished": 0, "pending_vlm": 0, "pending_svc": 0, "pending_geom": 0, "failed": 0}

    if args.stage == "geom":
        from mindseye.geometry.vggt_model import VGGTRunner

        vggt = VGGTRunner(device=args.device)
        for ex in todo:
            path = geom_dir / f"{geometry_key(ex, view_cfg)}.npz"
            if path.exists():
                continue
            media, _ = select_media(ex, view_cfg)
            try:
                recon = vggt.reconstruct(load_images(media))
            except ValueError as err:  # e.g. images of one question differ in shape
                log.warning("geometry failed for %s: %s", ex.id, err)
                counts["failed"] += 1
                continue
            depth = np.where(recon.depth > 0, recon.depth, np.nan)
            tmp = path.with_suffix(".tmp.npz")
            np.savez_compressed(
                tmp,
                images=recon.images,
                c2w=recon.c2w,
                K=recon.K,
                median_depth=np.nanmedian(depth.reshape(len(depth), -1), axis=1),
            )
            tmp.replace(path)
            counts["finished"] += 1
    else:
        if args.stage == "vlm":
            backend = build_backend(cfg, vlm_cache)
            runner = SVCCacheOnly(settings, svc_cache)
        else:
            from mindseye.imagine.svc_model import SVCRunner
            from mindseye.vlm.qwen_local import fingerprint

            if cfg["vlm"]["backend"] == "mock":  # dry runs: the mock needs no GPU
                backend = build_backend(cfg, vlm_cache)
            else:
                backend = CacheOnlyVLM(cfg["vlm"]["model_id"], fingerprint(cfg["vlm"]), vlm_cache)
            runner = SVCRunner(args.svc_repo, args.svc_weights, settings, svc_cache, args.device)
        gen_cfg = {**cfg["generation"], "seed": seed}
        for i, ex in enumerate(todo, 1):
            if args.max_seconds and time.perf_counter() - start > args.max_seconds:
                counts["skipped_time_limit"] = len(todo) - i + 1
                log.warning("time limit reached; %d questions left", len(todo) - i + 1)
                break
            path = geom_dir / f"{geometry_key(ex, view_cfg)}.npz"
            if not path.exists():
                counts["pending_geom"] += 1
                continue
            g = np.load(path)
            media, view_info = select_media(ex, view_cfg)
            wm = SVCWorldModel(runner, g["images"], g["c2w"], g["K"])
            geometry = imagine.SceneGeometry(g["c2w"], g["K"], g["median_depth"])
            try:
                record = imagine.answer(
                    ex,
                    load_images(media),
                    view_info,
                    geometry,
                    wm,
                    backend,
                    icfg,
                    gen_cfg,
                    is_video=view_cfg["mode"] != "images",
                    circular=view_cfg.get("circular", False),
                    sheet_path=run_dir / "sheets" / f"{ex.id}.png" if args.sheets else None,
                )
            except Pending as need:
                counts[f"pending_{need}"] += 1
            else:
                counts["finished"] += 1
                if args.stage == "vlm":  # only the VLM stage sees real answers
                    append_jsonl(pred_path, record)
            if i % 10 == 0 or i == len(todo):
                log.info("%d/%d, %.0fs, %s", i, len(todo), time.perf_counter() - start, counts)

    summary = {
        "stage": args.stage,
        "shard": args.shard,
        "end_time": utc_now(),
        "n_examples": len(examples),
        **counts,
        "wall_seconds": round(time.perf_counter() - start, 1),
        "peak_vram_gb_per_gpu": peak_vram_gb(),
    }
    append_jsonl(run_dir / "stages.jsonl", summary)
    log.info("stage finished: %s", json.dumps(summary))


if __name__ == "__main__":
    main()
