"""Frame-count sweep, geometry side: how much do K observed frames cover?

    python scripts/frame_sweep_geometry.py --out runs/frame_sweep/geometry.jsonl

For every scene of the fixed imagination subset and each K, VGGT is run once on
the K observed frames together with the 8 held-out frames, which puts all their
cameras in one coordinate system. The map is then built from the observed
frames' depth only and scored on (a) how much observed frames overlap each other
and (b) how much of each held-out view the map accounts for.

Caveat: because the forward pass is joint, the observed frames' depth can benefit
from the held-out frames being present. This affects depth quality slightly, not
which surfaces the observed cameras see.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from mindseye.data.frames import frame_count, read_frames, sparse_view_split
from mindseye.data.splits import load_split
from mindseye.data.vsibench import load_vsibench
from mindseye.geometry.coverage import sparse_view_coverage
from mindseye.geometry.reconstruct import downsample
from mindseye.geometry.vggt_model import VGGTRunner
from mindseye.utils.runlog import append_jsonl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ks", type=int, nargs="+", default=[4, 8, 16])
    ap.add_argument("--video-root", default="cache/vsibench_videos")
    ap.add_argument("--weights", default="facebook/VGGT-1B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--limit-scenes", type=int, default=None)
    ap.add_argument("--max-loo", type=int, default=4, help="observed views used for overlap")
    args = ap.parse_args()

    ids = set(load_split("data/splits/imagine_subset.json")["vsi_ids"])
    dev_scenes = load_split("data/splits/vsibench.json")["dev_scenes"]
    examples = load_vsibench("cache/hf/VSI-Bench/test.jsonl", "dev", args.video_root, dev_scenes)
    scenes = sorted({e.meta["scene"] for e in examples if e.id in ids})[: args.limit_scenes]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {(r["scene"], r["k"]) for r in map(json.loads, out.read_text().splitlines())}
    runner = VGGTRunner(args.weights, device=args.device, dtype=args.dtype)
    import torch

    for scene in scenes:
        video = Path(args.video_root) / f"{scene}.mp4"
        n = frame_count(video)
        for k in args.ks:
            if (scene, k) in done:
                continue
            observed, heldout = sparse_view_split(n, k, 8)
            record = {"scene": scene, "k": k, "n_video_frames": n, "n_heldout_frames": len(heldout)}
            try:
                frames = read_frames(video, observed + heldout)
                if args.device.startswith("cuda"):
                    torch.cuda.reset_peak_memory_stats()
                t0 = time.perf_counter()
                recon = runner.reconstruct(frames)
                record["vggt_seconds"] = round(time.perf_counter() - t0, 2)
                if args.device.startswith("cuda"):
                    record["vggt_peak_vram_gb"] = round(
                        torch.cuda.max_memory_allocated() / 2**30, 2
                    )
                obs_idx = list(range(len(observed)))
                held_idx = list(range(len(observed), len(frames)))
                stats = sparse_view_coverage(
                    downsample(recon, 2), obs_idx, held_idx, max_overlap_views=args.max_loo
                )
                record.update(ok=True, **stats)
            except Exception as err:  # noqa: BLE001 - one bad scene must not stop the sweep
                record.update(ok=False, error=f"{type(err).__name__}: {str(err)[:300]}")
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()
            append_jsonl(out, record)
            print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
