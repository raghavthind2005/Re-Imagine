"""Imagine held-out views with SVC and save them beside the real frames.

    python scripts/heldout_imagine.py --svc-repo /kaggle/svc --svc-weights <dir> \
        --out runs/heldout_imagine --n-scenes 5

Per scene: VGGT on the observed and held-out frames together gives every camera
in one coordinate system. SVC receives the observed frames with their poses and is
asked to render each held-out camera. Extra targets repeat observed cameras as
controls: there SVC should reproduce a frame it was given.

Saved per scene: `views.npz` (real and imagined held-out views, the geometric
render of the same poses, masks, poses) and `sheet.png` for inspection. Held-out
frames are used here only as ground truth for scoring.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from mindseye.data.frames import frame_count, read_frames, sparse_view_split
from mindseye.data.splits import load_split
from mindseye.data.vsibench import load_vsibench
from mindseye.geometry.reconstruct import fuse
from mindseye.geometry.vggt_model import VGGTRunner
from mindseye.imagine.geo_render import footprint_radii, render_points
from mindseye.imagine.svc_model import SVCRunner, SVCSettings, fit_intrinsics, fit_to
from mindseye.utils.cache import DiskCache
from mindseye.utils.runlog import append_jsonl


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(255.0**2 / mse)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--svc-repo", required=True)
    ap.add_argument("--svc-weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-scenes", type=int, default=5)
    ap.add_argument("--scene-file", help="split file with a fixed `scenes` list")
    ap.add_argument("--n-controls", type=int, default=1)
    ap.add_argument("--n-observed", type=int, default=8)
    ap.add_argument("--video-root", default="cache/vsibench_videos")
    ap.add_argument("--cache-dir", default="cache/svc")
    ap.add_argument("--vggt-device", default="cuda:1")
    ap.add_argument("--svc-device", default="cuda:0")
    args = ap.parse_args()

    from PIL import Image

    ids = set(load_split("data/splits/imagine_subset.json")["vsi_ids"])
    dev_scenes = load_split("data/splits/vsibench.json")["dev_scenes"]
    examples = load_vsibench("cache/hf/VSI-Bench/test.jsonl", "dev", args.video_root, dev_scenes)
    scenes = sorted({e.meta["scene"] for e in examples if e.id in ids})[: args.n_scenes]
    if args.scene_file:
        scenes = load_split(args.scene_file)["scenes"]

    settings = SVCSettings()
    vggt = VGGTRunner(device=args.vggt_device)
    svc = SVCRunner(
        args.svc_repo, args.svc_weights, settings, DiskCache(args.cache_dir), args.svc_device
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    W, H = settings.width, settings.height

    for scene in scenes:
        scene_dir = out / scene.replace("/", "__")
        if (scene_dir / "views.npz").exists():
            continue
        video = Path(args.video_root) / f"{scene}.mp4"
        observed, heldout = sparse_view_split(frame_count(video), args.n_observed, 8)
        recon = vggt.reconstruct(read_frames(video, observed + heldout))
        n_obs = len(observed)
        obs_idx, held_idx = list(range(n_obs)), list(range(n_obs, recon.n_frames))
        h0, w0 = recon.depth.shape[1:]

        # Controls come last: observed cameras spread over the video.
        controls = sorted({int(i) for i in np.linspace(0, n_obs - 1, args.n_controls).round()})
        targets = held_idx + controls
        n_held = len(held_idx)
        t0 = time.perf_counter()
        imagined = svc.render(
            [recon.images[i] for i in obs_idx],
            recon.c2w[obs_idx],
            recon.K[obs_idx],
            recon.c2w[targets],
            recon.K[targets],
        )
        seconds = time.perf_counter() - t0

        real = [fit_to(recon.images[i], W, H) for i in targets]
        cloud = fuse(recon, frames=obs_idx)
        geo, geo_mask = [], []
        for i in targets:
            K = fit_intrinsics(recon.K[i], w0, h0, W, H)
            radii = footprint_radii(cloud.points, cloud.depth, recon.c2w[i], focal_ratio=W / w0)
            view = render_points(
                cloud.points, cloud.colors, recon.c2w[i], K, W, H, point_radius=radii
            )
            geo.append(view.image)
            geo_mask.append(view.mask)

        scene_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            scene_dir / "views.npz",
            real=np.stack(real),
            imagined=np.stack(imagined),
            geo=np.stack(geo),
            geo_mask=np.stack(geo_mask),
            c2w=recon.c2w[targets],
            is_control=np.arange(len(targets)) >= n_held,
            control_observed_index=np.array(controls),
            observed=np.stack([fit_to(recon.images[i], W, H) for i in obs_idx]),
            observed_frames=np.array(observed),
            heldout_frames=np.array(heldout),
        )
        np.savez_compressed(
            scene_dir / "recon_observed.npz",
            images=recon.images[obs_idx],
            depth=recon.depth[obs_idx].astype(np.float32),
            confidence=recon.confidence[obs_idx].astype(np.float32),
            c2w=recon.c2w[obs_idx],
            K=recon.K[obs_idx],
            target_K=np.stack([fit_intrinsics(recon.K[i], w0, h0, W, H) for i in targets]),
            confidence_threshold=np.quantile(recon.confidence, 0.3),
        )
        rows = [np.concatenate(r, axis=1) for r in (real, imagined, geo)]
        Image.fromarray(np.concatenate(rows, axis=0)[::2, ::2]).save(scene_dir / "sheet.png")
        record = {
            "scene": scene,
            "n_observed": n_obs,
            "n_heldout": len(held_idx),
            "svc_seconds": round(seconds, 1),
            "svc_settings": settings.__dict__,
            "psnr_heldout": [
                round(psnr(a, b), 2) for a, b in zip(real[:n_held], imagined[:n_held], strict=True)
            ],
            "psnr_control": [
                round(psnr(a, b), 2) for a, b in zip(real[n_held:], imagined[n_held:], strict=True)
            ],
            "geo_coverage_heldout": [round(float(m.mean()), 3) for m in geo_mask[:n_held]],
        }
        append_jsonl(out / "results.jsonl", record)
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
