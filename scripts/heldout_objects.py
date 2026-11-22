"""Object-level hallucination measurement on imagined held-out views.

    python scripts/heldout_objects.py --in runs/heldout_imagine --out runs/heldout_objects

For every scene directory written by `heldout_imagine.py`:

1. SAM 3 detects a fixed object vocabulary in each real held-out view, each
   imagined view of the same camera, and each observed frame.
2. Real and imagined views are compared label by label (matched / invented /
   missing, coarse position).
3. A labelled 3D map is built from the observed frames only (SAM 3 masks lifted
   with VGGT depth). Each object the imagined view shows is turned into the claim
   "<label> is visible from this camera" and checked against the map:
   - supported     the map has that label and it projects, unoccluded, into the view
   - contradicted  the map covers at least half of the region where the imagined
                   object sits and shows no such object there
   - no_coverage   the map does not cover that region
4. Control targets (observed cameras) give the detector's own disagreement rate
   between a real frame and SVC's copy of it.

`--shard i/n` processes every n-th scene, so two GPUs can share the work; each
shard writes its own `views_shard<i>.jsonl`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mindseye.eval.hallucination import Obj, compare_view
from mindseye.geometry.camera import project, unproject_depth
from mindseye.geometry.sam3_model import VSI_OBJECTS, SAM3Runner
from mindseye.imagine.geo_render import render_points
from mindseye.utils.runlog import append_jsonl, completed_ids

MIN_VISIBLE_FRACTION = 0.002  # of the view's pixels, for a mapped object to count as visible
REGION_COVERED = 0.5


def to_objs(dets) -> list[Obj]:
    return [Obj(d.label, *d.centroid) for d in dets]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default="facebook/sam3")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--shard", default="0/1")
    args = ap.parse_args()
    shard, n_shards = (int(x) for x in args.shard.split("/"))
    sam = SAM3Runner(args.weights, device=args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    out_file = out / (f"views_shard{shard}.jsonl" if n_shards > 1 else "views.jsonl")
    done = completed_ids(out_file, key="scene")
    scene_dirs = sorted(p for p in Path(args.inp).iterdir() if (p / "views.npz").exists())
    for scene_dir in scene_dirs[shard::n_shards]:
        if scene_dir.name in done:
            continue
        v = np.load(scene_dir / "views.npz")
        r = np.load(scene_dir / "recon_observed.npz")
        H, W = v["real"].shape[1:3]
        n_targets = len(v["real"])

        # Labelled 3D points from the observed frames.
        label_points: dict[str, list[np.ndarray]] = {}
        all_points = []
        for f in range(len(r["images"])):
            pts = unproject_depth(r["depth"][f].astype(np.float64), r["c2w"][f], r["K"][f])
            trusted = (r["confidence"][f] >= r["confidence_threshold"]) & (r["depth"][f] > 0)
            all_points.append(pts[trusted])
            for det in sam.detect(r["images"][f], VSI_OBJECTS, threshold=args.threshold):
                m = det.mask & trusted
                if m.any():
                    label_points.setdefault(det.label, []).append(pts[m])
        cloud = np.concatenate(all_points)
        grey = np.zeros((len(cloud), 3), np.uint8)

        records = []  # written together, so a scene is either complete or absent
        for t in range(n_targets):
            if "is_control" in v:
                is_control = bool(v["is_control"][t])
            else:  # scenes written before several controls were supported
                is_control = t == n_targets - 1
            real = sam.detect(v["real"][t], VSI_OBJECTS, threshold=args.threshold)
            imag = sam.detect(v["imagined"][t], VSI_OBJECTS, threshold=args.threshold)
            cmp_ = compare_view(to_objs(real), to_objs(imag), W, H)
            c2w, K = v["c2w"][t], r["target_K"][t]
            zbuf = render_points(cloud, grey, c2w, K, W, H, point_radius=1).depth
            geo_mask = v["geo_mask"][t]

            claims = []
            for label in sorted({d.label for d in imag}):
                visible = 0
                for pts in label_points.get(label, []):
                    proj = project(pts, c2w, K, W, H)
                    uv = np.floor(proj.uv[proj.in_view]).astype(np.int64)
                    z = proj.depth[proj.in_view]
                    visible += int((z <= zbuf[uv[:, 1], uv[:, 0]] * 1.05).sum())
                region = np.any([d.mask for d in imag if d.label == label], axis=0)
                region_covered = float(geo_mask[region].mean())
                if visible >= MIN_VISIBLE_FRACTION * H * W:
                    verdict = "supported"
                elif region_covered >= REGION_COVERED:
                    verdict = "contradicted"
                else:
                    verdict = "no_coverage"
                claims.append(
                    {
                        "label": label,
                        "true_in_real_view": label in cmp_.matched,
                        "verdict": verdict,
                        "map_pixels_visible": visible,
                        "region_covered": round(region_covered, 3),
                        "label_in_map": label in label_points,
                    }
                )
            record = {
                "scene": scene_dir.name,
                "target": t,
                "is_control": is_control,
                "geo_coverage": round(float(geo_mask.mean()), 3),
                "real_labels": sorted({d.label for d in real}),
                "imagined_labels": sorted({d.label for d in imag}),
                "matched": cmp_.matched,
                "invented": cmp_.invented,
                "missing": cmp_.missing,
                "position_checked": cmp_.position_checked,
                "position_agree": cmp_.position_agree,
                "claims": claims,
                "map_labels": sorted(label_points),
            }
            records.append(record)
            shown = ("scene", "target", "is_control", "matched", "invented", "missing")
            print(json.dumps({k: record[k] for k in shown}), flush=True)

        for record in records:
            append_jsonl(out_file, record)


if __name__ == "__main__":
    main()
