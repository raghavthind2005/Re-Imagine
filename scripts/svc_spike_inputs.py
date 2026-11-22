"""Pick input images for the SVC spike from dev data only (never test).

Writes one SAT dev image and the middle frame of the first two ARKitScenes dev
videos to the output directory, and prints their paths.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from mindseye.data.frames import frame_count, read_frames
from mindseye.data.sat import load_sat
from mindseye.data.splits import load_split


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sat-val", default="cache/hf/SAT/SAT_val.parquet")
    ap.add_argument("--video-root", default="cache/vsibench_videos")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    row = load_split("data/splits/sat.json")["dev_val_rows"][0]
    sat = load_sat(args.sat_val, "val", "dev", out / "sat_tmp", [row])[0]
    Image.open(sat.media[0]).convert("RGB").save(out / "sat_dev.png")

    scenes = [s for s in load_split("data/splits/vsibench.json")["dev_scenes"] if "arkit" in s]
    for i, scene in enumerate(scenes[:2]):
        video = Path(args.video_root) / f"{scene}.mp4"
        frame = read_frames(video, [frame_count(video) // 2])[0]
        Image.fromarray(frame).save(out / f"arkit_dev_{i}.png")
    print(" ".join(str(p) for p in sorted(out.glob("*.png"))))


if __name__ == "__main__":
    main()
