"""Feasibility spike: can Stable Virtual Camera imagine views on a 16 GB T4?

Runs the official `demo.py` (task `img2trajvid_s-prob`: one input image, a preset
camera move) under several memory-saving settings, each in its own process so an
out-of-memory failure in one does not affect the others. For each variant it
records whether it ran, peak VRAM, seconds per generated view and what was
changed from the defaults, and keeps the generated frames beside the input.

    python scripts/svc_spike.py driver --svc-repo /path/to/stable-virtual-camera \
        --weights /path/to/weights_dir --images a.png b.png --out runs/svc_spike

This is a measurement script, not part of the agent pipeline.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Ordered from the authors' defaults to the most reduced setting.
VARIANTS = [
    {"name": "default_bf16_576", "dtype": "bfloat16", "lowvram": False, "H": 576, "W": 576,
     "T": 21, "num_steps": 50, "num_targets": 8},
    {"name": "lowvram_bf16_576", "dtype": "bfloat16", "lowvram": True, "H": 576, "W": 576,
     "T": 21, "num_steps": 50, "num_targets": 8},
    {"name": "lowvram_fp16_576", "dtype": "float16", "lowvram": True, "H": 576, "W": 576,
     "T": 21, "num_steps": 50, "num_targets": 8},
    {"name": "lowvram_fp16_576_s20", "dtype": "float16", "lowvram": True, "H": 576, "W": 576,
     "T": 21, "num_steps": 20, "num_targets": 8},
    {"name": "lowvram_fp16_384_s20", "dtype": "float16", "lowvram": True, "H": 384, "W": 384,
     "T": 21, "num_steps": 20, "num_targets": 8},
    {"name": "lowvram_fp16_384_T8_s20", "dtype": "float16", "lowvram": True, "H": 384, "W": 384,
     "T": 8, "num_steps": 20, "num_targets": 7},
]  # fmt: skip
# Presets of the official demo. It has no turn-in-place preset; rotations need a
# custom camera path (see src/mindseye/imagine/).
TRAJ_PRIORS = ["move-forward", "move-left"]


def worker(args: argparse.Namespace) -> None:
    """Run one variant on one image. Executed in a subprocess with cwd = the SVC repo."""
    import torch

    variant = json.loads(args.variant)
    sys.path.insert(0, os.getcwd())
    import seva.eval as seva_eval

    seva_eval.set_lowvram_mode(variant["lowvram"])
    import demo

    load_seconds = []
    original_load = demo.load_model

    def timed_load(*a, **kw):
        t = time.perf_counter()
        model = original_load(*a, **kw)
        if variant["dtype"] != "bfloat16":
            model = model.to(getattr(torch, variant["dtype"]))
        load_seconds.append(time.perf_counter() - t)
        return model

    demo.load_model = timed_load

    demo.WORK_DIR = args.out
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    demo.main(
        data_path=args.data_dir,
        task="img2trajvid_s-prob",
        version=1.1,
        H=variant["H"],
        W=variant["W"],
        T=variant["T"],
        pretrained_model_name_or_path=args.weights,
        weight_name="modelv1.1.safetensors",
        seed=0,
        traj_prior=args.traj_prior,
        num_steps=variant["num_steps"],
        num_targets=variant["num_targets"],
    )
    seconds = time.perf_counter() - start
    frames = glob.glob(f"{args.out}/**/samples-rgb/*.png", recursive=True)
    sampling = seconds - sum(load_seconds)
    result = {
        "seconds_total": round(seconds, 1),
        "seconds_weight_load": round(sum(load_seconds), 1),
        "n_views": len(frames),
        "seconds_per_view": round(sampling / max(len(frames), 1), 2),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "gpu": torch.cuda.get_device_name(0),
    }
    Path(args.result).write_text(json.dumps(result))


def driver(args: argparse.Namespace) -> None:
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    results_path = out / "results.jsonl"
    deadline = time.time() + args.budget_s
    by_name = {v["name"]: v for v in VARIANTS}
    if args.gallery:
        # One chosen variant on every image and move, for the example views.
        jobs = [(by_name[args.gallery], im, pr) for im in args.images for pr in TRAJ_PRIORS]
    else:
        # Every variant once, on the first image and move, for VRAM and speed.
        names = args.variants or list(by_name)
        jobs = [(by_name[n], args.images[0], TRAJ_PRIORS[0]) for n in names]
    for variant, image, prior in jobs:
        if time.time() > deadline:
            print("time budget reached; stopping")
            return
        run_one(args, out, results_path, variant, image, prior)


def run_one(args, out: Path, results_path: Path, variant: dict, image: str, prior: str) -> None:
    tag = f"{variant['name']}__{Path(image).stem}__{prior}"
    work = out / tag
    data_dir = work / "input"
    data_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(image, data_dir / Path(image).name)
    result_file = work / "result.json"
    cmd = [
        sys.executable, str(Path(__file__).resolve()), "worker",
        "--variant", json.dumps(variant), "--data-dir", str(data_dir),
        "--weights", args.weights, "--traj-prior", prior,
        "--out", str(work / "svc"), "--result", str(result_file),
    ]  # fmt: skip
    proc = subprocess.run(cmd, cwd=args.svc_repo, capture_output=True, text=True)
    record = {"tag": tag, "variant": variant, "image": image, "traj_prior": prior}
    if proc.returncode == 0 and result_file.exists():
        record.update(ok=True, **json.loads(result_file.read_text()))
    else:
        tail = (proc.stderr or proc.stdout)[-1500:]
        record.update(ok=False, oom="out of memory" in tail.lower(), error_tail=tail)
    with open(results_path, "a") as f:
        f.write(json.dumps(record) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k != "error_tail"}), flush=True)
    if not record["ok"]:
        print(record["error_tail"][-600:], flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("driver")
    d.add_argument("--svc-repo", required=True)
    d.add_argument("--weights", required=True)
    d.add_argument("--images", nargs="+", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--budget-s", type=int, default=2 * 3600)
    d.add_argument("--variants", nargs="*", help="sweep only these variants")
    d.add_argument("--gallery", help="run this one variant on every image and move")
    w = sub.add_parser("worker")
    for name in ("variant", "data-dir", "weights", "traj-prior", "out", "result"):
        w.add_argument(f"--{name}", required=True)
    args = ap.parse_args()
    (driver if args.cmd == "driver" else worker)(args)


if __name__ == "__main__":
    main()
