"""Bundle the committed code and launch a job on a Kaggle GPU kernel.

    python scripts/kaggle_launch.py --job kaggle/jobs/p1_smoke10.json

Only committed code is bundled (git archive of HEAD), so every Kaggle run maps
to a commit. The working tree must be clean.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import time
from pathlib import Path

USER = "raghavthind05"
SRC_DATASET = f"{USER}/reimagine-src"


def run(cmd: list[str], **kw) -> str:
    print("$", " ".join(cmd))
    out = subprocess.run(cmd, capture_output=True, text=True, **kw)
    print(out.stdout[-2000:], out.stderr[-2000:])
    if out.returncode != 0 or "push error" in out.stdout.lower():
        raise SystemExit(f"failed: {' '.join(cmd)}")
    return out.stdout


def push_bundle(commit: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        stage = tmp_path / "stage"
        stage.mkdir()
        archive = subprocess.run(["git", "archive", "HEAD"], capture_output=True, check=True)
        subprocess.run(["tar", "-x", "-C", str(stage)], input=archive.stdout, check=True)
        (stage / "BUNDLE_COMMIT").write_text(commit)
        up = tmp_path / "upload"
        up.mkdir()
        subprocess.run(
            ["tar", "-czf", str(up / "reimagine_src.tar.gz"), "-C", str(stage), "."], check=True
        )
        meta = {"title": "reimagine-src", "id": SRC_DATASET, "licenses": [{"name": "other"}]}
        (up / "dataset-metadata.json").write_text(json.dumps(meta))
        exists = subprocess.run(
            ["kaggle", "datasets", "status", SRC_DATASET], capture_output=True, text=True
        )
        if exists.returncode == 0:
            run(["kaggle", "datasets", "version", "-p", str(up), "-m", f"commit {commit}", "-q"])
        else:
            run(["kaggle", "datasets", "create", "-p", str(up), "-q"])


def wait_until_ready(timeout_s: int = 300) -> None:
    """A kernel pushed before the dataset finishes processing starts without it."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        status = subprocess.run(
            ["kaggle", "datasets", "status", SRC_DATASET], capture_output=True, text=True
        ).stdout.strip()
        if status.endswith("ready"):
            return
        time.sleep(10)
    raise SystemExit(f"{SRC_DATASET} not ready after {timeout_s}s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--skip-bundle", action="store_true")
    args = ap.parse_args()
    job = json.loads(Path(args.job).read_text())

    if subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout:
        raise SystemExit("working tree is not clean; commit first so the run maps to a commit")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    marker = Path("cache/.kaggle_bundle_commit")
    if args.skip_bundle and (not marker.exists() or marker.read_text() != commit):
        raise SystemExit("--skip-bundle refused: the uploaded bundle is not this commit")
    if not args.skip_bundle:
        push_bundle(commit)
        marker.parent.mkdir(exist_ok=True)
        marker.write_text(commit)
        # A new dataset version takes a while to replace the old one, and the
        # status reads "ready" for the old version in the meantime.
        time.sleep(90)
        job["expect_commit"] = commit
    wait_until_ready()

    template = Path("kaggle/kernel_template.py").read_text()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "kernel.py").write_text(template.replace("__JOB__", repr(job), 1))
        meta = {
            "id": f"{USER}/{job['kernel']}",
            "title": job["kernel"],
            "code_file": "kernel.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": True,
            "enable_internet": True,
            "dataset_sources": [SRC_DATASET] + job.get("extra_datasets", []),
            "competition_sources": [],
            "kernel_sources": job.get("kernel_sources", []),
        }
        (tmp_path / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
        run(
            ["kaggle", "kernels", "push", "-p", str(tmp_path), "--accelerator", job["accelerator"]]
            + ["-t", str(job["timeout_s"])]
        )
    print(f"launched {USER}/{job['kernel']} at commit {commit}")


if __name__ == "__main__":
    main()
