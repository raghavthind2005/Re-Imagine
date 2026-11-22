"""Run directories: every run records what is needed to trace a number back to it."""

from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git_commit(repo: str | Path = ".") -> str:
    """Return the current commit hash, with a ``-dirty`` suffix for uncommitted changes.

    Off-repo runs (the Kaggle bundle has no ``.git``) pass it in ``MINDSEYE_GIT_COMMIT``.
    """
    if os.environ.get("MINDSEYE_GIT_COMMIT"):
        return os.environ["MINDSEYE_GIT_COMMIT"]
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
    return f"{head}-dirty" if dirty else head


def gpu_name() -> str:
    try:
        import torch
    except ImportError:
        return "torch-not-installed"
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"


def make_run_id(name: str) -> str:
    return f"{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}_{name}"


def init_run(runs_root: str | Path, run_id: str, config: dict[str, Any], seed: int) -> Path:
    """Create ``runs/<run_id>/`` with the resolved config and environment record."""
    run_dir = Path(runs_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True
    ).stdout
    env = {
        "run_id": run_id,
        "git_commit": git_commit(),
        "seed": seed,
        "start_time": utc_now(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "gpu": gpu_name(),
    }
    (run_dir / "env.txt").write_text(
        json.dumps(env, indent=2) + "\n\n# pip freeze\n" + freeze, encoding="utf-8"
    )
    return run_dir


def get_logger(run_dir: str | Path, name: str = "mindseye") -> logging.Logger:
    """Logger that writes to ``<run_dir>/log.txt`` and to stderr."""
    logger = logging.getLogger(f"{name}.{Path(run_dir).name}")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.FileHandler(Path(run_dir) / "log.txt"), logging.StreamHandler()):
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.propagate = False
    return logger


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    """Append one prediction record; runs resume by reading back completed ids."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def completed_ids(path: str | Path, key: str = "id") -> set[str]:
    """Ids already present in a predictions file, ignoring a truncated final line."""
    path = Path(path)
    if not path.exists():
        return set()
    done: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            done.add(str(json.loads(line)[key]))
        except (json.JSONDecodeError, KeyError):
            continue
    return done
