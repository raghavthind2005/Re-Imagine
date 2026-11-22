"""Fixed dev/test splits: seeded, stratified, saved to disk and never regenerated."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Hashable, Sequence
from pathlib import Path

import numpy as np


def stratified_sample(
    keys: Sequence[Hashable], strata: Sequence[Hashable], n: int, seed: int
) -> list[Hashable]:
    """Pick ``n`` keys with each stratum represented in proportion to its size.

    Uses largest-remainder rounding so the total is exactly ``n``.
    """
    if n > len(keys):
        raise ValueError(f"cannot sample {n} from {len(keys)}")
    groups: dict[Hashable, list[Hashable]] = defaultdict(list)
    for k, s in zip(keys, strata, strict=True):
        groups[s].append(k)
    names = sorted(groups, key=str)
    exact = [len(groups[g]) * n / len(keys) for g in names]
    take = [int(x) for x in exact]
    by_remainder = sorted(range(len(names)), key=lambda i: (exact[i] - take[i], -i), reverse=True)
    for i in by_remainder[: n - sum(take)]:
        take[i] += 1
    rng = np.random.default_rng(seed)
    chosen: list[Hashable] = []
    for g, t in zip(names, take, strict=True):
        members = sorted(groups[g], key=str)
        chosen.extend(members[i] for i in sorted(rng.permutation(len(members))[:t]))
    return chosen


def split_groups(
    groups: Sequence[Hashable], strata: Sequence[Hashable], dev_fraction: float, seed: int
) -> tuple[list[Hashable], list[Hashable]]:
    """Split unique groups (e.g. scenes) into dev and test, stratified by ``strata``.

    Splitting by group keeps every question about one scene on the same side.
    """
    unique: dict[Hashable, Hashable] = {}
    for g, s in zip(groups, strata, strict=True):
        if unique.setdefault(g, s) != s:
            raise ValueError(f"group {g!r} appears in more than one stratum")
    keys = sorted(unique, key=str)
    n_dev = round(len(keys) * dev_fraction)
    dev = set(stratified_sample(keys, [unique[k] for k in keys], n_dev, seed))
    return [k for k in keys if k in dev], [k for k in keys if k not in dev]


def save_split(path: str | Path, split: dict[str, list], meta: dict) -> None:
    """Write a split once. Refuses to overwrite: splits are never regenerated."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} already exists; splits are fixed once written")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"meta": meta, "split": split}, indent=1), encoding="utf-8")


def load_split(path: str | Path) -> dict[str, list]:
    return json.loads(Path(path).read_text(encoding="utf-8"))["split"]
