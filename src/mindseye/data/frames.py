"""Frame selection for videos, including the sparse-view observed/held-out split."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def uniform_indices(n_frames: int, k: int) -> list[int]:
    """``k`` frame indices spread evenly over a video, taken at segment midpoints."""
    if n_frames <= 0 or k <= 0:
        raise ValueError("n_frames and k must be positive")
    k = min(k, n_frames)
    return [int((i + 0.5) * n_frames / k) for i in range(k)]


def observed_indices(n_frames: int, k: int) -> list[int]:
    """Observed frames for the sparse-view setting.

    Evenly spaced segment midpoints, except that 4 frames are every other frame of
    the 8-frame set, so the 4-frame agent sees a strict subset of the 8-frame one.
    """
    if k == 4:
        return uniform_indices(n_frames, 8)[::2]
    return uniform_indices(n_frames, k)


def heldout_indices(n_frames: int, count: int = 8) -> list[int]:
    """Held-out frames: the starts of ``count`` equal segments.

    The rule does not depend on how many frames are observed, so every observed
    setting is scored against the same held-out views.
    """
    return sorted({int(i * n_frames / count) for i in range(count)})


def sparse_view_split(
    n_frames: int, n_observed: int, n_heldout: int = 8
) -> tuple[list[int], list[int]]:
    """Observed and held-out frame indices. The two sets never overlap.

    On very short videos a held-out index can coincide with an observed one; it is
    then dropped from the held-out set.
    """
    observed = observed_indices(n_frames, n_observed)
    taken = set(observed)
    return observed, [i for i in heldout_indices(n_frames, n_heldout) if i not in taken]


def read_frames(video_path: str | Path, indices: list[int]) -> list[np.ndarray]:
    """Decode the given frame indices as RGB arrays."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video_path}")
    frames = []
    try:
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, bgr = cap.read()
            if not ok:
                raise ValueError(f"{video_path}: cannot read frame {idx}")
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    return frames


def frame_count(video_path: str | Path) -> int:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video_path}")
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
