"""Turn per-frame depth and cameras (as predicted by VGGT) into a fused point cloud.

Kept free of torch so the fusion logic is testable with synthetic depth maps. The
model wrapper only has to produce a ``Reconstruction``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mindseye.geometry.camera import unproject_depth


@dataclass(frozen=True)
class Reconstruction:
    """Raw per-frame geometry for F frames of size (H, W)."""

    images: np.ndarray  # (F, H, W, 3) uint8, at the resolution the model saw
    depth: np.ndarray  # (F, H, W)
    confidence: np.ndarray  # (F, H, W); higher is more reliable
    c2w: np.ndarray  # (F, 4, 4)
    K: np.ndarray  # (F, 3, 3)

    @property
    def n_frames(self) -> int:
        return len(self.images)


def confidence_mask(recon: Reconstruction, keep_fraction: float) -> np.ndarray:
    """(F, H, W) mask keeping the most confident ``keep_fraction`` of all pixels."""
    if not 0 < keep_fraction <= 1:
        raise ValueError("keep_fraction must be in (0, 1]")
    threshold = np.quantile(recon.confidence, 1 - keep_fraction)
    return (recon.confidence >= threshold) & (recon.depth > 0)


def frame_points(recon: Reconstruction, i: int) -> np.ndarray:
    """(H, W, 3) world points of frame ``i``."""
    return unproject_depth(recon.depth[i].astype(np.float64), recon.c2w[i], recon.K[i])


@dataclass(frozen=True)
class FusedCloud:
    points: np.ndarray  # (N, 3)
    colors: np.ndarray  # (N, 3) uint8
    frame: np.ndarray  # (N,) index of the frame each point came from
    depth: np.ndarray  # (N,) depth at which each point was observed


def fuse(
    recon: Reconstruction,
    *,
    keep_fraction: float = 0.7,
    frames: list[int] | None = None,
) -> FusedCloud:
    """Fused point cloud from the chosen frames (all frames by default).

    The confidence threshold is always computed over all frames of ``recon``, so
    choosing a subset does not change which pixels of a frame are kept.
    """
    mask = confidence_mask(recon, keep_fraction)
    use = range(recon.n_frames) if frames is None else frames
    pts, cols, src, dep = [], [], [], []
    for i in use:
        m = mask[i]
        pts.append(frame_points(recon, i)[m])
        cols.append(recon.images[i][m])
        src.append(np.full(int(m.sum()), i))
        dep.append(recon.depth[i][m])
    return FusedCloud(
        np.concatenate(pts), np.concatenate(cols), np.concatenate(src), np.concatenate(dep)
    )


def subset(recon: Reconstruction, frames: list[int]) -> Reconstruction:
    """The same reconstruction restricted to some of its frames."""
    idx = np.asarray(frames)
    return Reconstruction(
        recon.images[idx], recon.depth[idx], recon.confidence[idx], recon.c2w[idx], recon.K[idx]
    )


def downsample(recon: Reconstruction, factor: int) -> Reconstruction:
    """Keep every ``factor``-th pixel in both directions, scaling the intrinsics to match."""
    if factor == 1:
        return recon
    K = recon.K.copy()
    K[:, :2] /= factor
    s = slice(None, None, factor)
    return Reconstruction(
        recon.images[:, s, s], recon.depth[:, s, s], recon.confidence[:, s, s], recon.c2w, K
    )
