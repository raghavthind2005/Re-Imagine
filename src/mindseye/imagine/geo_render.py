"""Geometric renderer: novel views by splatting an observed point cloud.

This is the fallback imagination source. It cannot invent content: pixels that no
observed point projects to are reported as holes, and the fraction of filled
pixels is a direct measure of how much of a view the real observations cover.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mindseye.geometry.camera import project


@dataclass(frozen=True)
class RenderedView:
    image: np.ndarray  # (H, W, 3) uint8; holes are filled with ``background``
    depth: np.ndarray  # (H, W) float; +inf where nothing was hit
    mask: np.ndarray  # (H, W) bool; True where a point was splatted

    @property
    def coverage(self) -> float:
        """Fraction of pixels backed by an observed point."""
        return float(self.mask.mean())


def footprint_radii(
    points: np.ndarray,
    source_depth: np.ndarray,
    c2w: np.ndarray,
    *,
    focal_ratio: float = 1.0,
    max_radius: int = 8,
) -> np.ndarray:
    """Splat radius for each point so that surfaces stay closed when viewed closer.

    A point observed at depth ``source_depth`` covered one pixel there. Seen from a
    new camera at depth z it covers about ``focal_ratio * source_depth / z`` pixels.
    """
    z = (points - c2w[:3, 3]) @ c2w[:3, 2]
    footprint = focal_ratio * source_depth / np.maximum(z, 1e-6)
    return np.clip(np.ceil((footprint - 1) / 2), 0, max_radius).astype(np.int64)


def render_points(
    points: np.ndarray,
    colors: np.ndarray,
    c2w: np.ndarray,
    K: np.ndarray,
    width: int,
    height: int,
    *,
    point_radius: int | np.ndarray = 1,
    background: tuple[int, int, int] = (0, 0, 0),
) -> RenderedView:
    """Z-buffered point splatting.

    Each point paints a square of side ``2 * r + 1`` pixels, where ``r`` is
    ``point_radius`` (one value, or one per point; see ``footprint_radii``). The
    nearest point wins each pixel.
    """
    proj = project(points, c2w, K, width, height)
    all_radii = np.broadcast_to(np.asarray(point_radius, dtype=np.int64), (len(points),))
    # A point just outside the image can still paint pixels inside it.
    margin = int(all_radii.max()) if len(points) else 0
    keep = (
        (proj.depth > 1e-9)
        & (proj.uv[:, 0] >= -margin)
        & (proj.uv[:, 0] < width + margin)
        & (proj.uv[:, 1] >= -margin)
        & (proj.uv[:, 1] < height + margin)
    )
    u = np.floor(proj.uv[keep, 0]).astype(np.int64)
    v = np.floor(proj.uv[keep, 1]).astype(np.int64)
    z = proj.depth[keep]
    rgb = np.asarray(colors)[keep]
    radius = all_radii[keep]

    depth = np.full(height * width, np.inf)
    winner = np.full(height * width, -1, dtype=np.int64)
    # Visit points far-to-near so that, when several land on one pixel in one pass,
    # the last (nearest) assignment wins. The `closer` test protects pixels already
    # claimed by a nearer point in an earlier pass.
    order = np.argsort(-z, kind="stable")
    for r in np.unique(radius):
        sel = order[radius[order] == r]
        for dv in range(-r, r + 1):
            for du in range(-r, r + 1):
                uu, vv = u[sel] + du, v[sel] + dv
                ok = (uu >= 0) & (uu < width) & (vv >= 0) & (vv < height)
                flat = vv[ok] * width + uu[ok]
                zz = z[sel][ok]
                closer = zz < depth[flat]
                depth[flat[closer]] = zz[closer]
                winner[flat[closer]] = sel[ok][closer]
    mask = winner >= 0
    image = np.empty((height * width, 3), dtype=np.uint8)
    image[:] = background
    image[mask] = rgb[winner[mask]]
    return RenderedView(
        image=image.reshape(height, width, 3),
        depth=depth.reshape(height, width),
        mask=mask.reshape(height, width),
    )
