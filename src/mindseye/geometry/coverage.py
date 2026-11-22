"""How much of a view does a reconstruction of other frames account for?"""

from __future__ import annotations

import numpy as np

from mindseye.geometry.reconstruct import Reconstruction, confidence_mask, fuse
from mindseye.imagine.geo_render import footprint_radii, render_points


def view_coverage_from(
    recon: Reconstruction,
    source_frames: list[int],
    target_frame: int,
    *,
    keep_fraction: float = 0.7,
    depth_tolerance: float = 0.1,
) -> dict[str, float]:
    """Render ``target_frame``'s viewpoint from the points of ``source_frames``.

    ``coverage``             fraction of the target view's pixels that a source point lands on
    ``consistent_coverage``  fraction of the target's own confident pixels where the rendered
                             depth is within ``depth_tolerance`` (relative) of the target's own
                             depth. This is the share of the view the map explains correctly,
                             not merely paints over.
    """
    cloud = fuse(recon, keep_fraction=keep_fraction, frames=source_frames)
    c2w, K = recon.c2w[target_frame], recon.K[target_frame]
    h, w = recon.depth.shape[1:]
    radii = footprint_radii(cloud.points, cloud.depth, c2w)
    view = render_points(cloud.points, cloud.colors, c2w, K, w, h, point_radius=radii)
    own = recon.depth[target_frame]
    trusted = confidence_mask(recon, keep_fraction)[target_frame]
    agree = view.mask & (np.abs(view.depth - own) <= depth_tolerance * own)
    return {
        "coverage": view.coverage,
        "consistent_coverage": float(agree[trusted].mean()) if trusted.any() else 0.0,
    }


def sparse_view_coverage(
    recon: Reconstruction,
    observed: list[int],
    heldout: list[int],
    *,
    max_overlap_views: int | None = None,
    **kw,
) -> dict[str, float]:
    """Coverage statistics for one scene.

    ``observed`` and ``heldout`` index frames of ``recon``. Observed overlap is
    leave-one-out: an observed view rendered from all the other observed frames
    (on an evenly spaced sample of at most ``max_overlap_views`` views, to bound
    cost). Held-out coverage renders each held-out view from all observed frames.
    """
    targets = observed if len(observed) > 1 else []
    if max_overlap_views is not None and len(targets) > max_overlap_views:
        picks = np.linspace(0, len(targets) - 1, max_overlap_views).round().astype(int)
        targets = [targets[i] for i in picks]
    loo = [view_coverage_from(recon, [j for j in observed if j != i], i, **kw) for i in targets]
    held = [view_coverage_from(recon, observed, i, **kw) for i in heldout]

    def mean(rows: list[dict], key: str) -> float:
        return float(np.mean([r[key] for r in rows])) if rows else float("nan")

    return {
        "observed_overlap": mean(loo, "coverage"),
        "observed_overlap_consistent": mean(loo, "consistent_coverage"),
        "heldout_coverage": mean(held, "coverage"),
        "heldout_coverage_consistent": mean(held, "consistent_coverage"),
        "heldout_views_half_covered": (
            float(np.mean([r["consistent_coverage"] >= 0.5 for r in held]))
            if held
            else float("nan")
        ),
        "n_heldout": len(held),
    }
