import numpy as np
import pytest

from mindseye.geometry.camera import intrinsics, project, rollout, unproject_depth, w2c_to_c2w
from mindseye.geometry.reconstruct import Reconstruction, confidence_mask, fuse
from mindseye.imagine.geo_render import render_points

W, H = 64, 48
K = intrinsics(W, H, 90)


def wall_depth(c2w, z_wall=4.0):
    """Exact depth map of the plane z = z_wall (world) seen from a camera, plus its colours."""
    u, v = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))
    rays_cam = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1)
    rays_world = rays_cam @ c2w[:3, :3].T
    depth = (z_wall - c2w[2, 3]) / rays_world[..., 2]
    hit = c2w[:3, 3] + rays_world * depth[..., None]
    colour = np.stack([(hit[..., 0] * 20 % 255), (hit[..., 1] * 20 % 255), np.full_like(u, 90)], -1)
    return depth, colour.astype(np.uint8)


def test_unproject_then_project_round_trips():
    c2w = rollout(np.eye(4), ["turn_left", "move_forward", "move_forward"])[-1]
    depth = np.random.default_rng(0).uniform(1, 5, (H, W))
    pts = unproject_depth(depth, c2w, K).reshape(-1, 3)
    proj = project(pts, c2w, K, W, H)
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    assert np.allclose(proj.uv[:, 0], u.ravel(), atol=1e-6)
    assert np.allclose(proj.uv[:, 1], v.ravel(), atol=1e-6)
    assert np.allclose(proj.depth, depth.ravel())


def test_w2c_to_c2w_inverts():
    c2w = rollout(np.eye(4), ["turn_right"] * 3 + ["move_forward"] * 2)[-1]
    assert np.allclose(w2c_to_c2w(np.linalg.inv(c2w)[:3]), c2w)


def make_recon(poses):
    depths, images = zip(*(wall_depth(p) for p in poses), strict=True)
    conf = np.ones((len(poses), H, W))
    return Reconstruction(
        np.stack(images), np.stack(depths), conf, np.stack(poses), np.stack([K] * len(poses))
    )


def test_fused_points_lie_on_the_true_surface_and_cross_render_agrees():
    poses = [np.eye(4), rollout(np.eye(4), ["turn_right", "move_forward"])[-1]]
    recon = make_recon(poses)
    cloud = fuse(recon, keep_fraction=1.0)
    assert np.allclose(cloud.points[:, 2], 4.0)
    assert set(cloud.frame.tolist()) == {0, 1}
    # Render frame 0's view using only frame 1's points: where covered, depth must match.
    one = fuse(recon, keep_fraction=1.0, frames=[1])
    view = render_points(one.points, one.colors, poses[0], K, W, H, point_radius=1)
    assert view.coverage > 0.6
    err = np.abs(view.depth[view.mask] - recon.depth[0][view.mask])
    assert np.median(err) < 0.05


def test_confidence_mask_keeps_requested_fraction_and_drops_bad_depth():
    recon = make_recon([np.eye(4)])
    conf = np.random.default_rng(0).random((1, H, W))
    depth = recon.depth.copy()
    depth[0, 0, 0] = 0.0
    recon = Reconstruction(recon.images, depth, conf, recon.c2w, recon.K)
    mask = confidence_mask(recon, 0.25)
    assert mask.mean() == pytest.approx(0.25, abs=0.01)
    assert not confidence_mask(recon, 1.0)[0, 0, 0]
    with pytest.raises(ValueError):
        confidence_mask(recon, 0.0)


def test_footprint_radii_close_a_surface_viewed_from_nearer():
    from mindseye.imagine.geo_render import footprint_radii

    recon = make_recon([np.eye(4)])
    cloud = fuse(recon, keep_fraction=1.0)
    near = np.eye(4)
    near[2, 3] = 3.0  # 1 m from the wall instead of 4 m: each point now spans about 4 pixels
    fixed = render_points(cloud.points, cloud.colors, near, K, W, H, point_radius=0)
    radii = footprint_radii(cloud.points, cloud.depth, near)
    adaptive = render_points(cloud.points, cloud.colors, near, K, W, H, point_radius=radii)
    assert set(np.unique(radii)) == {2}
    assert fixed.coverage < 0.2 and adaptive.coverage == 1.0
    assert np.allclose(adaptive.depth, 1.0)
    # From the original pose nothing needs enlarging.
    assert footprint_radii(cloud.points, cloud.depth, np.eye(4)).max() == 0


def test_view_coverage_distinguishes_seen_from_unseen_views():
    from mindseye.geometry.coverage import sparse_view_coverage, view_coverage_from

    front = np.eye(4)
    nearby = rollout(np.eye(4), ["turn_right", "move_forward"])[-1]
    away = rollout(np.eye(4), ["turn_right"] * 7)[-1]  # 63 degrees: mostly new wall
    recon = make_recon([front, nearby, away])
    close = view_coverage_from(recon, [0], 1, keep_fraction=1.0)
    far = view_coverage_from(recon, [0], 2, keep_fraction=1.0)
    assert close["coverage"] > 0.8 and close["consistent_coverage"] > 0.8
    assert far["consistent_coverage"] < 0.5 * close["consistent_coverage"]
    stats = sparse_view_coverage(recon, observed=[0, 1], heldout=[2], keep_fraction=1.0)
    assert stats["observed_overlap_consistent"] > 0.7
    assert (
        stats["heldout_coverage_consistent"] == far["consistent_coverage"]
        or stats["n_heldout"] == 1
    )


def test_vggt_preprocess_matches_reference_sizes():
    from mindseye.geometry.vggt_model import preprocess

    out = preprocess([np.zeros((480, 640, 3), np.uint8)] * 2)
    assert out.shape == (2, 392, 518, 3)  # the size VGGT's own loader produced for these frames
    assert preprocess([np.zeros((1920, 1440, 3), np.uint8)]).shape == (1, 518, 518, 3)


def test_svc_fit_to_and_intrinsics_agree():
    from mindseye.imagine.svc_model import fit_intrinsics, fit_to

    img = np.zeros((392, 518, 3), np.uint8)
    img[196, 259] = 255  # mark the principal point
    out = fit_to(img, 512, 384)
    assert out.shape == (384, 512, 3)
    K = np.array([[438.0, 0, 259.0], [0, 439.0, 196.0], [0, 0, 1]])
    K2 = fit_intrinsics(K, 518, 392, 512, 384)
    # The marked pixel must land where the new principal point says it is.
    v, u = np.unravel_index(out[..., 0].argmax(), out.shape[:2])
    assert abs(u - K2[0, 2]) <= 1.5 and abs(v - K2[1, 2]) <= 1.5
    assert K2[0, 0] == pytest.approx(438.0 * 512 / 518, rel=0.02)
