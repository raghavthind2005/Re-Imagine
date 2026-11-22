import numpy as np
import pytest

from mindseye.geometry.camera import (
    apply_action,
    describe,
    intrinsics,
    project,
    rollout,
    world_to_camera,
)
from mindseye.imagine.geo_render import render_points

W, H = 64, 48
K = intrinsics(W, H, fov_x_deg=90)
IDENTITY = np.eye(4)


# --- camera and actions -------------------------------------------------------


def test_intrinsics_and_projection_of_axis_point():
    assert K[0, 0] == pytest.approx(W / 2)  # 90 degree fov: f = W / 2
    proj = project(
        np.array([[0.0, 0.0, 2.0], [0.0, 0.0, -2.0], [5.0, 0.0, 1.0]]), IDENTITY, K, W, H
    )
    assert proj.uv[0] == pytest.approx([W / 2, H / 2])
    assert proj.in_view.tolist() == [True, False, False]  # centre, behind, far off to the side


def test_actions_move_and_turn_the_expected_way():
    fwd = apply_action(IDENTITY, "move_forward")
    assert fwd[:3, 3] == pytest.approx([0, 0, 0.25])
    # A point straight ahead appears on the LEFT of the image after turning RIGHT.
    ahead = np.array([[0.0, 0.0, 5.0]])
    right = apply_action(IDENTITY, "turn_right")
    left = apply_action(IDENTITY, "turn_left")
    assert world_to_camera(ahead, right)[0, 0] < 0
    assert world_to_camera(ahead, left)[0, 0] > 0
    assert right[:3, 3] == pytest.approx([0, 0, 0])  # turning in place


def test_ten_right_turns_make_a_quarter_circle_then_forward_goes_right():
    poses = rollout(IDENTITY, ["turn_right"] * 10 + ["move_forward"] * 4)
    assert len(poses) == 14
    assert poses[-1][:3, 3] == pytest.approx([1.0, 0, 0], abs=1e-9)  # 4 * 0.25 m along +x
    back = rollout(poses[9], ["turn_left"] * 10)[-1]
    assert back[:3, :3] == pytest.approx(np.eye(3), abs=1e-9)


def test_describe_merges_repeats():
    assert describe(["move_forward", "move_forward", "turn_left", "turn_right", "turn_right"]) == (
        "move forward 0.5 m, then turn left 9 degrees, then turn right 18 degrees"
    )
    assert describe([]) == "stay in place"


# --- renderer -----------------------------------------------------------------


def test_render_matches_brute_force_zbuffer():
    rng = np.random.default_rng(1)
    pts = rng.uniform([-2, -1.5, 1], [2, 1.5, 6], (3000, 3))
    cols = rng.integers(0, 255, (3000, 3)).astype(np.uint8)
    view = render_points(pts, cols, IDENTITY, K, W, H, point_radius=0)
    proj = project(pts, IDENTITY, K, W, H)
    depth = np.full((H, W), np.inf)
    image = np.zeros((H, W, 3), dtype=np.uint8)
    for i in np.flatnonzero(proj.in_view):
        u, v = int(proj.uv[i, 0]), int(proj.uv[i, 1])
        if proj.depth[i] < depth[v, u]:
            depth[v, u] = proj.depth[i]
            image[v, u] = cols[i]
    assert np.array_equal(view.depth, depth)
    assert np.array_equal(view.image, image)
    assert view.coverage == pytest.approx(np.isfinite(depth).mean())


def test_render_near_surface_hides_far_surface_and_reports_holes():
    xs, ys = np.meshgrid(np.linspace(-0.5, 0.5, 60), np.linspace(-0.5, 0.5, 60))
    near = np.stack([xs.ravel(), ys.ravel(), np.full(xs.size, 2.0)], axis=1)
    far = np.stack([xs.ravel() * 3, ys.ravel() * 3, np.full(xs.size, 6.0)], axis=1)
    pts = np.concatenate([far, near])
    cols = np.concatenate([np.full((len(far), 3), 50), np.full((len(near), 3), 200)]).astype(
        np.uint8
    )
    view = render_points(pts, cols, IDENTITY, K, W, H, point_radius=1)
    assert view.image[H // 2, W // 2].tolist() == [200, 200, 200]
    assert view.depth[H // 2, W // 2] == pytest.approx(2.0)
    assert 0 < view.coverage < 1
    assert not view.mask[0, 0] and np.isinf(view.depth[0, 0])
    # Looking the other way, nothing is observed: the view is all holes.
    behind = rollout(IDENTITY, ["turn_right"] * 20)[-1]
    assert render_points(pts, cols, behind, K, W, H).coverage == 0.0


# --- queries ------------------------------------------------------------------
