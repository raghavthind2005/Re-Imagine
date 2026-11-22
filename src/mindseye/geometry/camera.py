"""Camera conventions and the MindJourney action space.

Cameras use the OpenCV convention: x right, y down, z forward. A pose is a 4x4
camera-to-world matrix ``c2w``. ``K`` is a 3x3 pinhole intrinsics matrix in pixels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

Action = Literal["move_forward", "turn_left", "turn_right"]

# MindJourney's settings (paper appendix A.1.1): 0.25 m per forward step and
# 9 degrees per turn step.
FORWARD_STEP_M = 0.25
TURN_STEP_DEG = 9.0


def intrinsics(width: int, height: int, fov_x_deg: float) -> np.ndarray:
    """Pinhole intrinsics for a given horizontal field of view, principal point centred."""
    f = 0.5 * width / np.tan(np.radians(fov_x_deg) / 2)
    return np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]], dtype=np.float64)


def yaw(angle_deg: float) -> np.ndarray:
    """Rotation about the camera's vertical axis. Positive turns the view to the right."""
    a = np.radians(angle_deg)
    rot = np.eye(4)
    rot[:3, :3] = [[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]]
    return rot


def translate_forward(distance: float) -> np.ndarray:
    t = np.eye(4)
    t[2, 3] = distance
    return t


def apply_action(
    c2w: np.ndarray,
    action: Action,
    *,
    forward_m: float = FORWARD_STEP_M,
    turn_deg: float = TURN_STEP_DEG,
) -> np.ndarray:
    """Pose after one primitive egocentric action."""
    if action == "move_forward":
        return c2w @ translate_forward(forward_m)
    if action == "turn_left":
        return c2w @ yaw(-turn_deg)
    if action == "turn_right":
        return c2w @ yaw(turn_deg)
    raise ValueError(f"unknown action: {action}")


def rollout(c2w: np.ndarray, actions: list[Action], **steps: float) -> list[np.ndarray]:
    """Poses after each action of a trajectory (the start pose is not included)."""
    poses = []
    for action in actions:
        c2w = apply_action(c2w, action, **steps)
        poses.append(c2w)
    return poses


def describe(actions: list[Action], **steps: float) -> str:
    """Natural-language trajectory description, merging repeated actions."""
    forward = steps.get("forward_m", FORWARD_STEP_M)
    turn = steps.get("turn_deg", TURN_STEP_DEG)
    parts: list[str] = []
    i = 0
    while i < len(actions):
        j = i
        while j < len(actions) and actions[j] == actions[i]:
            j += 1
        n = j - i
        if actions[i] == "move_forward":
            parts.append(f"move forward {n * forward:g} m")
        else:
            side = "left" if actions[i] == "turn_left" else "right"
            parts.append(f"turn {side} {n * turn:g} degrees")
        i = j
    return ", then ".join(parts) if parts else "stay in place"


@dataclass(frozen=True)
class Projection:
    """Result of projecting world points into a camera."""

    uv: np.ndarray  # (N, 2) pixel coordinates
    depth: np.ndarray  # (N,) distance along the optical axis; <= 0 is behind the camera
    in_view: np.ndarray  # (N,) bool: in front of the camera and inside the image


def world_to_camera(points: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    w2c = np.linalg.inv(c2w)
    return points @ w2c[:3, :3].T + w2c[:3, 3]


def project(
    points: np.ndarray, c2w: np.ndarray, K: np.ndarray, width: int, height: int
) -> Projection:
    cam = world_to_camera(points, c2w)
    depth = cam[:, 2]
    safe = np.where(depth > 1e-9, depth, 1.0)
    uv = np.stack(
        [K[0, 0] * cam[:, 0] / safe + K[0, 2], K[1, 1] * cam[:, 1] / safe + K[1, 2]], axis=1
    )
    in_view = (
        (depth > 1e-9)
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < width)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < height)
    )
    return Projection(uv=uv, depth=depth, in_view=in_view)


def unproject_depth(depth: np.ndarray, c2w: np.ndarray, K: np.ndarray) -> np.ndarray:
    """World points for every pixel of a depth map (depth along the optical axis).

    Returns an (H, W, 3) array. Pixel (u, v) is taken at its integer coordinate,
    so ``project`` maps the result back to (u, v).
    """
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    x = (u - K[0, 2]) / K[0, 0] * depth
    y = (v - K[1, 2]) / K[1, 1] * depth
    cam = np.stack([x, y, depth], axis=-1)
    return cam @ c2w[:3, :3].T + c2w[:3, 3]


def w2c_to_c2w(extrinsic: np.ndarray) -> np.ndarray:
    """Invert a 3x4 (or 4x4) world-to-camera matrix into a 4x4 camera-to-world pose."""
    w2c = np.eye(4)
    w2c[:3, :4] = extrinsic[:3, :4]
    return np.linalg.inv(w2c)
