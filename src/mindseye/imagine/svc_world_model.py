"""SVC behind the ``WorldModel`` interface, conditioned on the observed frames.

Every observed frame goes in as a posed input view, so SVC sees the same evidence
as the agent and normalises the translation scale from the camera layout. All
paths of one search step are rendered together, in as few passes as fit.
"""

from __future__ import annotations

import numpy as np

from mindseye.imagine.svc_model import fit_intrinsics, fit_to
from mindseye.imagine.world_model import ImaginedView


class SVCWorldModel:
    name = "svc"

    def __init__(self, runner, images: np.ndarray, c2ws: np.ndarray, Ks: np.ndarray) -> None:
        """``runner`` is an ``SVCRunner`` or ``SVCCacheOnly``; images are (F, H, W, 3)."""
        s = runner.settings
        h, w = images.shape[1:3]
        self.runner = runner
        self.images = [fit_to(im, s.width, s.height) for im in images]
        self.c2ws = np.asarray(c2ws)
        self.Ks = np.stack([fit_intrinsics(K, w, h, s.width, s.height) for K in Ks])
        self.per_pass = s.frames_per_pass - len(self.images)
        if self.per_pass < 1:
            raise ValueError("no room for target views beside the input frames")
        self.passes = 0

    def _target_K(self, start_c2w: np.ndarray) -> np.ndarray:
        """Intrinsics of the observed camera nearest to the start pose."""
        d = np.linalg.norm(self.c2ws[:, :3, 3] - np.asarray(start_c2w)[:3, 3], axis=1)
        return self.Ks[int(np.argmin(d))]

    def render_batch(
        self, start_c2w: np.ndarray, paths: list[list[np.ndarray]]
    ) -> list[list[ImaginedView]]:
        poses = [p for path in paths for p in path]
        K = self._target_K(start_c2w)
        frames: list[np.ndarray] = []
        for i in range(0, len(poses), self.per_pass):
            chunk = np.stack(poses[i : i + self.per_pass])
            frames += self.runner.render(
                self.images, self.c2ws, self.Ks, chunk, np.repeat(K[None], len(chunk), axis=0)
            )
            self.passes += 1
        views = [ImaginedView(f, p) for f, p in zip(frames, poses, strict=True)]
        out, i = [], 0
        for path in paths:
            out.append(views[i : i + len(path)])
            i += len(path)
        return out

    def render(self, start_c2w: np.ndarray, poses: list[np.ndarray]) -> list[ImaginedView]:
        return self.render_batch(start_c2w, [poses])[0]
