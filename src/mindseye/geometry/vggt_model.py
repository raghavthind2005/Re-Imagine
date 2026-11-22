"""VGGT wrapper: frames in, a ``Reconstruction`` out.

Code: facebookresearch/vggt (pinned in the Kaggle jobs). Weights: facebook/VGGT-1B
(CC-BY-NC-4.0). Depth and cameras come from one joint forward pass over all the
frames given, in the first frame's coordinate system, up to an unknown scale.
"""

from __future__ import annotations

import numpy as np

from mindseye.geometry.camera import w2c_to_c2w
from mindseye.geometry.reconstruct import Reconstruction

TARGET_WIDTH = 518
PATCH = 14


def preprocess(frames: list[np.ndarray]) -> np.ndarray:
    """Resize frames the way VGGT's own loader does in "crop" mode.

    Width becomes 518, height the nearest multiple of 14 that keeps the aspect
    ratio (centre-cropped to 518 if taller). Returns (F, H, W, 3) uint8.
    """
    from PIL import Image

    out = []
    for frame in frames:
        h, w = frame.shape[:2]
        new_h = round(h * (TARGET_WIDTH / w) / PATCH) * PATCH
        img = np.asarray(
            Image.fromarray(frame).resize((TARGET_WIDTH, new_h), Image.Resampling.BICUBIC)
        )
        if new_h > TARGET_WIDTH:
            top = (new_h - TARGET_WIDTH) // 2
            img = img[top : top + TARGET_WIDTH]
        out.append(img)
    shapes = {im.shape for im in out}
    if len(shapes) != 1:
        raise ValueError(f"frames must share one aspect ratio, got {shapes}")
    return np.stack(out)


class VGGTRunner:
    def __init__(
        self, weights: str = "facebook/VGGT-1B", device: str = "cuda", dtype: str = "float16"
    ) -> None:
        import torch
        from vggt.models.vggt import VGGT

        self._torch = torch
        self.device = device
        self.dtype = getattr(torch, dtype)
        self.model = VGGT.from_pretrained(weights).eval().to(device)

    def reconstruct(self, frames: list[np.ndarray]) -> Reconstruction:
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        torch = self._torch
        images = preprocess(frames)
        x = torch.from_numpy(images).permute(0, 3, 1, 2).float().div(255).to(self.device)
        use_amp = self.device.startswith("cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=self.dtype, enabled=use_amp):
            pred = self.model(x)
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pred["pose_enc"].float(), x.shape[-2:])
        extrinsic = extrinsic[0].float().cpu().numpy()
        return Reconstruction(
            images=images,
            depth=pred["depth"][0, ..., 0].float().cpu().numpy(),
            confidence=pred["depth_conf"][0].float().cpu().numpy(),
            c2w=np.stack([w2c_to_c2w(e) for e in extrinsic]),
            K=intrinsic[0].float().cpu().numpy().astype(np.float64),
        )
