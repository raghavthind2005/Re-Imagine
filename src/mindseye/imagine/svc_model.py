"""Stable Virtual Camera (SVC) as a world model, driven by arbitrary camera poses.

Wraps ``seva.eval.run_one_scene`` from Stability-AI/stable-virtual-camera (pinned
commit, non-commercial licence). Unlike the official demo's preset moves, this
takes any number of posed input views and any target poses, so it serves both
MindJourney-style trajectories (turns and translations from a start view) and
"render this held-out camera" requests.

With two or more input views SVC normalises translations by the input camera
layout, so poses from an up-to-scale reconstruction can be used directly. With a
single input view the translation scale is not determined by the inputs
(``camera_scale`` sets it), which is the source of the forward-distance errors
MindJourney reports.

Three patches are needed on a Kaggle T4 (applied by the job, not here): allow
non-flash attention kernels, load the VAE from a mirror of SD 2.1 base, and
remove torchao.
"""

from __future__ import annotations

import glob
import hashlib
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass

import numpy as np

from mindseye.utils.cache import DiskCache, Pending, cache_key


@dataclass(frozen=True)
class SVCSettings:
    height: int = 384
    width: int = 512
    frames_per_pass: int = 21  # T: fewer than 21 produced artefacts in the spike
    num_steps: int = 20
    cfg: float = 2.0
    camera_scale: float = 2.0
    dtype: str = "float16"
    lowvram: bool = True
    seed: int = 0
    weights_version: str = "1.1"


def fit_to(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize to cover (width, height) and centre-crop, as SVC's "crop" mode does."""
    from PIL import Image

    h, w = image.shape[:2]
    scale = max(width / w, height / h)
    new_w, new_h = max(width, round(w * scale)), max(height, round(h * scale))
    resized = np.asarray(Image.fromarray(image).resize((new_w, new_h), Image.Resampling.BICUBIC))
    top, left = (new_h - height) // 2, (new_w - width) // 2
    return resized[top : top + height, left : left + width]


def fit_intrinsics(K: np.ndarray, w: int, h: int, width: int, height: int) -> np.ndarray:
    """Intrinsics of the image returned by ``fit_to``."""
    scale = max(width / w, height / h)
    new_w, new_h = max(width, round(w * scale)), max(height, round(h * scale))
    out = K.astype(np.float64).copy()
    out[0] *= new_w / w
    out[1] *= new_h / h
    out[0, 2] -= (new_w - width) // 2
    out[1, 2] -= (new_h - height) // 2
    return out


def render_key(settings: SVCSettings, images, input_c2ws, input_Ks, target_c2ws, target_Ks) -> str:
    """Cache key of one render: input frames, all camera poses and the SVC settings."""
    return cache_key(
        model="svc",
        settings=asdict(settings),
        images=[hashlib.sha256(np.ascontiguousarray(im).tobytes()).hexdigest() for im in images],
        input_c2ws=np.round(np.asarray(input_c2ws), 6),
        input_Ks=np.round(np.asarray(input_Ks), 4),
        target_c2ws=np.round(np.asarray(target_c2ws), 6),
        target_Ks=np.round(np.asarray(target_Ks), 4),
    )


class SVCCacheOnly:
    """Stands in for ``SVCRunner`` when the model is not loaded: cached renders only."""

    def __init__(self, settings: SVCSettings, cache: DiskCache) -> None:
        self.settings, self.cache = settings, cache

    def render(self, input_images, input_c2ws, input_Ks, target_c2ws, target_Ks):
        key = render_key(self.settings, input_images, input_c2ws, input_Ks, target_c2ws, target_Ks)
        if not self.cache.has(key, ".npz"):
            raise Pending("svc")
        return list(np.load(self.cache._path(key, ".npz"))["frames"])


class SVCRunner:
    def __init__(
        self,
        svc_repo: str,
        weights_dir: str,
        settings: SVCSettings | None = None,
        cache: DiskCache | None = None,
        device: str = "cuda:0",
    ) -> None:
        import sys

        import torch

        if svc_repo not in sys.path:
            sys.path.insert(0, svc_repo)
        import seva.eval as seva_eval
        from seva.model import SGMWrapper
        from seva.modules.autoencoder import AutoEncoder
        from seva.modules.conditioner import CLIPConditioner
        from seva.sampling import DiscreteDenoiser
        from seva.utils import load_model

        self.settings = settings or SVCSettings()
        self.cache = cache
        self._torch = torch
        self._eval = seva_eval
        seva_eval.set_lowvram_mode(self.settings.lowvram)
        s = self.settings
        net = load_model(
            model_version=float(s.weights_version),
            pretrained_model_name_or_path=weights_dir,
            weight_name=f"modelv{s.weights_version}.safetensors",
            device="cpu",
        ).eval()
        if s.dtype != "bfloat16":
            net = net.to(getattr(torch, s.dtype))
        self.model = SGMWrapper(net).to(device)
        self.ae = AutoEncoder(chunk_size=1).to(device)
        self.conditioner = CLIPConditioner().to(device)
        self.denoiser = DiscreteDenoiser(num_idx=1000, device=device)

    def _key(self, images, input_c2ws, input_Ks, target_c2ws, target_Ks) -> str:
        return render_key(self.settings, images, input_c2ws, input_Ks, target_c2ws, target_Ks)

    def render(
        self,
        input_images: list[np.ndarray],
        input_c2ws: np.ndarray,
        input_Ks: np.ndarray,
        target_c2ws: np.ndarray,
        target_Ks: np.ndarray,
    ) -> list[np.ndarray]:
        """Imagine the views at ``target_c2ws`` given posed input views.

        Images are (H, W, 3) uint8; intrinsics are in pixels of the image they
        belong to (targets use the input image size). Returns one uint8 image of
        size (settings.height, settings.width) per target, in order.
        """
        torch, s = self._torch, self.settings
        n_in, n_out = len(input_images), len(target_c2ws)
        if n_in + n_out > s.frames_per_pass:
            raise ValueError(f"{n_in} inputs + {n_out} targets exceed {s.frames_per_pass} per pass")
        key = self._key(input_images, input_c2ws, input_Ks, target_c2ws, target_Ks)
        if self.cache is not None and self.cache.has(key, ".npz"):
            path = self.cache._path(key, ".npz")
            return list(np.load(path)["frames"])

        c2ws = np.concatenate([np.asarray(input_c2ws), np.asarray(target_c2ws)])[:, :3]
        Ks = np.concatenate([np.asarray(input_Ks), np.asarray(target_Ks)])
        version = {
            "H": s.height, "W": s.width, "T": s.frames_per_pass, "C": 4, "f": 8,
            "options": {
                "chunk_strategy": "nearest-gt", "video_save_fps": 30.0,
                "beta_linear_start": 5e-6, "log_snr_shift": 2.4, "guider_types": 1,
                "cfg": s.cfg, "camera_scale": s.camera_scale, "num_steps": s.num_steps,
                "cfg_min": 1.2, "encoding_t": 1, "decoding_t": 1,
            },
        }  # fmt: skip
        out_dir = tempfile.mkdtemp(prefix="svc_")
        try:
            generator = self._eval.run_one_scene(
                "img2img",
                version,
                model=self.model,
                ae=self.ae,
                conditioner=self.conditioner,
                denoiser=self.denoiser,
                image_cond={
                    "img": list(input_images) + [None] * n_out,
                    "input_indices": list(range(n_in)),
                    "prior_indices": None,
                },
                camera_cond={
                    "c2w": torch.tensor(c2ws).float(),
                    "K": torch.tensor(Ks).float(),
                    "input_indices": list(range(n_in + n_out)),
                },
                save_path=out_dir,
                use_traj_prior=False,
                traj_prior_Ks=None,
                traj_prior_c2ws=None,
                seed=s.seed,
            )
            for _ in generator:
                pass
            from PIL import Image

            paths = sorted(glob.glob(os.path.join(out_dir, "samples-rgb", "*.png")))
            frames = [np.asarray(Image.open(p).convert("RGB")) for p in paths]
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)
        if len(frames) != n_out:
            raise RuntimeError(f"SVC returned {len(frames)} views for {n_out} target poses")
        if self.cache is not None:
            path = self.cache._path(key, ".npz")
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, frames=np.stack(frames))
            os.replace(tmp, path)
        return frames
