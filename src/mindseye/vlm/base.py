"""VLM backend interface. Every backend goes through the response cache."""

from __future__ import annotations

import hashlib
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from mindseye.utils.cache import DiskCache, Pending, cache_key

Image = str | Path | np.ndarray


@dataclass
class VLMResponse:
    text: str
    raw: dict[str, Any] = field(default_factory=dict)
    latency_s: float = 0.0
    cost_est: float = 0.0
    cached: bool = False
    cache_key: str = ""


def image_fingerprint(image: Image) -> str:
    """Content hash of an image, whether given as a path or an array."""
    if isinstance(image, np.ndarray):
        h = hashlib.sha256(str((image.dtype, image.shape)).encode())
        h.update(np.ascontiguousarray(image).tobytes())
        return h.hexdigest()
    return hashlib.sha256(Path(image).read_bytes()).hexdigest()


class VLMBackend(ABC):
    """Subclasses implement ``_generate``; callers use ``generate``."""

    model_id: str

    def __init__(self, cache: DiskCache | None = None) -> None:
        self.cache = cache

    def config_fingerprint(self) -> dict[str, Any]:
        """Settings that change outputs (precision, quantisation, image size)."""
        return {}

    @abstractmethod
    def _generate(
        self, images: list[Image], prompt: str, *, max_tokens: int, temperature: float, seed: int
    ) -> VLMResponse: ...

    def generate(
        self,
        images: list[Image],
        prompt: str,
        *,
        prompt_version: str,
        max_tokens: int = 16,
        temperature: float = 0.0,
        seed: int = 0,
    ) -> VLMResponse:
        key = cache_key(
            model_id=self.model_id,
            backend=self.config_fingerprint(),
            prompt_version=prompt_version,
            prompt=prompt,
            images=[image_fingerprint(im) for im in images],
            params={"max_tokens": max_tokens, "temperature": temperature, "seed": seed},
        )
        if self.cache is not None:
            hit = self.cache.get_json(key)
            if hit is not None:
                return VLMResponse(**{**hit, "cached": True, "cache_key": key})
        start = time.perf_counter()
        resp = self._generate(
            images, prompt, max_tokens=max_tokens, temperature=temperature, seed=seed
        )
        resp.latency_s = time.perf_counter() - start
        resp.cache_key = key
        if self.cache is not None:
            self.cache.put_json(key, asdict(resp))
        return resp


class MockVLM(VLMBackend):
    """Deterministic stand-in for tests and dry runs: always answers ``reply``."""

    def __init__(self, reply: str = "A", cache: DiskCache | None = None) -> None:
        super().__init__(cache)
        self.model_id = "mock"
        self.reply = reply
        self.calls = 0

    def config_fingerprint(self) -> dict[str, Any]:
        return {"reply": self.reply}

    def _generate(self, images, prompt, *, max_tokens, temperature, seed) -> VLMResponse:
        self.calls += 1
        return VLMResponse(text=self.reply)


class CacheOnlyVLM(VLMBackend):
    """Stands in for a backend that is not loaded: answers from the cache or raises ``Pending``.

    ``model_id`` and ``fingerprint`` must equal those of the real backend, so the
    cache keys match.
    """

    def __init__(self, model_id: str, fingerprint: dict[str, Any], cache: DiskCache) -> None:
        super().__init__(cache)
        self.model_id = model_id
        self._fingerprint = fingerprint

    def config_fingerprint(self) -> dict[str, Any]:
        return self._fingerprint

    def _generate(self, images, prompt, *, max_tokens, temperature, seed) -> VLMResponse:
        raise Pending("vlm")
