"""Content-addressed disk cache for expensive calls.

Every expensive result (VLM response, imagined view, VGGT output, SAM 3 mask) is
stored under a key derived from everything that determines it: model ID, prompt
template version, inputs and generation parameters. Re-running an experiment
must never re-pay for an identical call.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


def _canonical(obj: Any) -> Any:
    """Convert ``obj`` into a JSON-serialisable form that is stable across runs."""
    if isinstance(obj, (str, int, bool)) or obj is None:
        return obj
    if isinstance(obj, float):
        return repr(obj)
    if isinstance(obj, (bytes, bytearray)):
        return {"__bytes_sha256__": hashlib.sha256(bytes(obj)).hexdigest()}
    if isinstance(obj, np.ndarray):
        digest = hashlib.sha256(np.ascontiguousarray(obj).tobytes()).hexdigest()
        return {"__ndarray__": [str(obj.dtype), list(obj.shape), digest]}
    if isinstance(obj, np.generic):
        return _canonical(obj.item())
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _canonical(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [_canonical(v) for v in obj]
    raise TypeError(f"cannot build a cache key from {type(obj).__name__}")


def cache_key(**parts: Any) -> str:
    """Return a deterministic SHA-256 hex key for the given named parts.

    Callers should always pass ``model_id``, ``prompt_version``, the inputs and
    the generation parameters. Key order does not matter.
    """
    payload = json.dumps(_canonical(parts), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class Pending(Exception):
    """A result this process cannot produce is missing from the cache.

    The staged imagine pipeline runs the GPU models one at a time; a stage that
    meets work belonging to another model raises this and moves on.
    """


class DiskCache:
    """Directory-backed cache of JSON records and binary blobs."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str, suffix: str) -> Path:
        return self.root / key[:2] / f"{key}{suffix}"

    def _write_atomic(self, path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            os.replace(tmp, path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    def has(self, key: str, suffix: str = ".json") -> bool:
        return self._path(key, suffix).exists()

    def get_json(self, key: str) -> Any | None:
        path = self._path(key, ".json")
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def put_json(self, key: str, value: Any) -> None:
        data = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self._write_atomic(self._path(key, ".json"), data)

    def get_bytes(self, key: str, suffix: str = ".bin") -> bytes | None:
        path = self._path(key, suffix)
        return path.read_bytes() if path.exists() else None

    def put_bytes(self, key: str, value: bytes, suffix: str = ".bin") -> None:
        self._write_atomic(self._path(key, suffix), value)
