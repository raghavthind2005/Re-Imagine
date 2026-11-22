"""World-model interface: given a start view and a camera path, produce the views along it."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class ImaginedView:
    image: np.ndarray  # (H, W, 3) uint8
    c2w: np.ndarray  # pose the view was rendered for
    # Fraction of the view backed by observed geometry; None when the source is
    # generative and makes no such statement.
    coverage: float | None = None


class WorldModel(Protocol):
    name: str

    def render(self, start_c2w: np.ndarray, poses: list[np.ndarray]) -> list[ImaginedView]:
        """Views at each pose of a path that starts from ``start_c2w``."""
        ...
