"""Unified example schema that every benchmark loader converts into."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Example:
    """One benchmark question.

    ``media`` holds paths to image frames or a single video path. ``choices`` is
    ``None`` for numerical-answer questions.
    """

    id: str
    benchmark: str
    split: str
    question: str
    choices: list[str] | None
    answer: str
    media: list[str]
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id or not self.benchmark or not self.split:
            raise ValueError("id, benchmark and split are required")
        if self.choices is not None and len(self.choices) < 2:
            raise ValueError(f"{self.id}: multiple-choice examples need at least two choices")
        if not self.media:
            raise ValueError(f"{self.id}: an example needs at least one image or video")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Example:
        return cls(**d)
