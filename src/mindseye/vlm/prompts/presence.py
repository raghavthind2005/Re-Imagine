"""Yes/no prompts about whether an object is present in a view.

Bump ``VERSION`` on any wording change: it is part of every cache key.

``single`` is asked about one image. On a real frame it is a second ground truth
beside the detector. On an imagined frame it is self-verification in the style of
ViSA (arXiv 2512.05809): the judge sees only the imagined view. ``with_context``
is a stronger variant of that baseline, which also shows the real observed frames.
"""

from __future__ import annotations

VERSION = "presence-v1"


def single(label: str) -> str:
    return f"Is there a {label} visible in this image? Answer with yes or no."


def with_context(label: str, n_real: int) -> str:
    return (
        f"The first {n_real} images are real photos of an indoor scene. The last image is a "
        "computer-generated view of the same scene from a new camera position, and it may "
        f"show objects that are not really there. The last image shows a {label}. "
        f"Judging from the real photos, is that {label} really present at that place in the "
        "scene? Answer with yes or no."
    )


def parse_yes_no(text: str) -> bool | None:
    word = text.strip().lower().lstrip("*\"' ").split(maxsplit=1)[0:1]
    if not word:
        return None
    if word[0].startswith("yes"):
        return True
    if word[0].startswith("no"):
        return False
    return None
