"""Turn raw model text into an answer.

Two parsers are kept side by side. ``official_*`` reproduces the VSI-Bench
reference scorer exactly, for comparison with published numbers. ``parse_*`` is
slightly more tolerant and is what our own experiments use.
"""

from __future__ import annotations

import re

_LETTER = re.compile(r"(?<![A-Za-z])\(?([A-H])\)?(?![A-Za-z])")
_NUMBER = re.compile(r"[-+]?(?:\d+\.\d+|\d+|\.\d+)")


def official_first_token(text: str) -> str:
    """The reference scorer's ``fuzzy_matching``: first space-separated token, minus a full stop."""
    return text.split(" ")[0].rstrip(".").strip()


def official_number(text: str) -> float | None:
    try:
        return float(official_first_token(text))
    except ValueError:
        return None


def parse_letter(text: str, n_choices: int) -> str | None:
    """First standalone option letter in the text, e.g. ``"B"``, ``"(B)"``, ``"Answer: B."``."""
    for m in _LETTER.finditer(text.strip()):
        letter = m.group(1)
        if ord(letter) - ord("A") < n_choices:
            return letter
    return None


def parse_number(text: str) -> float | None:
    m = _NUMBER.search(text.replace(",", ""))
    return float(m.group()) if m else None
