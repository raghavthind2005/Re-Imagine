"""Question-answering prompt templates.

Bump ``VERSION`` on any wording change: it is part of every cache key.

The wording follows the official VSI-Bench task (``lmms_eval/tasks/vsibench``):
the same pre-prompt, option block and post-prompts.
"""

from __future__ import annotations

from mindseye.data.schema import Example

VERSION = "vqa-v1"

LETTERS = "ABCDEFGH"
PRE_VIDEO = "These are frames of a video."
POST_MCA = "Answer with the option's letter from the given choices directly."
POST_NA = "Please answer the question using a single word or phrase."


def options_block(choices: list[str]) -> str:
    return "Options:\n" + "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(choices))


def build_prompt(example: Example, *, is_video: bool) -> str:
    parts = [PRE_VIDEO] if is_video else []
    parts.append(example.question)
    if example.choices is not None:
        parts += [options_block(example.choices), POST_MCA]
    else:
        parts.append(POST_NA)
    return "\n".join(parts)


def gold_letter(example: Example) -> str:
    if example.choices is None:
        raise ValueError(f"{example.id} is not multiple choice")
    return LETTERS[example.choices.index(example.answer)]
