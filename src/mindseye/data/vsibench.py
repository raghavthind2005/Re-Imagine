"""VSI-Bench loader (HF ``nyu-visionx/VSI-Bench``).

Question types and scoring follow the official task in
``vision-x-nyu/thinking-in-space`` (``lmms_eval/tasks/vsibench``).

Source options come pre-lettered (``"A. left"``) with a letter as ground truth.
They are stored here as plain text with the answer as the correct choice's text,
the same convention as SAT; the prompt builder adds the letters back in order.
"""

from __future__ import annotations

import json
from collections.abc import Collection
from pathlib import Path

from mindseye.data.schema import Example

MCA_TYPES = {
    "object_rel_direction_easy",
    "object_rel_direction_medium",
    "object_rel_direction_hard",
    "object_rel_distance",
    "route_planning",
    "obj_appearance_order",
}
NA_TYPES = {
    "object_abs_distance",
    "object_counting",
    "object_size_estimation",
    "room_size_estimation",
}


LETTERS = "ABCDEFGH"


def strip_letter(option: str, index: int) -> str:
    """Turn ``"B. back-right"`` into ``"back-right"``, checking the letter is the expected one."""
    prefix = f"{LETTERS[index]}. "
    if not option.startswith(prefix):
        raise ValueError(f"option {option!r} does not start with {prefix!r}")
    return option[len(prefix) :]


def scene_key(dataset: str, scene_name: str) -> str:
    return f"{dataset}/{scene_name}"


def row_to_example(row: dict, split: str, video_root: Path) -> Example:
    qtype = row["question_type"]
    answer = str(row["ground_truth"])
    if qtype in MCA_TYPES:
        fmt, choices = "mca", [strip_letter(o, i) for i, o in enumerate(row["options"])]
        answer = choices[LETTERS.index(answer)]
    elif qtype in NA_TYPES:
        fmt, choices = "na", None
    else:
        raise ValueError(f"unknown VSI-Bench question type: {qtype}")
    scene = scene_key(row["dataset"], row["scene_name"])
    return Example(
        id=f"vsi-{row['id']}",
        benchmark="vsibench",
        split=split,
        question=row["question"],
        choices=choices,
        answer=answer,
        media=[str(video_root / f"{scene}.mp4")],
        meta={"question_type": qtype, "answer_format": fmt, "scene": scene},
    )


def read_rows(jsonl_path: str | Path) -> list[dict]:
    with open(jsonl_path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_vsibench(
    jsonl_path: str | Path,
    split: str,
    video_root: str | Path,
    scenes: Collection[str] | None = None,
) -> list[Example]:
    """Load VSI-Bench questions, optionally restricted to the given scene keys."""
    wanted = None if scenes is None else set(scenes)
    return [
        row_to_example(row, split, Path(video_root))
        for row in read_rows(jsonl_path)
        if wanted is None or scene_key(row["dataset"], row["scene_name"]) in wanted
    ]
