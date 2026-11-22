"""SAT loader (HF ``array/SAT``).

Source rows have: ``image_bytes`` (1-2 images), ``question``, ``answers`` (always
two), ``question_type`` and ``correct_answer``. The source has no ids, so the id is
the row index within its parquet file.

In the source the correct answer is always ``answers[0]``. Choices are therefore
re-ordered here with a per-example seeded coin flip, so a model that favours the
first option gains nothing.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path

from mindseye.data.schema import Example

# The same task has different names in the two SAT files.
TYPE_ALIASES = {"action_conseq": "action_consequence"}


def choice_order(example_id: str, n: int, seed: int) -> list[int]:
    """Deterministic permutation of ``range(n)`` that depends only on id and seed."""
    digest = hashlib.sha256(f"{seed}:{example_id}".encode()).digest()
    order = list(range(n))
    for i in range(n - 1, 0, -1):
        j = digest[i % len(digest)] % (i + 1)
        order[i], order[j] = order[j], order[i]
    return order


def row_to_example(
    row: dict, index: int, source_split: str, split: str, image_dir: Path, seed: int
) -> Example:
    ex_id = f"sat-{source_split}-{index}"
    answers = list(row["answers"])
    if row["correct_answer"] not in answers:
        raise ValueError(f"{ex_id}: correct answer is not among the choices")
    order = choice_order(ex_id, len(answers), seed)
    media = []
    for k, blob in enumerate(row["image_bytes"]):
        path = image_dir / source_split / f"{index}_{k}.png"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(blob)
        media.append(str(path))
    qtype = TYPE_ALIASES.get(row["question_type"], row["question_type"])
    return Example(
        id=ex_id,
        benchmark="sat",
        split=split,
        question=row["question"],
        choices=[answers[i] for i in order],
        answer=row["correct_answer"],
        media=media,
        meta={"question_type": qtype, "answer_format": "mca", "source_split": source_split},
    )


def load_sat(
    parquet_path: str | Path,
    source_split: str,
    split: str,
    image_dir: str | Path,
    indices: Iterable[int] | None = None,
    seed: int = 0,
) -> list[Example]:
    """Load SAT rows as Examples, extracting images to ``image_dir``.

    ``indices`` restricts to the given row numbers (used for the fixed dev subset).
    """
    import pyarrow.parquet as pq

    wanted = None if indices is None else set(indices)
    examples: list[Example] = []
    offset = 0
    for batch in pq.ParquetFile(parquet_path).iter_batches(batch_size=64):
        rows = batch.to_pylist()
        for i, row in enumerate(rows, start=offset):
            if wanted is None or i in wanted:
                examples.append(row_to_example(row, i, source_split, split, Path(image_dir), seed))
        offset += len(rows)
    return examples


def read_question_types(parquet_path: str | Path) -> list[str]:
    import pyarrow.parquet as pq

    col = pq.read_table(parquet_path, columns=["question_type"]).column("question_type")
    return [TYPE_ALIASES.get(t, t) for t in col.to_pylist()]
