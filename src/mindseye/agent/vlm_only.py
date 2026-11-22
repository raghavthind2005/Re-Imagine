"""``vlm_only`` mode: the VLM answers from the given images, nothing else."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from mindseye.data.frames import frame_count, read_frames, sparse_view_split, uniform_indices
from mindseye.data.schema import Example
from mindseye.eval.metrics import mean_relative_accuracy
from mindseye.eval.parse import official_first_token, official_number, parse_letter, parse_number
from mindseye.vlm.base import Image, VLMBackend
from mindseye.vlm.prompts import vqa


def select_media(example: Example, view_cfg: dict[str, Any]) -> tuple[list[Image], dict[str, Any]]:
    """Images the model is allowed to see, plus a record of how they were chosen.

    ``view_cfg["mode"]`` is ``images`` (still-image benchmarks), ``sparse`` (a few
    observed frames; the rest held out) or ``full`` (many evenly spaced frames).
    """
    mode = view_cfg["mode"]
    if mode == "images":
        return list(example.media), {"mode": mode}
    video = example.media[0]
    n = frame_count(video)
    if mode == "sparse":
        observed, heldout = sparse_view_split(n, view_cfg["n_observed"], view_cfg["n_heldout"])
    elif mode == "full":
        observed, heldout = uniform_indices(n, view_cfg["n_frames"]), []
    else:
        raise ValueError(f"unknown view mode: {mode}")
    info = {"mode": mode, "n_video_frames": n, "observed": observed, "heldout": heldout}
    return read_frames(video, observed), info


def score(example: Example, text: str) -> dict[str, Any]:
    """Score one reply with our parser and with the official VSI-Bench rule."""
    if example.choices is not None:
        gold = vqa.gold_letter(example)
        pred = parse_letter(text, len(example.choices))
        return {
            "pred": pred,
            "gold": gold,
            "parsed": pred is not None,
            "score": float(pred == gold),
            "score_official": float(official_first_token(text).lower() == gold.lower()),
        }
    gold_value = float(example.answer)
    pred_value = parse_number(text)
    return {
        "pred": pred_value,
        "gold": gold_value,
        "parsed": pred_value is not None,
        "score": mean_relative_accuracy(pred_value, gold_value),
        "score_official": mean_relative_accuracy(official_number(text), gold_value),
    }


def _ask(example: Example, images: list[Image], backend: VLMBackend, is_video: bool, gen_cfg):
    resp = backend.generate(
        images,
        vqa.build_prompt(example, is_video=is_video),
        prompt_version=vqa.VERSION,
        max_tokens=gen_cfg["max_tokens"],
        temperature=gen_cfg["temperature"],
        seed=gen_cfg["seed"],
    )
    return resp, score(example, resp.text)


def answer(
    example: Example,
    backend: VLMBackend,
    view_cfg: dict[str, Any],
    gen_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Run one example and return its prediction record (one JSONL line).

    With ``view_cfg["circular"]`` a two-choice question is asked twice, once per
    option order. ``score`` is then 1 only if both orders are answered correctly
    (strict circular evaluation); ``score_avg`` is the mean of the two.
    """
    images, view_info = select_media(example, view_cfg)
    is_video = view_cfg["mode"] != "images"
    resp, scored = _ask(example, images, backend, is_video, gen_cfg)
    record = {
        "id": example.id,
        "benchmark": example.benchmark,
        "question_type": example.meta["question_type"],
        "scene": example.meta.get("scene"),
        "mode": "vlm_only",
        "response": resp.text,
        **scored,
        "latency_s": resp.latency_s,
        "cached": resp.cached,
        "cache_key": resp.cache_key,
        "raw": resp.raw,
        "view": view_info,
    }
    if view_cfg.get("circular") and example.choices is not None and len(example.choices) == 2:
        flipped = replace(example, choices=example.choices[::-1])
        resp2, scored2 = _ask(flipped, images, backend, is_video, gen_cfg)
        record.update(
            response_flipped=resp2.text,
            pred_flipped=scored2["pred"],
            gold_flipped=scored2["gold"],
            score_first_order=scored["score"],
            score_flipped_order=scored2["score"],
            score_avg=(scored["score"] + scored2["score"]) / 2,
            score=float(scored["score"] == 1.0 and scored2["score"] == 1.0),
            parsed=scored["parsed"] and scored2["parsed"],
            latency_s=resp.latency_s + resp2.latency_s,
            cached=resp.cached and resp2.cached,
            cache_key_flipped=resp2.cache_key,
        )
    return record
