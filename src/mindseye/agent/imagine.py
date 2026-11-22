"""``imagine`` mode: MindJourney-style search with a world model, then answer.

1. The VLM picks the observed frame to start from (skipped for a single image).
2. Spatial beam search imagines views along short camera paths from that frame;
   the VLM scores each view for exploration and helpfulness.
3. The VLM answers from the observed frames plus the helpful imagined views.

When the search keeps no view, the final prompt is exactly the ``vlm_only`` one.
No verification happens here: this is the baseline our method is compared with.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from mindseye.agent.vlm_only import score
from mindseye.data.schema import Example
from mindseye.imagine.beam_search import Candidate, SearchConfig, beam_search
from mindseye.imagine.world_model import WorldModel
from mindseye.vlm.base import VLMBackend
from mindseye.vlm.prompts import imagine as prompts
from mindseye.vlm.prompts import vqa


@dataclass(frozen=True)
class SceneGeometry:
    """Poses of the observed frames, from the reconstruction of those frames only."""

    c2w: np.ndarray  # (F, 4, 4)
    K: np.ndarray  # (F, 3, 3)
    median_depth: np.ndarray  # (F,) in map units


class VLMScorer:
    def __init__(self, backend: VLMBackend, start_image: np.ndarray, seed: int) -> None:
        self.backend, self.start_image, self.seed = backend, start_image, seed
        self.log: list[dict[str, Any]] = []
        self.seen: list[Candidate] = []

    def score(self, question: str, candidates: list[Candidate]) -> list[tuple[float, float]]:
        out = []
        self.seen += candidates
        for c in candidates:
            resp = self.backend.generate(
                [self.start_image, c.view.image],
                prompts.score_view(question, c.description),
                prompt_version=prompts.VERSION,
                max_tokens=24,
                seed=self.seed,
            )
            parsed = prompts.parse_scores(resp.text)
            out.append(parsed or (0.0, 0.0))  # an unparsed reply keeps the view out
            self.log.append(
                {
                    "actions": list(c.actions),
                    "response": resp.text,
                    "parsed": parsed is not None,
                    "explore": out[-1][0],
                    "helpful": out[-1][1],
                    "latency_s": resp.latency_s,
                }
            )
        return out


def save_sheet(path: str | Path, start_image: np.ndarray, views: list[np.ndarray]) -> None:
    """Contact sheet: the start view, then the candidates in generation order."""
    from PIL import Image

    h, w = views[0].shape[:2]
    tiles = [np.asarray(Image.fromarray(start_image).resize((w, h)))] + list(views)
    cols = 5
    tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % cols)
    rows = [np.concatenate(tiles[i : i + cols], axis=1) for i in range(0, len(tiles), cols)]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate(rows, axis=0)[::2, ::2]).save(path)


def question_text(example: Example) -> str:
    if example.choices is None:
        return example.question
    return example.question + "\n" + vqa.options_block(example.choices)


def search_config(cfg: dict[str, Any], forward_units: float) -> SearchConfig:
    """Search settings from the config; the forward step is given in map units."""
    keys = ("n_steps", "max_repeat", "beam_width", "n_helpful", "explore_threshold",
            "helpful_threshold", "max_trajectory_len", "turn_deg")  # fmt: skip
    kwargs = {k: cfg[k] for k in keys if k in cfg}
    return SearchConfig(
        **kwargs, forward_m=forward_units, forward_label_m=cfg.get("forward_label_m", 0.25)
    )


def answer(
    example: Example,
    images: list[np.ndarray],
    view_info: dict[str, Any],
    geometry: SceneGeometry,
    world_model: WorldModel,
    backend: VLMBackend,
    cfg: dict[str, Any],
    gen_cfg: dict[str, Any],
    *,
    is_video: bool,
    circular: bool = False,
    sheet_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run one example and return its prediction record.

    ``sheet_path`` saves the start view beside every imagined candidate, for inspection.
    """
    seed = gen_cfg["seed"]
    question = question_text(example)
    n = len(images)

    start, start_reply = 0, None
    if n > 1:
        resp = backend.generate(
            images,
            prompts.pick_start(question, n),
            prompt_version=prompts.VERSION,
            max_tokens=4,
            seed=seed,
        )
        start_reply = resp.text
        picked = prompts.parse_frame(resp.text, n)
        start = n // 2 if picked is None else picked

    # MindJourney's 0.25 m step, as a share of the start view's median depth: the
    # reconstruction has no metric scale.
    forward_units = cfg["forward_depth_fraction"] * float(geometry.median_depth[start])
    scfg = search_config(cfg, forward_units)
    scorer = VLMScorer(backend, images[start], seed)
    result = beam_search(question, geometry.c2w[start], world_model, scorer, scfg)

    if sheet_path is not None and scorer.seen:
        save_sheet(sheet_path, images[start], [c.view.image for c in scorer.seen])

    # Best views first, one per trajectory, up to the cap.
    evidence: list[Candidate] = []
    for c in sorted(result.evidence, key=lambda c: -c.helpful_score):
        if all(c.actions != e.actions for e in evidence):
            evidence.append(c)
    evidence = evidence[: cfg.get("max_evidence", 4)]

    def ask(ex: Example):
        if evidence:
            preface = prompts.evidence_preface(n, start, [c.description for c in evidence])
            prompt = preface + "\n" + vqa.build_prompt(ex, is_video=False)
            version = f"{prompts.VERSION}+{vqa.VERSION}"
            media = images + [c.view.image for c in evidence]
        else:
            prompt, version, media = vqa.build_prompt(ex, is_video=is_video), vqa.VERSION, images
        resp = backend.generate(
            media,
            prompt,
            prompt_version=version,
            max_tokens=gen_cfg["max_tokens"],
            temperature=gen_cfg["temperature"],
            seed=seed,
        )
        return resp, score(ex, resp.text)

    resp, scored = ask(example)
    record = {
        "id": example.id,
        "benchmark": example.benchmark,
        "question_type": example.meta["question_type"],
        "scene": example.meta.get("scene"),
        "mode": "imagine",
        "response": resp.text,
        **scored,
        "latency_s": resp.latency_s,
        "cache_key": resp.cache_key,
        "raw": resp.raw,
        "view": view_info,
        "imagine": {
            "world_model": world_model.name,
            "start_frame": start,
            "start_reply": start_reply,
            "forward_units": forward_units,
            "steps_run": result.steps_run,
            "stopped_early": result.stopped_early,
            "n_world_model_calls": result.n_world_model_calls,
            "n_views_rendered": result.n_views_rendered,
            "evidence": [
                {"actions": list(c.actions), "description": c.description,
                 "explore": c.explore_score, "helpful": c.helpful_score}
                for c in evidence
            ],  # fmt: skip
            "scores": scorer.log,
        },
    }
    if circular and example.choices is not None and len(example.choices) == 2:
        resp2, scored2 = ask(replace(example, choices=example.choices[::-1]))
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
        )
    return record
