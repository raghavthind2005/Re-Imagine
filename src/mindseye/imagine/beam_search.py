"""Spatial beam search over imagined camera trajectories.

Follows MindJourney (arXiv 2507.12508, Algorithm 1 and appendix A.1.1):

- actions are move-forward, turn-left, turn-right;
- each expansion extends a beam node by 1..k repetitions of one action, so a node
  yields up to 3k candidates from three world-model calls (one per action, each
  rendering the k intermediate views);
- candidates that immediately reverse the previous turn, or exceed the maximum
  trajectory length, are dropped before rendering;
- a scorer gives each candidate an exploration score and a helpfulness score;
  candidates under a threshold are dropped; the top B by exploration form the
  next beam and the top H by helpfulness join the evidence buffer.

Scoring is behind an interface so the search logic can be tested without a VLM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from mindseye.geometry.camera import Action, describe, rollout
from mindseye.imagine.world_model import ImaginedView, WorldModel

ACTIONS: tuple[Action, ...] = ("move_forward", "turn_left", "turn_right")
_REVERSE = {"turn_left": "turn_right", "turn_right": "turn_left"}


@dataclass(frozen=True)
class SearchConfig:
    """Defaults are MindJourney's reported settings, except ``n_helpful``.

    The paper does not state H (evidence views kept per step); 2 is our placeholder,
    to be set on dev.
    """

    n_steps: int = 3
    max_repeat: int = 3  # k
    beam_width: int = 2  # B
    n_helpful: int = 2  # H: evidence views kept per step
    explore_threshold: float = 8.0
    helpful_threshold: float = 8.0
    max_trajectory_len: int = 8
    forward_m: float = 0.25
    turn_deg: float = 9.0
    # Distance named in trajectory descriptions, when ``forward_m`` is in map units
    # rather than metres (an up-to-scale reconstruction). None: use ``forward_m``.
    forward_label_m: float | None = None


@dataclass(frozen=True)
class Candidate:
    actions: tuple[Action, ...]
    view: ImaginedView
    description: str
    explore_score: float = float("nan")
    helpful_score: float = float("nan")


class Scorer(Protocol):
    def score(self, question: str, candidates: list[Candidate]) -> list[tuple[float, float]]:
        """(exploration score, helpfulness score) for each candidate, in order."""
        ...


@dataclass
class SearchResult:
    evidence: list[Candidate] = field(default_factory=list)
    n_world_model_calls: int = 0
    n_views_rendered: int = 0
    steps_run: int = 0
    stopped_early: bool = False


def expansions(
    trajectory: tuple[Action, ...], cfg: SearchConfig
) -> dict[Action, list[tuple[Action, ...]]]:
    """Allowed extensions of a trajectory, grouped by action: 1..k repeats each."""
    out: dict[Action, list[tuple[Action, ...]]] = {}
    last = trajectory[-1] if trajectory else None
    room = cfg.max_trajectory_len - len(trajectory)
    for action in ACTIONS:
        if last is not None and _REVERSE.get(last) == action:
            continue  # would undo the previous turn
        repeats = min(cfg.max_repeat, room)
        if repeats > 0:
            out[action] = [trajectory + (action,) * r for r in range(1, repeats + 1)]
    return out


def beam_search(
    question: str,
    start_c2w: np.ndarray,
    world_model: WorldModel,
    scorer: Scorer,
    cfg: SearchConfig | None = None,
) -> SearchResult:
    cfg = cfg or SearchConfig()
    steps = {"forward_m": cfg.forward_m, "turn_deg": cfg.turn_deg}
    label_steps = {**steps, "forward_m": cfg.forward_label_m or cfg.forward_m}
    result = SearchResult()
    beam: list[tuple[Action, ...]] = [()]
    for _ in range(cfg.n_steps):
        candidates: list[Candidate] = []
        # One path per (beam node, action): the longest extension's path passes
        # through the shorter ones, so it renders all repeats of that action.
        jobs = [
            (extended, rollout(start_c2w, list(extended[-1]), **steps)[len(trajectory) :])
            for trajectory in beam
            for extended in expansions(trajectory, cfg).values()
        ]
        if hasattr(world_model, "render_batch"):
            # A generative model renders a fixed number of frames per pass, so all
            # paths of this step go into as few passes as possible.
            rendered = world_model.render_batch(start_c2w, [poses for _, poses in jobs])
            result.n_world_model_calls += 1 if jobs else 0
        else:
            rendered = [world_model.render(start_c2w, poses) for _, poses in jobs]
            result.n_world_model_calls += len(jobs)
        for (extended, _), views in zip(jobs, rendered, strict=True):
            result.n_views_rendered += len(views)
            for actions, view in zip(extended, views, strict=True):
                candidates.append(Candidate(actions, view, describe(list(actions), **label_steps)))
        if not candidates:
            result.stopped_early = True
            break
        result.steps_run += 1
        scores = scorer.score(question, candidates)
        if len(scores) != len(candidates):
            raise ValueError("scorer must return one score pair per candidate")
        scored = [
            Candidate(c.actions, c.view, c.description, float(e), float(h))
            for c, (e, h) in zip(candidates, scores, strict=True)
        ]
        helpful = [c for c in scored if c.helpful_score >= cfg.helpful_threshold]
        helpful.sort(key=lambda c: -c.helpful_score)  # stable: ties keep generation order
        result.evidence.extend(helpful[: cfg.n_helpful])
        explore = [c for c in scored if c.explore_score >= cfg.explore_threshold]
        explore.sort(key=lambda c: -c.explore_score)
        beam = [c.actions for c in explore[: cfg.beam_width]]
        if not beam:
            result.stopped_early = True
            break
    return result
