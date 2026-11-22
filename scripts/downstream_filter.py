"""Downstream check: does filtering imagined views remove imagination's harm?

    python scripts/downstream_filter.py --config <imagine config> --run <imagine run dir> \\
        --cache-dir <cache root with geometry/ and svc/> --out <dir> --stage objects
    python scripts/downstream_filter.py ... --stage answers

Pre-registered in docs/PREREGISTRATION.md, section 2. Uses the cached renders of an earlier
`imagine` run; nothing is rendered. Stage `objects` (VGGT, SAM 3) finds the objects in each
imagined evidence view and checks each against the labelled map of the observed frames.
Stage `answers` (Qwen) runs the ViSA-style self-check and answers every question four ways:

- `vlm_only`        observed frames only
- `imagine`         observed frames + all evidence views of the run
- `imagine_visa`    + only the evidence views whose objects the self-check confirms
- `imagine_ours`    + only the evidence views whose objects the labelled map confirms

A view is kept only if it shows at least one object and every object passes the check.
Confidence (multiple-choice only) is Qwen's probability of the chosen option letter; for
SAT with both option orders it is the lower of the two.
"""

# ruff: noqa: E501, E741, B023  (`ask` is called inside the loop iteration that defines it)
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml
from run_eval import load_examples, restrict_to_ids, subsample
from run_imagine import geometry_key, load_images

from mindseye.agent import imagine
from mindseye.agent.vlm_only import score, select_media
from mindseye.geometry.camera import project, unproject_depth
from mindseye.geometry.sam3_model import VSI_OBJECTS
from mindseye.imagine.beam_search import beam_search
from mindseye.imagine.geo_render import render_points
from mindseye.imagine.svc_model import SVCCacheOnly, SVCSettings
from mindseye.imagine.svc_world_model import SVCWorldModel
from mindseye.utils.cache import DiskCache
from mindseye.utils.runlog import append_jsonl, completed_ids
from mindseye.vlm.prompts import imagine as iprompts
from mindseye.vlm.prompts import presence, vqa

# Same values as scripts/heldout_objects.py (fixed before the pilot).
MIN_VISIBLE_FRACTION = 0.002
SAM_THRESHOLD = 0.5
MODES = ("vlm_only", "imagine", "imagine_visa", "imagine_ours")


class ReplayScorer:
    """Gives each candidate the scores the original run gave it, and keeps the candidates."""

    def __init__(self, scores: list[dict]) -> None:
        self.by_actions = {tuple(s["actions"]): (s["explore"], s["helpful"]) for s in scores}
        self.seen = []

    def score(self, question, candidates):
        self.seen += candidates
        return [self.by_actions[tuple(c.actions)] for c in candidates]


def labelled_map(recon, sam) -> tuple[np.ndarray, dict[str, list[np.ndarray]]]:
    threshold = np.quantile(recon.confidence, 0.3)
    cloud, by_label = [], {}
    for f in range(recon.n_frames):
        pts = unproject_depth(recon.depth[f].astype(np.float64), recon.c2w[f], recon.K[f])
        trusted = (recon.confidence[f] >= threshold) & (recon.depth[f] > 0)
        cloud.append(pts[trusted])
        for det in sam.detect(recon.images[f], VSI_OBJECTS, threshold=SAM_THRESHOLD):
            m = det.mask & trusted
            if m.any():
                by_label.setdefault(det.label, []).append(pts[m])
    return np.concatenate(cloud), by_label


def visible_pixels(cloud, by_label, label, c2w, K, W, H) -> int:
    zbuf = render_points(
        cloud, np.zeros((len(cloud), 3), np.uint8), c2w, K, W, H, point_radius=1
    ).depth
    n = 0
    for pts in by_label.get(label, []):
        proj = project(pts, c2w, K, W, H)
        uv = np.floor(proj.uv[proj.in_view]).astype(np.int64)
        n += int((proj.depth[proj.in_view] <= zbuf[uv[:, 1], uv[:, 0]] * 1.05).sum())
    return n


def stage_objects(examples, preds, cfg, args, out: Path) -> None:
    from mindseye.geometry.sam3_model import SAM3Runner
    from mindseye.geometry.vggt_model import VGGTRunner

    sam = SAM3Runner(args.sam3_weights, device="cuda:0")
    vggt = VGGTRunner(device="cuda:1")
    settings = SVCSettings(**cfg.get("svc", {}))
    svc = SVCCacheOnly(settings, DiskCache(Path(args.cache_dir) / "svc"))
    view_cfg, icfg = cfg["view"], cfg["imagine"]
    done = completed_ids(out / "objects.jsonl")
    maps: dict[str, tuple] = {}
    # Grouped by scene so each labelled map is built once.
    start = time.perf_counter()
    for ex in sorted(examples, key=lambda e: (geometry_key(e, view_cfg), e.id)):
        if ex.id in done:
            continue
        if args.max_seconds and time.perf_counter() - start > args.max_seconds:
            print("time limit reached in stage objects", flush=True)
            break
        t_q = time.perf_counter()
        rec = preds[ex.id]["imagine"]
        g = np.load(Path(args.cache_dir) / "geometry" / f"{geometry_key(ex, view_cfg)}.npz")
        wm = SVCWorldModel(svc, g["images"], g["c2w"], g["K"])
        scorer = ReplayScorer(rec["scores"])
        scfg = imagine.search_config(icfg, rec["forward_units"])
        start_c2w = g["c2w"][rec["start_frame"]]
        beam_search(imagine.question_text(ex), start_c2w, wm, scorer, scfg)
        wanted = [tuple(e["actions"]) for e in rec["evidence"]]
        views = {tuple(c.actions): c.view for c in scorer.seen}
        key = geometry_key(ex, view_cfg)
        if key not in maps:
            media, _ = select_media(ex, view_cfg)
            recon = vggt.reconstruct(load_images(media))
            pose_error = float(np.abs(recon.c2w - g["c2w"]).max())
            maps = {key: (*labelled_map(recon, sam), pose_error)}  # previous scene is done
        cloud, by_label, pose_error = maps[key]
        K = wm._target_K(start_c2w)
        H, W = settings.height, settings.width
        evidence, images = [], []
        for actions in wanted:
            view = views[actions]
            labels = sorted(
                {d.label for d in sam.detect(view.image, VSI_OBJECTS, threshold=SAM_THRESHOLD)}
            )
            vis = {lab: visible_pixels(cloud, by_label, lab, view.c2w, K, W, H) for lab in labels}
            evidence.append(
                {
                    "actions": list(actions),
                    "labels": labels,
                    "map_pixels_visible": vis,
                    "supported": {lab: v >= MIN_VISIBLE_FRACTION * H * W for lab, v in vis.items()},
                }
            )
            images.append(view.image)
        np.savez_compressed(out / "evidence" / f"{ex.id}.npz", images=np.stack(images))
        append_jsonl(
            out / "objects.jsonl",
            {
                "id": ex.id,
                "evidence": evidence,
                "map_labels": sorted(by_label),
                "pose_error": pose_error,
            },
        )
        print(
            ex.id,
            f"{time.perf_counter() - t_q:.1f}s",
            [e["supported"] for e in evidence],
            flush=True,
        )


def stage_answers(examples, preds, cfg, args, out: Path) -> None:
    from mindseye.vlm.qwen_local import QwenLocal

    vlm = QwenLocal(
        cfg["vlm"]["model_id"],
        dtype=cfg["vlm"].get("dtype", "float16"),
        max_pixels=cfg["vlm"].get("max_pixels"),
        cache=DiskCache(Path(args.cache_dir) / "vlm"),
    )
    gen = cfg["generation"]
    seed = cfg["seed"]
    view_cfg = cfg["view"]
    is_video = view_cfg["mode"] != "images"
    circular = view_cfg.get("circular", False)
    objects = {
        json.loads(l)["id"]: json.loads(l) for l in (out / "objects.jsonl").read_text().splitlines()
    }
    done = completed_ids(out / "predictions_imagine_ours.jsonl")
    start = time.perf_counter()
    for ex in examples:
        if ex.id in done or ex.id not in objects:
            continue
        if args.max_seconds and time.perf_counter() - start > args.max_seconds:
            print("time limit reached in stage answers", flush=True)
            break
        rec = preds[ex.id]["imagine"]
        media, view_info = select_media(ex, view_cfg)
        images = load_images(media)
        ev_images = list(np.load(out / "evidence" / f"{ex.id}.npz")["images"])
        ev = objects[ex.id]["evidence"]
        visa = []
        for e, im in zip(ev, ev_images, strict=True):
            p = {
                lab: vlm.yes_no([im], presence.single(lab), prompt_version=presence.VERSION)["p_no"]
                for lab in e["labels"]
            }
            visa.append(p)
        keep = {
            "vlm_only": [False] * len(ev),
            "imagine": [True] * len(ev),
            "imagine_visa": [
                bool(e["labels"]) and all(v < 0.5 for v in p.values())
                for e, p in zip(ev, visa, strict=True)
            ],
            "imagine_ours": [bool(e["labels"]) and all(e["supported"].values()) for e in ev],
        }
        for mode in MODES:
            kept = [i for i, k in enumerate(keep[mode]) if k]
            t0 = time.perf_counter()

            def ask(example):
                if kept:
                    preface = iprompts.evidence_preface(
                        len(images),
                        rec["start_frame"],
                        [rec["evidence"][i]["description"] for i in kept],
                    )
                    prompt = preface + "\n" + vqa.build_prompt(example, is_video=False)
                    version = f"{iprompts.VERSION}+{vqa.VERSION}"
                    imgs = images + [ev_images[i] for i in kept]
                else:
                    prompt, version, imgs = (
                        vqa.build_prompt(example, is_video=is_video),
                        vqa.VERSION,
                        images,
                    )
                resp = vlm.generate(imgs, prompt, prompt_version=version, max_tokens=gen["max_tokens"],
                                    temperature=gen["temperature"], seed=seed)  # fmt: skip
                scored = score(example, resp.text)
                conf = None
                if example.choices is not None:
                    letters = vqa.LETTERS[: len(example.choices)]
                    probs = vlm.option_probs(imgs, prompt, letters, prompt_version=version)
                    conf = probs.get(scored["pred"], 0.0) if scored["pred"] else 0.0
                return resp.text, scored, conf

            text, scored, conf = ask(ex)
            record = {
                "id": ex.id,
                "benchmark": ex.benchmark,
                "question_type": ex.meta["question_type"],
                "scene": ex.meta.get("scene"),
                "mode": mode,
                "response": text,
                **scored,
                "confidence": conf,
                "n_imagined_views": len(kept),
                "kept": keep[mode],
                "visa_p_no": visa,
                "view": view_info,
            }
            if circular and ex.choices is not None and len(ex.choices) == 2:
                text2, scored2, conf2 = ask(replace(ex, choices=ex.choices[::-1]))
                record.update(
                    response_flipped=text2,
                    score_first_order=scored["score"],
                    score_flipped_order=scored2["score"],
                    score_avg=(scored["score"] + scored2["score"]) / 2,
                    score=float(scored["score"] == 1.0 and scored2["score"] == 1.0),
                    confidence=min(conf, conf2),
                )
            record["seconds"] = round(time.perf_counter() - t0, 2)
            append_jsonl(out / f"predictions_{mode}.jsonl", record)
        print(
            ex.id,
            f"{time.perf_counter() - start:.0f}s total",
            {m: sum(keep[m]) for m in MODES},
            flush=True,
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument(
        "--run", required=True, help="run dir of the imagine run whose renders are reused"
    )
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", required=True, choices=["objects", "answers"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--sam3-weights", default="facebook/sam3")
    ap.add_argument(
        "--max-seconds", type=float, default=None, help="stop starting new questions after this"
    )
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    examples = subsample(restrict_to_ids(load_examples(cfg), cfg["data"]), args.limit, cfg["seed"])
    preds = {
        json.loads(l)["id"]: json.loads(l)
        for l in (Path(args.run) / "predictions.jsonl").read_text().splitlines()
    }
    examples = [e for e in examples if e.id in preds]
    out = Path(args.out)
    (out / "evidence").mkdir(parents=True, exist_ok=True)
    if args.stage == "objects":
        stage_objects(examples, preds, cfg, args, out)
    else:
        stage_answers(examples, preds, cfg, args, out)


if __name__ == "__main__":
    main()
