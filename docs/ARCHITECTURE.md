# Architecture

Two kinds of code: a small library (`src/mindseye/`) and scripts (`scripts/`) that run one
step each and write raw records. Every table and figure is produced by a script from those
records; nothing is edited by hand.

## Pipeline of the hallucination study

```
VSI-Bench video ── frames.py ──► 8 observed frames + 8 held-out frames
                                     │
     heldout_imagine.py              ▼
        VGGT (vggt_model.py) on all 16 frames ─► poses; map from the 8 observed only
        Stable Virtual Camera (svc_model.py) ─► imagined view at each held-out camera
        geo_render.py ─► map rendered at the same camera (coverage mask)
                                     │  views.npz per scene
     heldout_objects.py              ▼
        SAM 3 (sam3_model.py) on imagined, real and observed frames
        claim = "object X is visible from this camera"; map check → supported /
        contradicted / no coverage; region coverage of the claim
                                     │  views*.jsonl
     heldout_vlm_judge.py / heldout_vlm_scores.py
        Qwen3-VL: object present? on the real frame, the imagined frame (self-check),
        and the imagined frame with the observed frames; yes/no probabilities
                                     │  judgements.jsonl, context.jsonl, scores.jsonl
     human labels (make_label_sheet.py, make_test_label_sheet.py → CSV)
                                     ▼
     reports: hallucination_report.py, hallucination_headtohead.py, hallucination_curves.py,
              human_label_report.py, human_label_test_report.py, human_label_extra.py,
              make_summary.py (README table), make_figures.py
```

## Library (`src/mindseye/`)

| Package | Contents |
|---|---|
| `data/` | Unified `Example` schema; SAT and VSI-Bench loaders; fixed seeded splits; frame selection (observed / held-out) |
| `vlm/` | Backend interface with a content-addressed response cache; Qwen3-VL backend (generation, yes/no and option-letter probabilities); versioned prompts |
| `agent/` | `vlm_only` and `imagine` (MindJourney-style search) answer functions |
| `imagine/` | Spatial beam search; Stable Virtual Camera wrapper with a render cache; SVC as a world model; point-cloud renderer |
| `geometry/` | Camera maths and egocentric actions; VGGT wrapper; depth fusion; coverage; SAM 3 wrapper |
| `eval/` | Answer parsing, VSI-Bench metrics, bootstrap CIs (scene-clustered), McNemar, selective-prediction and ranking metrics, hallucination comparison |
| `utils/` | Cache keys, seeding, run directories (config, environment, resumable JSONL) |

## Running on Kaggle

`scripts/kaggle_launch.py --job kaggle/jobs/<job>.json` uploads the committed tree as a
private dataset, renders `kaggle/kernel_template.py` with the job, and pushes it as a
2×T4 script kernel. The kernel records the commit, fails fast without GPU or internet, and
kills each command at the job's own deadline so outputs are always saved.

## `imagine` mode

`scripts/run_imagine.py` runs in stages (geometry → VLM → SVC → VLM) because Qwen3-VL-8B,
SVC and VGGT do not fit on two T4s together. Each stage replays every question from the
caches and stops at the first result that belongs to a model it has not loaded.
