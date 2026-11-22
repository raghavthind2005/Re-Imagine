# Licences of models, data and code used

The code in this repository is MIT-licensed (`LICENSE`); most models and data it uses are
non-commercial. "Checked" means the licence was read at
the source named in the last column on or before 2026-10-15; "assumed" means it was not read in full and we act conservatively.

## Models

| Model | Used for | Licence | What it allows | Source |
|---|---|---|---|---|
| Qwen3-VL-8B-Instruct | VLM answering, scoring, presence judgements | Apache-2.0 (checked) | Research and commercial use with notice | HF model card tag |
| VGGT-1B (`facebook/VGGT-1B`, code at `a288dd0`) | Camera poses and depth | CC-BY-NC-4.0 (checked) | Non-commercial use with attribution | HF model card tag |
| SAM 3 (`facebook/sam3`) | Open-vocabulary object masks | SAM License, 2025-11-19 (checked, full text read) | Use, modification and redistribution under the same licence (copy of the licence must accompany redistributed SAM materials); publications must acknowledge SAM; trade-control restrictions apply | Licence file shipped with the weights |
| Stable Virtual Camera v1.1 (`stabilityai/stable-virtual-camera`, code at `fe19948`) | World model (imagined views) | Stability AI Non-Commercial License (checked) | Research and non-commercial use only | HF model card |
| Stable Diffusion 2.1 base VAE, via the `sd2-community/stable-diffusion-2-1-base` mirror | SVC's autoencoder (official repo returned 401) | CreativeML OpenRAIL++-M (checked tag; mirror provenance unverified) | Use with the licence's use restrictions | HF model card tag |

No model weights are committed to this repository.

## Data

| Dataset | Used for | Licence / terms | Imagery in this repo |
|---|---|---|---|
| VSI-Bench (`nyu-visionx/VSI-Bench`) | Questions and videos | Apache-2.0 for the benchmark release (checked HF tag); the videos come from ScanNet, ScanNet++ and ARKitScenes and keep their own terms | — |
| ARKitScenes (Apple), via VSI-Bench | Videos of 10 dev and 10 test scenes | Apple licence (checked): non-commercial use, reproduction and redistribution allowed, notice retained; separate commercial terms | **Yes**: the example figure, the SVC spike sheet and `demo/examples/` (37 dev views). Non-commercial use only; © Apple Inc., ARKitScenes |
| ScanNet, via VSI-Bench | Videos | Terms of use assumed to forbid redistribution of the data | **None** (checked 2026-11-22). The local labelling sheets that contain ScanNet frames are gitignored |
| ScanNet++, via VSI-Bench | Videos | Terms of use assumed to forbid redistribution of the data | **None.** As above |
| SAT (`array/SAT`) | Spatial questions (synthetic images) | MIT (checked HF tag) | One spike sheet (`results/svc_spike/variants_sat_dev.png`) |

Images in this repository (checked 2026-11-22): the SVC spike sheets (ARKitScenes, SAT), plots,
the ARKitScenes example figure and `demo/examples/` (ARKitScenes only). No ScanNet or ScanNet++ frame.

## Code

- MindJourney's search procedure was reimplemented from the paper; its repository has no
  licence file, so none of its code is included.
- This repository's own code: MIT (`LICENSE`). It does not relicense any model, dataset or
  image listed above.

## Acknowledgements required

- SAM 3: publications using these results must acknowledge the use of SAM materials.
- ARKitScenes and VGGT: cite the respective papers; non-commercial use only.
