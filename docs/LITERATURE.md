# Literature notes

Written 2026-10-12, before any experiment here. Every number below was read in the source during
that session; the "Read" line under each paper says how much of it was read.
Nothing here is reproduced by us yet.

## 1. MindJourney (arXiv 2507.12508v2, NeurIPS 2025)

Read: full paper including appendix (arXiv HTML v2), repo README and scripts at
commit `aeed003` of `UMass-Embodied-AGI/MindJourney`.

Authors: Yang, Liu, Zhang, Zhou, Tan, Yang, Du, Gan. Lead affiliation is UMass
Amherst, with JHU, HKUST, Microsoft Research and Harvard.

**Method.** A frozen VLM answers a spatial question about a single image by
searching imagined camera moves. The action space is three primitives: move
forward, turn left, turn right. A camera-controlled video diffusion model
renders the view after each candidate trajectory. At each step the VLM gives
every candidate two scores, one for "keep exploring" and one for "helpful for
answering now". Candidates under a threshold are dropped, the top ones by
exploration score form the next beam, and the top ones by helpfulness go into an
evidence buffer. The VLM then answers from the original image plus the buffered
views and a text description of each trajectory.

**Settings.** Search depth 3, beam size 2, both thresholds 8, up to 3
repetitions of an action per expansion, 0.25 m per forward step, 9 degrees per
turn, maximum trajectory length 8. The released SWM launch script uses 2 search
steps, 2 beams and 20 diffusion steps instead.

**World models.** Two are used:

- Search World Model (SWM): their own, built on Wan2.2-TI2V-5B with camera
  conditioning following ReCamMaster, trained on Habitat/HM3D, RealEstate10K
  and DL3DV-10K on eight H100s for about three days.
- Stable Virtual Camera (SVC): Stability AI's public model.

**Results on SAT-Real (150 questions), average accuracy.**

| VLM | Baseline | + MindJourney (SWM) | + MindJourney (SVC) |
|---|---|---|---|
| GPT-4o | 60.3 | 70.6 | 69.3 |
| GPT-4.1 | 74.0 | 80.6 | not reported |
| InternVL3-14B | 59.3 | 66.6 | not reported |
| OpenAI o1 | 74.6 | 84.7 | 77.3 |

The paper states an average gain of 7.7 points on SAT-Real and 8.0 on a
500-question subset of SAT-Synthesized. Gains are not uniform per task: GPT-4o
with SWM drops on object movement from 85.0 to 56.5.

**Compute.** A40 GPUs for SWM, H100 80 GB for SVC, and H100 for InternVL3-14B
with SWM. The README recommends 80 GB of VRAM or more. Their timing table
(GPT-4o as VLM) gives 26.93 s per question at 2 steps and 20 diffusion steps
for 68.6% accuracy, and 16.27 s at 1 step for 64.7%.

**Stated limitations.** The pipeline assumes one reference view and does not
use extra views when a question supplies them. The world model is unaware of
the question and "can hallucinate views that are irrelevant—or even
contradictory". The failure analysis lists wrong forward distances, unintended
roll, unstable rotation, artefacts in unseen regions and scene
misinterpretation on out-of-domain images.

**Relevance.** This is the `imagine` baseline. The hallucination limitation the
authors name is the gap our verifier targets.

## 2. ViSA (arXiv 2512.05809v1, Dec 2025) — closest prior work on verification

"Probing the effectiveness of World Models for Spatial Reasoning through
Test-time Scaling", Jha, Mirza, Lin, Yang, Chandar (Mila and others). The repo
describes it as a workshop paper.

Read: full main text, results tables, part of the appendix.

**Findings about MindJourney.** Its helpfulness score gives little calibration:
selecting frames at random reduces the VLM's answer entropy as much or more.
MindJourney's selection has a left-turn bias (46–53% of moves).

**Method.** For each imagined frame a VLM generates question-conditioned
"micro-claims" about spatial relations; a verifier VLM judges each claim
against that same frame as ENTAILED, CONTRADICTED or INSUFFICIENT with a
confidence. The fraction entailed times mean confidence is an evidence-quality
score used to pick frames for the evidence buffer.

**Results.** InternVL3-14B with SVC on 2×A100 80 GB. On SAT-Real their baseline
is 41.33 and ViSA reaches 65.33–72.67 across settings at beam depth 1, against
63.33–67.33 for MindJourney's scorer and 63.33–66.00 for random selection. On
MMSI-Bench no verifier scales; accuracy stays around 27–36%.

**How ours differs.** ViSA's verifier is a VLM looking at the imagined frame
itself, so it scores how useful a frame is and cannot tell whether the frame is
faithful to the real scene. It selects evidence; it does not output answer
confidence or abstain. We check claims against geometry reconstructed from the
real observations, and use disagreement to set confidence. Our claim
extraction step is close to theirs and must cite it.

**Caution.** Their InternVL3-14B baseline on SAT-Real is 41.33; MindJourney
reports 59.3 for the same model and split. Baselines on this benchmark are
sensitive to prompt and parsing, so ours must be reported with full details.

## 3. Think3D (arXiv 2601.13029v5, last revised 9 Oct 2026)

Read: abstract, introduction, related work, method overview, results text.
Tables were not read cell by cell.

**Method.** A VLM agent calls a reconstruction tool on multi-view images to get
a point cloud and camera poses, then repeatedly picks an anchor camera, a
rotation and a global or ego view, and looks at the rendered point cloud. At
most 3 rounds. The reconstruction model is **Pi3**, not VGGT. A second part,
Think3D-RL, fine-tunes Qwen3-VL-4B with GRPO to explore well.

**Results as stated in the text.** On MindCube-1K, +7.67 points for GPT-4.1 and
+20.20 for Gemini 2.5 Pro. On VSI-Bench-Tiny, +2.96 and +6.45. For untrained
Qwen3-VL-4B-Instruct the tools **reduce** MindCube accuracy by 2.5 points; after
RL fine-tuning the tools add 3.58.

**Relevance.** This is the closest work to our `geometry` mode. It has no
generative imagination and no verification. Its small-model result is a warning
for us: a small VLM given free-form 3D tools got worse without training. Our
geometry half should use constrained queries, not open-ended exploration.

## 4. Thinking in Space / VSI-Bench (arXiv 2412.14171)

Read: abstract, benchmark description, main findings, parts of the appendix.

Over 5,000 question-answer pairs from 288 real indoor videos (150 ARKitScenes,
50 ScanNet++, 88 ScanNet validation scenes). Eight tasks of three types:
configurational, measurement estimation and spatiotemporal. Human evaluators
reach 79% average accuracy, 33 points above the best model. Zero-shot
chain-of-thought and tree-of-thoughts lower average performance by about 4%,
and self-consistency ends 1.1% below baseline. Prompting for a cognitive map
improves relative-distance questions.

Dataset on Hugging Face: `nyu-visionx/VSI-Bench`, Apache-2.0, not gated, about
5.7 GB of video zips. It ships `test.jsonl` plus `test_debiased.parquet` and
`test_pruned.parquet`; this repository uses `test.jsonl`. A
2026 paper, ReVSI (arXiv 2604.24300), argues the benchmark has validity
problems; only its title and a search snippet were read.

## 5. Other related work (abstracts only)

| Paper | What it does | Overlap with ours |
|---|---|---|
| AVIC, "When and How Much to Imagine" (2602.08236) | Decides whether current evidence is sufficient before calling the world model; RL variant. SAT, MMSI, R2R. | Controls when to imagine. Does not check imagined content against geometry. |
| Astra, "Thinking with Imagination" (2606.06476) | RL-trained VLM policy plus a view-consistency-tuned world simulator. | Improves the simulator's reliability by training. No test-time geometric check. |
| TwNV, "Thinking with Novel Views" (2605.10588) | Reasoner asks a "Painter" for a novel view, iterates. +1.3 to +3.9 points. | A search snippet mentions a quality verifier; the abstract does not. Read the full paper before the write-up. |
| Reasmory (2606.00963) | Reconstruction as explicit 3D memory with object instances and a validated DSL for queries. Gains of 6–18%. | Close to our geometry half (labelled 3D map, constrained queries). No imagination, no verification. |
| World2VLM (2604.26934), GeoWorld-VLM (2605.16713) | Distil world-model imagination or geometry into the VLM by training. | Training-time; we are test-time. |
| SpatialImaginer (2604.17385), WM-VLM (2609.34826) | Unified models that generate intermediate visual states. | Training-time; different setting. |

## 6. Novelty check

Searches run on 2026-10-12 (web search, standard and extended):

- world model + imagined views + hallucination + verification + 3D
  reconstruction + VLM spatial reasoning
- MindJourney follow-up + geometric verification + point cloud + VGGT +
  abstain/confidence
- generative novel view synthesis + hallucination + reconstructed point cloud +
  selective prediction
- imagination + 3D reconstruction + verify imagined views + VSI-Bench

**Result.** No paper was found that cross-checks claims derived from
generatively imagined views against geometry reconstructed from the real
observations and uses the disagreement to set confidence or abstain. That
combination still appears open.

**Limits of this check.** It used web search only, not Google Scholar or
Papers with Code directly, and most neighbours were read at abstract level.
The area is moving fast: five of the papers above appeared between February
and September 2026. The check should be repeated before any write-up.

**What we can and cannot claim.** Claim-level verification of imagined views
exists (ViSA). Reconstruction-based spatial reasoning with labelled 3D memory
exists (Think3D, Reasmory). Our claim is the cross-check between the two and
its use as a hallucination and confidence signal, evaluated with
confident-error and risk–coverage metrics that none of these papers report.
