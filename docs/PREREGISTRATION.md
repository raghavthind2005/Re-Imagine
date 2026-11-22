# Pre-registrations

Four analyses were written down and committed, with their success criteria, before the data
they concern were opened. This repository was published with a fresh history, so the texts are
copied here verbatim from the private development repository, where each is a dated commit
(hash and time given per section; available on request). Square brackets mark the only edits:
wording that referred to internal project stages. Commit hashes inside the texts and in the run
records (`results/raw/*/env.txt`) also refer to that repository.

Outcomes are in the README's results table and in the generated reports named per section.

## 1. Dev head-to-head: map check against the ViSA-style self-check (automatic ground truth)

Original commit `53a2345`, 2026-10-14 00:10:17 -0400. Report: `results/hallucination_headtohead.md`.

**Question.** Does geometry with coverage-aware trust detect world-model hallucinations better than ViSA-style VLM self-verification?

**Data.** The scaled study already collected: 20 dev scenes (`data/splits/hallucination_scenes.json`), 160 held-out views, 60 control views, 371 object claims (kernel `reimagine-p2-hallucination-v2`, commit `fa2c2e6`, records in `results/hallucination_v2/`). No new rendering and no GPU time.

**Already seen, disclosed.** Methods (a), (b) and (d) below were computed on these same records and reported on 2026-10-13 (`results/hallucination_v2.md`), under all three ground truths. Methods (c) and (e) have not been computed. The primary ground truth below is chosen after seeing (b) against (d), so the result must also hold in direction under the other two truths (robustness condition).

**Unit.** One claim = one object label that SAM 3 finds in an imagined held-out view. A claim is *invented* if the ground truth says the object is not in the real view from that camera.

**Ground truth.** Primary: SAM 3 and the VLM (Qwen3-VL-8B on the real frame) agree; claims where they disagree are dropped. Secondary: SAM 3 alone; VLM alone.

**Methods (flag = "this object is invented").**
- (a) Geometry verdicts only: flag if contradicted.
- (b) Geometry + no-coverage rule: flag if contradicted or no coverage.
- (c) Geometry where covered, VLM where not: flag if contradicted; keep if supported; if no coverage, flag if the ViSA-style judge (VLM on the imagined frame alone) says the object is absent. Secondary variant (c′): the VLM judge that also sees the observed frames is used in uncovered regions.
- (d) ViSA-style self-check: flag if the VLM, looking only at the imagined frame, says the object is absent.
- (e) Trivial baseline: flag every claim in a view whose map coverage is below a threshold *t*. Primary: *t* chosen to maximise F1 on these same records over {0.05, 0.10, …, 0.95} (favours the baseline). Also reported at *t* = 0.5.

**Metrics.** Precision, recall, F1 for invented objects; false-flag rate on real objects (flagged real / all real); 95% intervals from 2,000 scene-level bootstrap resamples; SAM 3 noise floor from the 60 control views.

**"Ours wins"** if (b) or (c) has a higher F1 than both (d) and (e), with the paired scene-level bootstrap 95% interval of each F1 difference excluding 0, under the primary ground truth; and the same F1 differences are positive under both secondary truths. Two candidate methods are tested, which is noted when reporting. Otherwise ours does not win.

## 2. Downstream check: answer accuracy with filtered imagined views (not run)

Original commit `9c70feb`, 2026-10-14 11:03:39 -0400. Report: none (not run); `scripts/downstream_table.py` would write `results/downstream.md`.

Written before any of these runs. No tuning.

- **Questions:** the same 100 VSI-Bench dev questions and 97 SAT perspective questions as [the `imagine` versus `vlm_only` comparison].
- **Imagined views:** reused from the "always top 2" runs (`p2_imagine_vsi_sub200_sparse8_top2_qwen8b_v1-n100`, `p2_imagine_sat_persp_circular_top2_qwen8b_v1`), the setting where imagination hurt. Nothing is re-rendered.
- **Modes:** `vlm_only`; `imagine` (both evidence views); `imagine_visa` (keep a view only if it shows at least one object and the ViSA-style self-check, Qwen on that view alone, says each object is present, p("no") < 0.5); `imagine_ours` (keep a view only if it shows at least one object and the labelled map of the observed frames confirms each object — method (b), same thresholds as the hallucination study).
- **Metrics:** accuracy (VSI-Bench mean score; SAT strict and order-averaged), paired against `vlm_only` and `imagine`, scene-level bootstrap (VSI-Bench) or question-level (SAT); confident-error rate on multiple-choice questions = share wrong among answers with confidence ≥ 0.9, where confidence is Qwen's probability of the chosen option letter (SAT: the lower of the two orders); AURC; compute per mode (imagined views used, VLM calls, seconds).
- **Success for ours:** not worse than `vlm_only` (paired difference ≥ −1 point with the 95% interval reaching above 0) and a lower confident-error rate than `imagine` (point estimate). Reported whichever way it goes; with about 100 questions per benchmark the intervals will be wide.

## 3. Test-set confirmation run (automatic ground truth; criterion not met)

Original commit `748c05d`, 2026-10-14 11:24:27 -0400. Report: `results/hallucination_headtohead_test.md`, `results/hallucination_curves_test.md`, `results/hallucination_test_report.md`.

**Data.** 20 VSI-Bench *test* scenes, fixed in `data/splits/hallucination_test_scenes.json` (seed 20261015, stratified by dataset: 10 ARKitScenes, 6 ScanNet, 4 ScanNet++). No test questions are used. Run once.

**Protocol (identical to dev, kernel job `kaggle/jobs/p3_hallucination_test.json`).** 8 observed frames, 8 held-out frames, 3 control cameras per scene; VGGT on observed + held-out frames for poses, map from observed frames only; SVC v1.1 at 384×512, 21 frames per pass, 20 steps, cfg 2.0, fp16, seed 0; SAM 3 with the 40-label vocabulary, score ≥ 0.5, area ≥ 0.2% of the image; Qwen3-VL-8B prompts `presence-v1`, greedy, plus yes/no probabilities.

**Final method (b), coverage-aware trust.** An imagined object is flagged as invented unless the labelled map of the observed frames shows that label, unoccluded, on at least 0.2% of the view's pixels from that camera. Continuous score: minus that visible share.

**Baselines.** (d) ViSA-style self-check: flagged if Qwen, seeing only the imagined frame, answers "no" (greedy; equivalently p("no") > 0.5); continuous score p("no"). (e) flag all claims in views whose map coverage is below *t*, with *t* fitted on the test records to maximise F1 (favours the baseline), also reported at the dev value *t* = 0.25. (c′) geometry where covered, Qwen with the observed frames where not (greedy), continuous score as in `scripts/hallucination_curves.py`.

**Ground truths.** Primary: SAM 3 and Qwen agree on the real frame (disagreements dropped). Secondary: SAM 3 alone; Qwen alone. Human labels exist for dev only.

**Win criterion (same as dev).** (b) has a higher F1 than both (d) and (e, fitted *t*), with the paired scene-level bootstrap 95% interval of each difference excluding 0 under the primary truth, and positive differences under both secondary truths.

**Reported regardless.** Precision, recall, F1, false-flag rate; AUROC, average precision, recall at 4/8/16% false flags, best F1 over thresholds; (b) − (c′); SAM 3 noise floor from control views; hallucination rate by coverage bin. Scripts: `hallucination_report.py`, `hallucination_headtohead.py`, `hallucination_curves.py`, unchanged except for input paths.

## 4. Human-labelled genuine-hallucination comparison on test (criterion met)

Original commit `04dc420`, 2026-10-14 18:05:43 -0400. Report: `results/hallucination_human_test.md`.

**Question.** On world-model hallucinations confirmed by a human (the object is drawn in the imagined view but absent from the real frame), does the map check (b) catch more than the ViSA-style self-check (d)?

**Claims.** The 389 object claims of the test run (`results/hallucination_test/`, kernel `reimagine-p3-hallucination-test`, commit `748c05d`). Method verdicts are already fixed in those records; nothing is re-run.

**Sampling ([design chosen by the author]).** Seed 20261016, drawn by `scripts/make_test_label_sheet.py` and saved to `data/labels/test_claim_sample.json` before any labelling.
- Every claim SAM 3 marks invented (census, 124 claims; weight 1).
- 40 claims drawn uniformly at random from the 265 SAM 3 marks real (weight 265/40 = 6.625).
- 30 claims drawn uniformly at random from the dev sheet, mixed in unmarked, to measure labelling consistency (excluded from all test results).
- Order shuffled; no model output or stratum shown. Same two questions as the dev sheet.
- Second labeller: 50 claims drawn uniformly at random from the 164 test claims, as a separate sheet.

**Definitions.** Genuine hallucination: the labeller says the object is visible in the imagined view and not visible in the real frame. Real object: visible in both. Detector error: not visible in the imagined view. Claims with "unsure" or blank on either question are excluded.

**Primary criterion.** Among genuine hallucinations, (b) catches more than (d): exact two-sided McNemar test on the paired catches, p < 0.05, with more claims caught only by (b) than only by (d). Recall of each method reported with exact (Clopper–Pearson) 95% intervals. If fewer than 10 genuine hallucinations are found, the result is reported as inconclusive.

**Reported alongside.** False-flag rate on real objects for (b), (c′), (d), weighted by the sampling weights (unweighted counts shown too), with 95% scene-level bootstrap intervals; the same comparison for (b) against (c′); share of each SAM 3 category that is a detector error, a genuine hallucination or a real object; labeller consistency on the 30 repeats (agreement and Cohen's kappa against the dev answers, per question); agreement between the two labellers on their shared 50 claims, when available. The primary labeller's answers are the ground truth.
