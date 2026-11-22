import json

import numpy as np
import pytest

from mindseye.agent import vlm_only
from mindseye.data.frames import sparse_view_split, uniform_indices
from mindseye.data.sat import choice_order, row_to_example
from mindseye.data.schema import Example
from mindseye.data.splits import load_split, save_split, split_groups, stratified_sample
from mindseye.data.vsibench import row_to_example as vsi_row
from mindseye.eval.metrics import (
    MRA_THRESHOLDS,
    bootstrap_ci,
    mcnemar_exact,
    mean_relative_accuracy,
    vsibench_overall,
)
from mindseye.eval.parse import official_first_token, official_number, parse_letter, parse_number
from mindseye.utils.cache import DiskCache
from mindseye.vlm.base import MockVLM
from mindseye.vlm.prompts import vqa

# --- splits -------------------------------------------------------------------


def test_stratified_sample_is_proportional_exact_and_deterministic():
    keys = list(range(1000))
    strata = ["a"] * 600 + ["b"] * 300 + ["c"] * 100
    picked = stratified_sample(keys, strata, 50, seed=1)
    assert len(picked) == len(set(picked)) == 50
    counts = [sum(strata[k] == s for k in picked) for s in "abc"]
    assert counts == [30, 15, 5]
    assert picked == stratified_sample(keys, strata, 50, seed=1)
    assert picked != stratified_sample(keys, strata, 50, seed=2)


def test_stratified_sample_rounding_still_sums_to_n():
    strata = ["a"] * 10 + ["b"] * 10 + ["c"] * 10
    assert len(stratified_sample(list(range(30)), strata, 10, seed=0)) == 10


def test_split_groups_keeps_each_group_on_one_side():
    groups = [f"s{i // 5}" for i in range(200)]  # 40 scenes, 5 questions each
    strata = ["x" if i < 100 else "y" for i in range(200)]
    dev, test = split_groups(groups, strata, 0.25, seed=3)
    assert len(dev) == 10 and len(test) == 30
    assert not set(dev) & set(test)
    assert set(dev) | set(test) == set(groups)
    assert sum(int(g[1:]) < 20 for g in dev) == 5  # stratified: 5 scenes from each stratum


def test_split_groups_rejects_group_in_two_strata():
    with pytest.raises(ValueError):
        split_groups(["s1", "s1"], ["x", "y"], 0.5, seed=0)


def test_saved_split_cannot_be_overwritten(tmp_path):
    path = tmp_path / "split.json"
    save_split(path, {"dev": [1, 2]}, {"seed": 0})
    assert load_split(path) == {"dev": [1, 2]}
    with pytest.raises(FileExistsError):
        save_split(path, {"dev": [3]}, {"seed": 0})


def test_committed_vsibench_split_has_no_scene_overlap():
    split = load_split("data/splits/vsibench.json")
    assert not set(split["dev_scenes"]) & set(split["test_scenes"])
    assert len(split["dev_scenes"]) + len(split["test_scenes"]) == 288


# --- loaders ------------------------------------------------------------------


def test_choice_order_is_a_deterministic_permutation():
    orders = [choice_order(f"sat-val-{i}", 2, seed=0) for i in range(400)]
    assert all(sorted(o) == [0, 1] for o in orders)
    assert orders == [choice_order(f"sat-val-{i}", 2, seed=0) for i in range(400)]
    first = sum(o == [0, 1] for o in orders)
    assert 150 < first < 250  # the correct answer is no longer always first


def test_sat_row_to_example(tmp_path):
    row = {
        "image_bytes": [b"png-a", b"png-b"],
        "question": "Which way?",
        "answers": ["left", "right"],
        "question_type": "action_conseq",
        "correct_answer": "left",
    }
    ex = row_to_example(row, 7, "test", "test", tmp_path, seed=0)
    assert ex.id == "sat-test-7" and ex.answer == "left"
    assert sorted(ex.choices) == ["left", "right"]
    assert ex.meta["question_type"] == "action_consequence"
    assert [open(m, "rb").read() for m in ex.media] == [b"png-a", b"png-b"]


def test_vsibench_rows(tmp_path):
    mca = vsi_row(
        {
            "id": 5,
            "dataset": "scannet",
            "scene_name": "scene0011_00",
            "question_type": "object_rel_direction_easy",
            "question": "q",
            "ground_truth": "B",
            "options": ["A. left", "B. right"],
        },
        "dev",
        tmp_path,
    )
    assert mca.choices == ["left", "right"] and mca.answer == "right"
    assert vqa.gold_letter(mca) == "B"
    assert mca.meta["scene"] == "scannet/scene0011_00"
    assert mca.media == [str(tmp_path / "scannet/scene0011_00.mp4")]
    na = vsi_row(
        {
            "id": 6,
            "dataset": "arkitscenes",
            "scene_name": "41069025",
            "question_type": "object_counting",
            "question": "How many?",
            "ground_truth": "4",
            "options": None,
        },
        "dev",
        tmp_path,
    )
    assert na.choices is None and na.answer == "4" and na.meta["answer_format"] == "na"


# --- frames -------------------------------------------------------------------


def test_uniform_indices():
    assert uniform_indices(100, 4) == [12, 37, 62, 87]
    assert uniform_indices(3, 8) == [0, 1, 2]


@pytest.mark.parametrize("n,k_obs", [(1000, 4), (1000, 8), (1000, 16), (20, 8), (9, 8), (8, 8)])
def test_sparse_view_split_never_overlaps(n, k_obs):
    observed, heldout = sparse_view_split(n, k_obs, 8)
    assert not set(observed) & set(heldout)
    assert all(0 <= i < n for i in observed + heldout)
    assert (observed, heldout) == sparse_view_split(n, k_obs, 8)


def test_heldout_frames_are_the_same_for_every_observed_count():
    splits = {k: sparse_view_split(5045, k, 8) for k in (4, 8, 16)}
    assert splits[4][1] == splits[8][1] == splits[16][1]
    assert len(splits[8][1]) == 8
    assert [len(splits[k][0]) for k in (4, 8, 16)] == [4, 8, 16]
    assert set(splits[4][0]) < set(splits[8][0])  # 4-frame view is a subset of the 8-frame view
    assert splits[8][0] == uniform_indices(5045, 8)  # same frames as the 8-frame baseline


# --- prompts and parsing ------------------------------------------------------


def _mca():
    return Example("x", "vsibench", "dev", "Where?", ["left", "right", "back"], "back", ["v.mp4"],
                   {"question_type": "t"})  # fmt: skip


def test_prompt_matches_official_vsibench_wording():
    assert vqa.build_prompt(_mca(), is_video=True) == (
        "These are frames of a video.\nWhere?\nOptions:\nA. left\nB. right\nC. back\n"
        "Answer with the option's letter from the given choices directly."
    )
    na = Example("y", "vsibench", "dev", "How many?", None, "4", ["v.mp4"])
    assert vqa.build_prompt(na, is_video=True) == (
        "These are frames of a video.\nHow many?\n"
        "Please answer the question using a single word or phrase."
    )


@pytest.mark.parametrize(
    "text,expected",
    [
        ("B", "B"),
        ("B.", "B"),
        ("(C)", "C"),
        ("Answer: A", "A"),
        ("The answer is B. right", "B"),
        ("D", None),  # only three choices
        ("back", None),
        ("", None),
    ],
)
def test_parse_letter(text, expected):
    assert parse_letter(text, 3) == expected


@pytest.mark.parametrize(
    "text,expected",
    [("4", 4.0), ("2.5 meters", 2.5), ("about 1,200 cm", 1200.0), ("none", None), (".5", 0.5)],
)
def test_parse_number(text, expected):
    assert parse_number(text) == expected


def test_official_parsers_match_reference_behaviour():
    assert official_first_token("B. right") == "B"
    assert official_number("2.5 meters") == 2.5
    assert official_number("about 3") is None


# --- metrics ------------------------------------------------------------------


def test_mra_thresholds_match_reference_formula():
    start, end, interval = 0.5, 0.95, 0.05
    reference = np.linspace(start, end, int((end - start) / interval + 2))
    assert np.allclose(MRA_THRESHOLDS, reference)
    assert len(MRA_THRESHOLDS) == 10


def test_mean_relative_accuracy_known_values():
    assert mean_relative_accuracy(10.0, 10.0) == 1.0
    assert mean_relative_accuracy(None, 10.0) == 0.0
    assert mean_relative_accuracy(100.0, 10.0) == 0.0
    # 17% relative error passes thresholds 0.5 .. 0.8 (1 - c >= 0.17): 7 of 10
    assert mean_relative_accuracy(11.7, 10.0) == pytest.approx(0.7)
    assert mean_relative_accuracy(float("nan"), 10.0) == 0.0


def test_vsibench_overall_merges_relative_direction():
    types = [
        "object_rel_direction_easy",
        "object_rel_direction_medium",
        "object_rel_direction_hard",
        "object_counting",
    ]
    assert vsibench_overall([1.0, 0.0, 0.5, 0.0], types) == pytest.approx(0.25)
    with pytest.raises(ValueError):
        vsibench_overall([1.0, 0.0], types[:1] + types[3:])


def test_bootstrap_ci_covers_the_mean_and_shrinks_with_n():
    rng = np.random.default_rng(0)
    small = rng.random(50) < 0.6
    large = rng.random(5000) < 0.6
    lo_s, hi_s = bootstrap_ci(lambda i: float(small[i].mean()), len(small))
    lo_l, hi_l = bootstrap_ci(lambda i: float(large[i].mean()), len(large))
    assert lo_s <= small.mean() <= hi_s
    assert lo_l <= 0.6 <= hi_l
    assert (hi_l - lo_l) < (hi_s - lo_s) / 3


def test_cluster_bootstrap_is_wider_when_clusters_are_correlated():
    scores = np.repeat(np.array([1.0, 0.0] * 20), 25)  # 40 scenes, all-right or all-wrong
    clusters = np.repeat(np.arange(40), 25)
    naive = bootstrap_ci(lambda i: float(scores[i].mean()), len(scores))
    clustered = bootstrap_ci(lambda i: float(scores[i].mean()), len(scores), clusters=clusters)
    assert (clustered[1] - clustered[0]) > 3 * (naive[1] - naive[0])


def test_mcnemar_exact():
    same = mcnemar_exact([True, False, True], [True, False, True])
    assert same["p_value"] == 1.0
    a = [True] * 10 + [False] * 0 + [True] * 5
    b = [False] * 10 + [True] * 0 + [True] * 5
    res = mcnemar_exact(a, b)
    assert (res["only_a"], res["only_b"]) == (10, 0)
    assert res["p_value"] == pytest.approx(2 / 2**10)


# --- agent --------------------------------------------------------------------


def test_vlm_only_scores_and_caches(tmp_path):
    img = tmp_path / "a.png"
    img.write_bytes(b"fake")
    ex = Example("sat-val-1", "sat", "dev", "Which?", ["left", "right"], "right", [str(img)],
                 {"question_type": "goal_aim"})  # fmt: skip
    backend = MockVLM("B", cache=DiskCache(tmp_path / "cache"))
    gen = {"max_tokens": 16, "temperature": 0.0, "seed": 0}
    first = vlm_only.answer(ex, backend, {"mode": "images"}, gen)
    second = vlm_only.answer(ex, backend, {"mode": "images"}, gen)
    assert first["score"] == 1.0 and first["pred"] == "B" and first["gold"] == "B"
    assert not first["cached"] and second["cached"]
    assert backend.calls == 1
    assert first["cache_key"] == second["cache_key"]
    json.dumps(first)  # the record must be JSON-serialisable

    wrong = vlm_only.answer(ex, MockVLM("A"), {"mode": "images"}, gen)
    assert wrong["score"] == 0.0


def test_vlm_only_numeric_scoring():
    ex = Example("vsi-1", "vsibench", "dev", "How far?", None, "10", ["v.mp4"],
                 {"question_type": "object_abs_distance", "scene": "s"})  # fmt: skip
    rec = vlm_only.score(ex, "about 11.7 meters")
    assert rec["score"] == pytest.approx(0.7)
    assert rec["score_official"] == 0.0  # the reference scorer reads only the first token


def test_qwen_resize_respects_max_pixels():
    from mindseye.vlm.qwen_local import QwenLocal

    backend = QwenLocal.__new__(QwenLocal)  # no model load
    backend.max_pixels = 200704
    big = np.zeros((480, 640, 3), dtype=np.uint8)
    out = backend._to_pil(big)
    assert out.width * out.height <= 200704
    assert abs(out.width / out.height - 640 / 480) < 0.01
    small = np.zeros((100, 120, 3), dtype=np.uint8)
    assert backend._to_pil(small).size == (120, 100)
    backend.max_pixels = None
    assert backend._to_pil(big).size == (640, 480)


def test_circular_evaluation_requires_both_orders(tmp_path):
    img = tmp_path / "a.png"
    img.write_bytes(b"fake")
    ex = Example("sat-val-2", "sat", "dev", "Which?", ["left", "right"], "right", [str(img)],
                 {"question_type": "perspective"})  # fmt: skip
    gen = {"max_tokens": 16, "temperature": 0.0, "seed": 0}
    view = {"mode": "images", "circular": True}
    # A model that always says "B" is right in one order only.
    rec = vlm_only.answer(ex, MockVLM("B"), view, gen)
    assert rec["score_first_order"] == 1.0 and rec["score_flipped_order"] == 0.0
    assert rec["score"] == 0.0 and rec["score_avg"] == 0.5
    assert rec["gold"] == "B" and rec["gold_flipped"] == "A"

    class Oracle(MockVLM):
        def _generate(self, images, prompt, **kw):
            self.reply = "A" if "A. right" in prompt else "B"
            return super()._generate(images, prompt, **kw)

    rec = vlm_only.answer(ex, Oracle(), view, gen)
    assert rec["score"] == 1.0 and rec["score_avg"] == 1.0
    # Without the flag the record has no order-averaged score.
    assert "score_avg" not in vlm_only.answer(ex, MockVLM("B"), {"mode": "images"}, gen)
