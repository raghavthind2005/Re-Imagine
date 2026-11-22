import numpy as np
import pytest

from mindseye.agent import imagine
from mindseye.data.schema import Example
from mindseye.imagine.svc_model import SVCCacheOnly, SVCSettings, render_key
from mindseye.imagine.svc_world_model import SVCWorldModel
from mindseye.imagine.world_model import ImaginedView
from mindseye.utils.cache import DiskCache, Pending
from mindseye.vlm.base import CacheOnlyVLM, MockVLM, VLMBackend, VLMResponse
from mindseye.vlm.prompts import imagine as prompts
from mindseye.vlm.prompts import vqa


class FakeRunner:
    """Returns flat images whose value encodes the call number."""

    def __init__(self, settings):
        self.settings, self.calls = settings, []

    def render(self, images, c2ws, Ks, target_c2ws, target_Ks):
        self.calls.append(len(target_c2ws))
        s = self.settings
        return [np.full((s.height, s.width, 3), len(self.calls), np.uint8) for _ in target_c2ws]


class ScriptedVLM(VLMBackend):
    model_id = "scripted"

    def __init__(self, score_reply):
        super().__init__(None)
        self.score_reply, self.prompts = score_reply, []

    def _generate(self, images, prompt, *, max_tokens, temperature, seed):
        self.prompts.append((len(images), prompt))
        if "best starting viewpoint" in prompt:
            return VLMResponse("2")
        if "exploration:" in prompt:
            return VLMResponse(self.score_reply)
        return VLMResponse("A")


def example():
    return Example(
        id="x-1",
        benchmark="vsibench",
        split="dev",
        question="Which is closer?",
        choices=["chair", "table"],
        answer="chair",
        media=["v.mp4"],
        meta={"question_type": "object_rel_distance", "scene": "s"},
    )


def setup(n=3):
    images = [np.full((40, 60, 3), i, np.uint8) for i in range(n)]
    c2w = np.stack([np.eye(4)] * n)
    c2w[:, 0, 3] = np.arange(n)
    K = np.stack([np.array([[50.0, 0, 30], [0, 50, 20], [0, 0, 1]])] * n)
    return images, imagine.SceneGeometry(c2w, K, np.full(n, 2.0))


CFG = {"forward_depth_fraction": 0.125, "n_steps": 1, "max_evidence": 2}
GEN = {"max_tokens": 16, "temperature": 0.0, "seed": 0}


def test_parse_scores_and_frame():
    assert prompts.parse_scores("exploration=7, helpfulness=9") == (7.0, 9.0)
    assert prompts.parse_scores("Exploration: 3\nHelpfulness: 10/10") == (3.0, 10.0)
    assert prompts.parse_scores("looks fine") is None
    assert prompts.parse_frame("Frame 3", 8) == 2
    assert prompts.parse_frame("12", 8) is None


def test_svc_world_model_batches_a_step_into_one_pass():
    s = SVCSettings(height=32, width=48, frames_per_pass=21)
    images, geo = setup(8)
    wm = SVCWorldModel(FakeRunner(s), np.stack(images), geo.c2w, geo.K)
    paths = [[np.eye(4)] * 3, [np.eye(4)] * 3, [np.eye(4)] * 3]
    out = wm.render_batch(np.eye(4), paths)
    assert [len(v) for v in out] == [3, 3, 3] and wm.runner.calls == [9]
    out = wm.render_batch(np.eye(4), paths * 2)  # 18 targets, 13 fit beside 8 inputs
    assert wm.runner.calls == [9, 13, 5] and sum(len(v) for v in out) == 18
    assert isinstance(out[0][0], ImaginedView) and out[0][0].image.shape == (32, 48, 3)


def test_imagine_uses_helpful_views_as_evidence():
    images, geo = setup()
    s = SVCSettings(height=32, width=48)
    wm = SVCWorldModel(FakeRunner(s), np.stack(images), geo.c2w, geo.K)
    vlm = ScriptedVLM("exploration=2, helpfulness=9")
    rec = imagine.answer(example(), images, {}, geo, wm, vlm, CFG, GEN, is_video=True)
    info = rec["imagine"]
    assert info["start_frame"] == 1 and info["n_views_rendered"] == 9
    assert info["forward_units"] == 0.25 and len(info["evidence"]) == 2
    n_images, final_prompt = vlm.prompts[-1]
    assert n_images == 3 + 2 and "imagined views" in final_prompt
    assert rec["score"] == 1.0 and rec["mode"] == "imagine"


def test_imagine_without_evidence_asks_the_vlm_only_prompt():
    images, geo = setup()
    s = SVCSettings(height=32, width=48)
    wm = SVCWorldModel(FakeRunner(s), np.stack(images), geo.c2w, geo.K)
    vlm = ScriptedVLM("exploration=1, helpfulness=1")
    rec = imagine.answer(example(), images, {}, geo, wm, vlm, CFG, GEN, is_video=True)
    assert rec["imagine"]["evidence"] == []
    assert vlm.prompts[-1] == (3, vqa.build_prompt(example(), is_video=True))


def test_stages_raise_pending_for_the_other_model(tmp_path):
    images, geo = setup()
    s = SVCSettings(height=32, width=48)
    cache = DiskCache(tmp_path)
    # SVC not rendered yet: the VLM stage stops at the first render.
    wm = SVCWorldModel(SVCCacheOnly(s, cache), np.stack(images), geo.c2w, geo.K)
    with pytest.raises(Pending):
        imagine.answer(example(), images, {}, geo, wm, MockVLM("1"), CFG, GEN, is_video=True)
    # VLM not loaded: the SVC stage stops at the first uncached question.
    wm = SVCWorldModel(FakeRunner(s), np.stack(images), geo.c2w, geo.K)
    off = CacheOnlyVLM("mock", {"reply": "1"}, cache)
    with pytest.raises(Pending):
        imagine.answer(example(), images, {}, geo, wm, off, CFG, GEN, is_video=True)


def test_render_key_changes_with_pose_and_settings():
    im = [np.zeros((4, 4, 3), np.uint8)]
    eye, K = np.eye(4)[None], np.eye(3)[None]
    moved = eye.copy()
    moved[0, 2, 3] = 0.1
    a = render_key(SVCSettings(), im, eye, K, eye, K)
    assert a == render_key(SVCSettings(), im, eye, K, eye, K)
    assert a != render_key(SVCSettings(), im, eye, K, moved, K)
    assert a != render_key(SVCSettings(num_steps=30), im, eye, K, eye, K)
