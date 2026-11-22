import numpy as np

from mindseye.imagine.beam_search import SearchConfig, beam_search, expansions
from mindseye.imagine.world_model import ImaginedView

START = np.eye(4)


class FakeWorldModel:
    name = "fake"

    def __init__(self):
        self.calls = []

    def render(self, start_c2w, poses):
        self.calls.append(len(poses))
        return [ImaginedView(np.zeros((2, 2, 3), np.uint8), p) for p in poses]


class ScriptedScorer:
    """Scores from a function of the candidate's action tuple."""

    def __init__(self, fn):
        self.fn, self.seen = fn, []

    def score(self, question, candidates):
        self.seen.append([c.actions for c in candidates])
        return [self.fn(c.actions) for c in candidates]


def test_root_expansion_gives_nine_candidates_from_three_calls():
    wm = FakeWorldModel()
    scorer = ScriptedScorer(lambda a: (0, 0))
    res = beam_search("q", START, wm, scorer, SearchConfig())
    assert wm.calls == [3, 3, 3]
    assert len(scorer.seen[0]) == 9  # MindJourney Fig. 3: k=3 gives 9 new observations
    assert res.n_world_model_calls == 3 and res.n_views_rendered == 9
    assert res.steps_run == 1 and res.stopped_early and res.evidence == []


def test_expansions_prune_reversals_and_respect_length_budget():
    cfg = SearchConfig()
    assert set(expansions(("turn_left",), cfg)) == {"move_forward", "turn_left"}
    assert set(expansions(("turn_right",), cfg)) == {"move_forward", "turn_right"}
    assert set(expansions(("move_forward",), cfg)) == {"move_forward", "turn_left", "turn_right"}
    seven = ("move_forward",) * 7
    assert [len(t) for t in expansions(seven, cfg)["move_forward"]] == [8]
    assert expansions(("move_forward",) * 8, cfg) == {}


def test_candidate_poses_follow_the_trajectory():
    wm = FakeWorldModel()

    class Capture:
        def score(self, question, candidates):
            self.c = {c.actions: c for c in candidates}
            return [(0, 0)] * len(candidates)

    cap = Capture()
    beam_search("q", START, wm, cap, SearchConfig())
    fwd3 = cap.c[("move_forward",) * 3]
    assert np.allclose(fwd3.view.c2w[:3, 3], [0, 0, 0.75])
    assert fwd3.description == "move forward 0.75 m"
    assert cap.c[("turn_left", "turn_left")].description == "turn left 18 degrees"


def test_beam_selection_thresholds_and_evidence():
    # Only "turn_left x2" and "move_forward x1" are worth exploring; forward moves are helpful.
    def fn(actions):
        explore = {("turn_left", "turn_left"): 10, ("move_forward",): 9, ("turn_right",): 7}
        helpful = 9 if actions[-1] == "move_forward" else 0
        return explore.get(actions, 0), helpful

    wm, scorer = FakeWorldModel(), ScriptedScorer(fn)
    res = beam_search("q", START, wm, scorer, SearchConfig(n_steps=2))
    # Step 2 expands exactly the two surviving nodes; ("turn_right",) was under threshold.
    prefixes = {a[:2] for a in scorer.seen[1] if a[0] == "turn_left"}
    assert prefixes == {("turn_left", "turn_left")}
    assert all(a[0] in ("turn_left", "move_forward") for a in scorer.seen[1])
    # From (L, L): no right turn allowed -> 2 actions; from (F,): 3 actions. 5 calls + 3 at root.
    assert res.n_world_model_calls == 8
    assert res.steps_run == 2
    # Evidence: top-2 helpful per step, all forward-ending and above threshold.
    assert len(res.evidence) == 4
    assert all(c.actions[-1] == "move_forward" and c.helpful_score == 9 for c in res.evidence)


def test_search_stops_when_no_candidate_passes_exploration_threshold():
    res = beam_search("q", START, FakeWorldModel(), ScriptedScorer(lambda a: (7.9, 10)))
    assert res.steps_run == 1 and res.stopped_early
    assert len(res.evidence) == 2  # helpful views are still kept from the step that ran


def test_search_respects_trajectory_length_cap():
    res = beam_search(
        "q", START, FakeWorldModel(), ScriptedScorer(lambda a: (10, 0)),
        SearchConfig(n_steps=10, max_trajectory_len=5),
    )  # fmt: skip
    assert res.stopped_early and res.steps_run < 10
