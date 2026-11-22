import numpy as np
import pytest

from mindseye.eval.selective import (
    aurc,
    confident_error_rate,
    coverage,
    paired_bootstrap_diff,
    risk_coverage_curve,
    selective_risk,
)


def test_confident_error_rate():
    correct = [True, False, True, False, True]
    high = [True, True, True, False, False]
    assert confident_error_rate(correct, high) == pytest.approx(1 / 3)
    assert np.isnan(confident_error_rate(correct, [False] * 5))
    with pytest.raises(ValueError):
        confident_error_rate(correct, high[:3])


def test_coverage_and_selective_risk():
    correct = [True, False, True, False]
    answered = [True, True, True, False]
    assert coverage(answered) == 0.75
    assert selective_risk(correct, answered) == pytest.approx(1 / 3)
    assert np.isnan(selective_risk(correct, [False] * 4))


def test_risk_coverage_curve_known_values():
    correct = [True, True, False, True, False]
    conf = [0.9, 0.8, 0.7, 0.6, 0.5]
    cov, risk = risk_coverage_curve(correct, conf)
    assert cov.tolist() == pytest.approx([0.2, 0.4, 0.6, 0.8, 1.0])
    assert risk.tolist() == pytest.approx([0, 0, 1 / 3, 1 / 4, 2 / 5])
    assert aurc(correct, conf) == pytest.approx(np.mean([0, 0, 1 / 3, 1 / 4, 2 / 5]))


def test_good_confidence_beats_bad_confidence_and_final_risk_is_error_rate():
    rng = np.random.default_rng(0)
    correct = rng.random(500) < 0.6
    good = correct + rng.normal(0, 0.3, 500)  # confidence correlated with being right
    bad = -good
    assert aurc(correct, good) < aurc(correct, rng.random(500)) < aurc(correct, bad)
    for conf in (good, bad):
        assert risk_coverage_curve(correct, conf)[1][-1] == pytest.approx(1 - correct.mean())


def test_curve_with_ties_does_not_depend_on_input_order():
    correct = np.array([True, False, True, False, True, True])
    conf = np.array([1.0, 1.0, 1.0, 0.5, 0.5, 0.5])
    base = aurc(correct, conf)
    perm = np.random.default_rng(3).permutation(6)
    assert aurc(correct[perm], conf[perm]) == pytest.approx(base)
    # Uninformative (constant) confidence gives the plain error rate at every coverage.
    assert aurc(correct, np.ones(6)) == pytest.approx(1 - correct.mean())


def test_paired_bootstrap_diff():
    rng = np.random.default_rng(0)
    base = rng.random(400) < 0.5
    better = base | (rng.random(400) < 0.2)
    res = paired_bootstrap_diff(better, base)
    assert res["n"] == 400 and res["diff"] > 0
    assert res["ci_low"] > 0  # a real paired gain is detected
    same = paired_bootstrap_diff(base, base)
    assert same["diff"] == same["ci_low"] == same["ci_high"] == 0.0
    clustered = paired_bootstrap_diff(better, base, clusters=np.arange(400) // 20)
    assert clustered["diff"] == pytest.approx(res["diff"])
