import math

from mindseye.eval.ranking import auroc, average_precision, operating_points, recall_at_false_flag


def test_auroc_perfect_random_and_ties():
    assert auroc([True, True, False, False], [0.9, 0.8, 0.2, 0.1]) == 1.0
    assert auroc([True, True, False, False], [0.1, 0.2, 0.8, 0.9]) == 0.0
    assert auroc([True, False], [0.5, 0.5]) == 0.5
    assert math.isnan(auroc([True, True], [0.1, 0.2]))


def test_average_precision():
    assert average_precision([True, False, True], [0.9, 0.5, 0.1]) == 1 / 2 * 1 + 1 / 2 * 2 / 3
    assert average_precision([True, True, False], [1.0, 1.0, 0.0]) == 1.0


def test_ties_are_flagged_together():
    pts = operating_points([True, False, True, False], [1.0, 1.0, 0.5, 0.0])
    assert pts[0]["recall"] == 0.5 and pts[0]["false_flag"] == 0.5


def test_recall_at_false_flag():
    labels = [True, True, False, True, False, False]
    scores = [0.9, 0.8, 0.7, 0.6, 0.2, 0.1]
    assert recall_at_false_flag(labels, scores, 0.0) == 2 / 3
    assert recall_at_false_flag(labels, scores, 0.34) == 1.0
