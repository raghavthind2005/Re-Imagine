import math

from mindseye.eval.hallucination import Obj, compare_view, grid_cell, rates


def test_grid_cell_edges():
    assert grid_cell(0, 0, 300, 300) == (0, 0)
    assert grid_cell(299.9, 299.9, 300, 300) == (2, 2)
    assert grid_cell(300, 150, 300, 300) == (2, 1)  # clamped at the border


def test_compare_view_classifies_labels_and_positions():
    real = [
        Obj("chair", 50, 50),
        Obj("table", 150, 150),
        Obj("lamp", 250, 50),
        Obj("door", 20, 250),
    ]
    imagined = [
        Obj("chair", 60, 40),
        Obj("table", 280, 280),
        Obj("sofa", 150, 150),
        Obj("door", 10, 240),
        Obj("door", 200, 240),
    ]
    c = compare_view(real, imagined, 300, 300)
    assert c.matched == ["chair", "door", "table"]
    assert c.invented == ["sofa"] and c.missing == ["lamp"]
    # door has two instances in the imagined view, so only chair and table are position-checked
    assert c.position_checked == 2 and c.position_agree == 1


def test_rates_pool_over_views():
    a = compare_view([Obj("chair", 1, 1)], [Obj("chair", 1, 1), Obj("tv", 5, 5)], 9, 9)
    b = compare_view([Obj("bed", 1, 1), Obj("lamp", 8, 8)], [Obj("stove", 4, 4)], 9, 9)
    r = rates([a, b])
    assert r["n_views"] == 2
    assert r["invented_rate"] == 2 / 3  # tv and stove invented, chair real
    assert r["missing_rate"] == 2 / 3  # bed and lamp missing, chair kept
    assert r["position_agreement"] == 1.0 and r["n_position_checked"] == 1
    empty = rates([compare_view([], [], 9, 9)])
    assert math.isnan(empty["invented_rate"]) and math.isnan(empty["position_agreement"])


def test_flag_metrics_counts_and_ratios():
    from mindseye.eval.hallucination import flag_metrics

    m = flag_metrics([True, True, False, False, True], [True, False, True, False, True])
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 1, 1, 1)
    assert m["precision"] == 2 / 3 and m["recall"] == 2 / 3


def test_flag_metrics_nothing_flagged_is_nan_precision():
    import math

    from mindseye.eval.hallucination import flag_metrics

    m = flag_metrics([True, False], [False, False])
    assert math.isnan(m["precision"]) and m["recall"] == 0.0


def test_cohen_kappa_perfect_and_chance():
    from mindseye.eval.hallucination import cohen_kappa

    assert cohen_kappa([True, False, True, False], [True, False, True, False]) == 1.0
    assert cohen_kappa([True, True, False, False], [True, False, True, False]) == 0.0
