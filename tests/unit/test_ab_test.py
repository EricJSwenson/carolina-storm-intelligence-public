from storm_eval.evaluation.ab_test import paired_ab


def test_significant_when_separated():
    a = [0.0, 0.1, 0.0, 0.0, 0.1, 0.0]
    b = [1.0, 0.9, 1.0, 1.0, 0.9, 1.0]
    res = paired_ab("reward", "A", a, "B", b)
    assert res.diff > 0.5 and res.significant


def test_not_significant_when_identical():
    a = [0.5, 0.6, 0.7]
    res = paired_ab("reward", "A", a, "B", list(a))
    assert not res.significant
