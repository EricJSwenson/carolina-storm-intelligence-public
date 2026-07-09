from storm_eval.evaluation.ground_truth import (
    extract_landfall_category, check_landfall_category, hallucination_flag,
)


def test_extract_category_numeric_and_word():
    assert extract_landfall_category("made landfall as a Category 1 hurricane") == 1
    assert extract_landfall_category("a Category Three storm") == 3
    assert extract_landfall_category("no category mentioned here") is None


def test_check_contradiction():
    assert check_landfall_category("Category 3", 1).contradicted is True
    assert check_landfall_category("Category 1", 1).contradicted is False
    assert check_landfall_category("no number", 1).checkable is True


def test_hallucination_flag():
    assert hallucination_flag("Category 3 hurricane", {"landfall_category": 1}) is True
    assert hallucination_flag("Category 1 hurricane", {"landfall_category": 1}) is False
    assert hallucination_flag("anything", {}) is None
