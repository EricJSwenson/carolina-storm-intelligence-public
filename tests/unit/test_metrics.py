from storm_eval.evaluation.metrics import groundedness, answer_relevance, cost_usd


def test_groundedness_supported_vs_unsupported():
    ctx = ["Hurricane Irene made landfall in North Carolina as a Category 1 hurricane."]
    supported = groundedness("Irene hit North Carolina as a Category 1 hurricane.", ctx)
    unsupported = groundedness("The stock market rallied on Tuesday afternoon.", ctx)
    assert supported > unsupported


def test_answer_relevance_range():
    r = answer_relevance("What category was the storm?", "It was a Category 1 hurricane.")
    assert 0.0 <= r <= 1.0


def test_cost_zero_for_mock():
    assert cost_usd("mock-grounded", 1000, 1000) == 0.0
    assert cost_usd("openai:gpt-4o-mini", 1000, 1000) > 0.0
