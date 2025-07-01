import pytest

from juddges.evals.metrics import (
    evaluate_date,
    evaluate_enum,
    evaluate_list_greedy,
    evaluate_number,
    evaluate_string_rouge,
)


@pytest.mark.parametrize(
    "predicted, gold, expected",
    [
        ("2024-01-01", "2024-01-01", True),
        ("Jan 1, 2024", "2024-01-01", True),
        ("2024-01-01", "2024-01-02", False),
        (None, "2024-01-01", False),
        ("2024-01-01", None, False),
        (None, None, True),
        ("invalid-date", "2024-01-01", False),
    ],
)
def test_evaluate_date(predicted, gold, expected):
    assert evaluate_date(predicted, gold) == expected


@pytest.mark.parametrize(
    "predicted, gold, expected",
    [
        (10, 10, True),
        (10.5, 10.5, True),
        (10, 10.0, True),
        ("10", 10, True),
        (10, 11, False),
        (None, 10, False),
        (10, None, False),
        (None, None, True),
        ("ten", 10, False),
    ],
)
def test_evaluate_number(predicted, gold, expected):
    assert evaluate_number(predicted, gold) == expected


def test_evaluate_string_rouge():
    predicted = "this is a test"
    gold = "this is a test"
    scores = evaluate_string_rouge(predicted, gold)
    assert scores is not None
    assert scores["rougeL"] == 1.0

    predicted = "a different string"
    scores = evaluate_string_rouge(predicted, gold)
    assert scores is not None
    assert scores["rougeL"] < 1.0

    assert evaluate_string_rouge(None, gold) is None
    assert evaluate_string_rouge(predicted, None) is None


@pytest.mark.parametrize(
    "predicted, gold, choices, expected_match, expected_hallucinated",
    [
        ("A", "A", ["A", "B", "C", None], True, False),
        ("A", "B", ["A", "B", "C", None], False, False),
        ("D", "A", ["A", "B", "C", None], False, True),
        (None, "A", ["A", "B", "C", None], False, False),
        ("A", None, ["A", "B", "C", None], False, False),
        (None, None, ["A", "B", "C", None], True, False),
        ("D", None, ["A", "B", "C", None], False, True),
    ],
)
def test_evaluate_enum(predicted, gold, choices, expected_match, expected_hallucinated):
    result = evaluate_enum(predicted, gold, choices)
    assert result["match"] == expected_match
    assert result["hallucinated"] == expected_hallucinated


@pytest.mark.parametrize(
    "predicted, gold, expected_tp, expected_fp, expected_fn",
    [
        (["a", "b", "c"], ["a", "b", "c"], 3, 0, 0),
        (["a", "b"], ["a", "b", "c"], 2, 0, 1),
        (["a", "b", "d"], ["a", "b", "c"], 2, 1, 1),
        ([], ["a", "b"], 0, 0, 2),
        (["a", "b"], [], 0, 2, 0),
        (None, ["a"], 0, 0, 1),
        (["a"], None, 0, 1, 0),
        (None, None, 0, 0, 0),
        (["a", "a"], ["a"], 1, 1, 0),
        (["a"], ["a", "a"], 1, 0, 1),
    ],
)
def test_evaluate_list_greedy(predicted, gold, expected_tp, expected_fp, expected_fn):
    result = evaluate_list_greedy(predicted, gold)
    assert result["true_positives"] == expected_tp
    assert result["false_positives"] == expected_fp
    assert result["false_negatives"] == expected_fn
