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
        ("2024-01-01", "2024-01-01", 1),
        ("Jan 1, 2024", "2024-01-01", 1),
        ("2024-01-01", "2024-01-02", 0),
        (None, "2024-01-01", 0),
        ("2024-01-01", None, 0),
        (None, None, 1),
        ("invalid-date", "2024-01-01", 0),
    ],
)
def test_evaluate_date(predicted, gold, expected):
    assert evaluate_date(predicted, gold) == {"match": expected}


@pytest.mark.parametrize(
    "predicted, gold, expected",
    [
        (10, 10, 1),
        (10.5, 10.5, 1),
        (10, 10.0, 1),
        ("10", 10, 1),
        (10, 11, 0),
        (None, 10, 0),
        (10, None, 0),
        (None, None, 1),
        ("ten", 10, 0),
    ],
)
def test_evaluate_number(predicted, gold, expected):
    assert evaluate_number(predicted, gold) == {"match": expected}


def test_evaluate_string_rouge():
    predicted = "this is a test"
    gold = "this is a test"
    scores = evaluate_string_rouge(predicted, gold)
    assert scores is not None
    assert scores["rougeL"] == pytest.approx(1.0)

    predicted = "a different string"
    scores = evaluate_string_rouge(predicted, gold)
    assert scores is not None
    assert scores["rougeL"] < 1.0

    assert evaluate_string_rouge(None, gold) == {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    assert evaluate_string_rouge(predicted, None) == {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}


@pytest.mark.parametrize(
    "predicted, gold, choices, expected_match, expected_predicted_in_choices",
    [
        ("A", "A", ["A", "B", "C", None], 1, 1),
        ("A", "B", ["A", "B", "C", None], 0, 1),
        ("D", "A", ["A", "B", "C", None], 0, 0),
        (None, "A", ["A", "B", "C", None], 0, 1),
        ("A", None, ["A", "B", "C", None], 0, 1),
        (None, None, ["A", "B", "C", None], 1, 1),
        ("D", None, ["A", "B", "C", None], 0, 0),
    ],
)
def test_evaluate_enum(predicted, gold, choices, expected_match, expected_predicted_in_choices):
    result = evaluate_enum(predicted, gold, choices)
    assert result["match"] == expected_match
    assert result["predicted_in_choices"] == expected_predicted_in_choices


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
