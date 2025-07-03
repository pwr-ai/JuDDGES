from collections import Counter
from typing import Any

from dateutil import parser as date_parser
from torchmetrics.text import ROUGEScore


def evaluate_date(
    predicted: str | None,
    gold: str | None,
) -> int:
    """
    Parses dates and checks for an exact match.

    Args:
        predicted: The predicted date string.
        gold: The ground truth date string.

    Returns:
        True if dates match, False otherwise.
    """
    if predicted == gold:
        return 1

    try:
        predicted_date = date_parser.parse(predicted)
        gold_date = date_parser.parse(gold)
        return int(predicted_date == gold_date)
    except (ValueError, TypeError):
        return 0


def evaluate_number(
    predicted: Any,
    gold: Any,
) -> int:
    """
    Compares two numbers for an exact match.

    Args:
        predicted: The predicted number.
        gold: The ground truth number.

    Returns:
        True if numbers are equal, False otherwise.
    """
    return int(predicted == gold)


def evaluate_string_rouge(
    predicted: str | None,
    gold: str | None,
) -> dict[str, float] | None:
    """
    Calculates ROUGE scores for two strings using TorchMetrics.

    Args:
        predicted: The predicted string.
        gold: The ground truth string.

    Returns:
        A dictionary with ROUGE scores, or None if inputs are invalid.
    """
    if predicted == gold:
        return {"rouge1": 1, "rouge2": 1, "rougeL": 1}
    elif predicted is None or gold is None:
        return {"rouge1": 0, "rouge2": 0, "rougeL": 0}

    rouge = ROUGEScore()
    scores = rouge([predicted], [gold])

    return {
        "rouge1": scores["rouge1_fmeasure"].item(),
        "rouge2": scores["rouge2_fmeasure"].item(),
        "rougeL": scores["rougeL_fmeasure"].item(),
    }


def evaluate_enum(
    predicted: str | None,
    gold: str | None,
    choices: list[str],
) -> dict[str, Any]:
    """
    Evaluates enum classification with hallucination detection.

    Args:
        predicted: The predicted enum value.
        gold: The ground truth enum value.
        choices: List of valid enum choices.

    Returns:
        Dictionary with classification metrics and hallucination info.
    """
    return {
        "match": int(predicted == gold),
        "predicted_in_choices": int(predicted in choices),
    }


def evaluate_list_greedy(predicted: list | None, gold: list | None) -> dict[str, Any]:
    """
    Evaluates list matching using a greedy approach.

    todo: hungarian matching should be used instead of greedy matching in the final version

    Args:
        predicted: The predicted list.
        gold: The ground truth list.

    Returns:
        A dictionary with counts for true positives, false positives,
        false negatives, and precision, recall, F1-score.
    """
    if predicted is None:
        predicted = []
    if gold is None:
        gold = []

    gold_counts = Counter(gold)
    pred_counts = Counter(predicted)

    true_positives = 0
    for item, count in pred_counts.items():
        true_positives += min(count, gold_counts.get(item, 0))

    false_positives = len(predicted) - true_positives
    false_negatives = len(gold) - true_positives

    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0
    )
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
