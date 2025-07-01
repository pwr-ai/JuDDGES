from collections import Counter
from typing import Any, Optional

from dateutil import parser as date_parser
from torchmetrics.text import ROUGEScore


def evaluate_date(
    predicted: Optional[str],
    gold: Optional[str],
) -> bool:
    """
    Parses dates and checks for an exact match.

    Args:
        predicted: The predicted date string.
        gold: The ground truth date string.

    Returns:
        True if dates match, False otherwise.
    """
    if predicted is None and gold is None:
        return True
    if predicted is None or gold is None:
        return False

    try:
        predicted_date = date_parser.parse(predicted)
        gold_date = date_parser.parse(gold)
        return predicted_date == gold_date
    except (ValueError, TypeError):
        return False


def evaluate_number(
    predicted: Any,
    gold: Any,
) -> bool:
    """
    Compares two numbers for an exact match.

    Args:
        predicted: The predicted number.
        gold: The ground truth number.

    Returns:
        True if numbers are equal, False otherwise.
    """
    if predicted is None and gold is None:
        return True
    if predicted is None or gold is None:
        return False
    try:
        return float(predicted) == float(gold)
    except (ValueError, TypeError):
        return False


def evaluate_string_rouge(
    predicted: Optional[str],
    gold: Optional[str],
) -> Optional[dict[str, float]]:
    """
    Calculates ROUGE scores for two strings using TorchMetrics.

    Args:
        predicted: The predicted string.
        gold: The ground truth string.

    Returns:
        A dictionary with ROUGE scores, or None if inputs are invalid.
    """
    if not predicted or not gold:
        return None

    rouge = ROUGEScore()
    scores = rouge([predicted], [gold])

    return {
        "rouge1": scores["rouge1_fmeasure"].item(),
        "rouge2": scores["rouge2_fmeasure"].item(),
        "rougeL": scores["rougeL_fmeasure"].item(),
    }


def evaluate_enum(
    predicted: Optional[str],
    gold: Optional[str],
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
    predicted_in_choices = predicted in choices
    gold_in_choices = gold in choices
    hallucinated = not predicted_in_choices

    return {
        "match": predicted == gold,
        "hallucinated": hallucinated,
        "predicted_in_choices": predicted_in_choices,
        "gold_in_choices": gold_in_choices,
    }


def evaluate_list_greedy(predicted: Optional[list], gold: Optional[list]) -> dict[str, Any]:
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
