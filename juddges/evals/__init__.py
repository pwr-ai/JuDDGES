"""Evaluation module for JuDDGES project."""

try:
    from .extraction import ExtractionEvaluator
    from .metrics import evaluate_date, evaluate_list_greedy, evaluate_number, evaluate_string_rouge
except ImportError:
    ExtractionEvaluator = None  # type: ignore[assignment, misc]
    evaluate_date = None  # type: ignore[assignment]
    evaluate_list_greedy = None  # type: ignore[assignment]
    evaluate_number = None  # type: ignore[assignment]
    evaluate_string_rouge = None  # type: ignore[assignment]

__all__ = [
    "ExtractionEvaluator",
    "evaluate_date",
    "evaluate_list_greedy",
    "evaluate_number",
    "evaluate_string_rouge",
]
