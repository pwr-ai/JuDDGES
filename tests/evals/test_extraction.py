import pytest

from juddges.evals.extraction import ExtractionEvaluator
from juddges.llm_as_judge.base import ItemEvalResult
from juddges.llm_as_judge.data_model import PredictionLoader


@pytest.fixture
def sample_schema():
    return {
        "date_field": {"type": "date"},
        "enum_field": {"type": "enum", "choices": ["A", "B", "C"], "required": False},
        "list_field": {"type": "list", "items": {"type": "string"}},
        "string_field": {"type": "string"},
    }


@pytest.fixture
def evaluator(sample_schema):
    return ExtractionEvaluator(sample_schema)


def test_get_field_type(evaluator):
    assert evaluator.field_types["date_field"] == "date"
    assert evaluator.field_types["enum_field"] == "enum"
    assert evaluator.field_types["list_field"] == "list"
    assert evaluator.field_types["string_field"] == "string"


def test_evaluate_record_perfect_match(evaluator):
    predicted = {
        "date_field": "2024-01-01",
        "enum_field": "A",
        "list_field": ["x", "y"],
        "string_field": "hello world",
    }
    gold = {
        "date_field": "2024-01-01",
        "enum_field": "A",
        "list_field": ["x", "y"],
        "string_field": "hello world",
    }
    result = evaluator.evaluate_record(predicted, gold)
    assert result == {
        "date_field": {"match": 1},
        "enum_field": {"match": 1, "predicted_in_choices": 1},
        "list_field": {
            "true_positives": 2,
            "false_positives": 0,
            "false_negatives": 0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
        },
        "string_field": {
            "rouge1": 1.0,
            "rouge2": 1.0,
            "rougeL": 1.0,
        },
    }


def test_evaluate_record_mismatch(evaluator):
    predicted = {
        "date_field": "2024-01-02",
        "enum_field": "B",
        "list_field": ["x", "z"],
        "string_field": "hello there",
    }
    gold = {
        "date_field": "2024-01-01",
        "enum_field": "A",
        "list_field": ["x", "y"],
        "string_field": "hello world",
    }
    result = evaluator.evaluate_record(predicted, gold)
    assert result == {
        "date_field": {"match": 0},
        "enum_field": {"match": 0, "predicted_in_choices": 1},
        "list_field": {
            "true_positives": 1,
            "false_positives": 1,
            "false_negatives": 1,
            "precision": 0.5,
            "recall": 0.5,
            "f1": 0.5,
        },
        "string_field": {
            "rouge1": 0.5,
            "rouge2": 0.0,
            "rougeL": 0.5,
        },
    }


def test_enum_hallucination(evaluator):
    predicted = {"enum_field": "D"}
    gold = {"enum_field": "A"}
    result = evaluator.evaluate_record(predicted, gold)
    assert result == {
        "date_field": {"match": 0},
        "enum_field": {"match": 0, "predicted_in_choices": 0},
        "list_field": {
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        },
        "string_field": {
            "rouge1": 0.0,
            "rouge2": 0.0,
            "rougeL": 0.0,
        },
    }


def test_run_and_aggregate(evaluator):
    predictions = [
        {
            "answer": '```json\n{"date_field": "2024-01-01", "enum_field": "A"}\n```',
            "gold": '{"date_field": "2024-01-01", "enum_field": "A"}',
            "finish_reason": "stop",
            "original_index": 0,
        },
        {
            "answer": '```json\n{"date_field": "2024-01-02", "enum_field": "D"}\n```',
            "gold": '{"date_field": "2024-01-01", "enum_field": "A"}',
            "finish_reason": "stop",
            "original_index": 1,
        },
        {"answer": "invalid json", "gold": "{}", "finish_reason": "stop", "original_index": 2},
        {"answer": "{}", "gold": "{}", "finish_reason": "length", "original_index": 3},
    ]
    parsed_preds = PredictionLoader.load_predictions(
        schema=evaluator.schema,
        preds=predictions,
        verbose=False,
    )

    eval_results = evaluator.run(parsed_preds)
    assert eval_results.results == [
        ItemEvalResult(
            status="success",
            error=None,
            result={
                "date_field": {"match": 1},
                "enum_field": {"match": 1, "predicted_in_choices": 1},
                "list_field": {
                    "true_positives": 0,
                    "false_positives": 0,
                    "false_negatives": 0,
                    "precision": 0,
                    "recall": 0,
                    "f1": 0,
                },
                "string_field": {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0},
            },
            missing_keys=["list_field", "string_field"],
            extra_keys=[],
        ),
        ItemEvalResult(
            status="success",
            error=None,
            result={
                "date_field": {"match": 0},
                "enum_field": {"match": 0, "predicted_in_choices": 0},
                "list_field": {
                    "true_positives": 0,
                    "false_positives": 0,
                    "false_negatives": 0,
                    "precision": 0,
                    "recall": 0,
                    "f1": 0,
                },
                "string_field": {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0},
            },
            missing_keys=["list_field", "string_field"],
            extra_keys=[],
        ),
        ItemEvalResult(
            status="parsing_error",
            error="Expecting value: line 1 column 1 (char 0)",
            result={
                "date_field": {"match": 0},
                "enum_field": {"match": 0, "predicted_in_choices": 0},
                "list_field": {
                    "true_positives": 0,
                    "false_positives": 0,
                    "false_negatives": 0,
                    "precision": 0,
                    "recall": 0,
                    "f1": 0,
                },
                "string_field": {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0},
            },
            missing_keys=[],
            extra_keys=[],
        ),
        ItemEvalResult(
            status="success",
            error=None,
            result={
                "date_field": {"match": 0},
                "enum_field": {"match": 0, "predicted_in_choices": 0},
                "list_field": {
                    "true_positives": 0,
                    "false_positives": 0,
                    "false_negatives": 0,
                    "precision": 0,
                    "recall": 0,
                    "f1": 0,
                },
                "string_field": {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0},
            },
            missing_keys=["date_field", "enum_field", "list_field", "string_field"],
            extra_keys=[],
        ),
    ]
