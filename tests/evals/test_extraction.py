import pytest

from juddges.evals.extraction import ExtractionEvaluator


@pytest.fixture
def sample_schema():
    return {
        "date_field": {"type": "string"},
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
    assert not result["missing_keys"]
    assert not result["extra_keys"]
    assert result["field_results"]["date_field"]["match"] is True
    assert result["field_results"]["enum_field"]["match"] is True
    assert result["field_results"]["enum_field"]["hallucinated"] is False
    assert result["field_results"]["list_field"]["metrics"]["f1"] == 1.0
    assert result["field_results"]["string_field"]["rouge"]["rougeL"] == 1.0


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
    assert result["field_results"]["date_field"]["match"] is False
    assert result["field_results"]["enum_field"]["match"] is False
    # todo: fix the test to assert concrete f1 and rougeL value
    assert result["field_results"]["list_field"]["metrics"]["f1"] < 1.0
    assert result["field_results"]["string_field"]["rouge"]["rougeL"] < 1.0


def test_key_mismatches(evaluator):
    predicted = {"date_field": "2024-01-01", "extra": 1}
    gold = {"date_field": "2024-01-01", "missing": 2}
    result = evaluator.evaluate_record(predicted, gold)
    assert result["missing_keys"] == ["missing"]
    assert result["extra_keys"] == ["extra"]


def test_enum_hallucination(evaluator):
    predicted = {"enum_field": "D"}
    gold = {"enum_field": "A"}
    result = evaluator.evaluate_record(predicted, gold)
    assert result["field_results"]["enum_field"]["match"] is False
    assert result["field_results"]["enum_field"]["hallucinated"] is True


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

    results = evaluator.run(predictions)

    assert results["summary_metrics"]["total_records"] == 4
    assert results["summary_metrics"]["filtered_records (finish_reason=='stop')"] == 3
    assert results["summary_metrics"]["parsing_errors"] == 1
    assert results["summary_metrics"]["evaluated_records"] == 2

    field_metrics = results["summary_metrics"]["field_metrics"]
    assert field_metrics["date_field"]["accuracy"] == 0.5
    assert field_metrics["enum_field"]["accuracy"] == 0.5
    assert field_metrics["enum_field"]["hallucinations"] == 1
    assert field_metrics["enum_field"]["hallucination_rate"] == 0.5

    assert len(results["per_record_results"]) == 3
    assert results["per_record_results"][0]["index"] == 0
    assert results["per_record_results"][1]["index"] == 1
    assert results["per_record_results"][2]["index"] == 2
    assert "error" in results["per_record_results"][2]
