from pytest import approx

from juddges.llm_as_judge.base import EvalResults, ItemEvalResult


def test_eval_results_basic_functionality():
    schema = {"key1": {}, "key2": {}}
    results = [
        ItemEvalResult.from_success({"key1": {"score": 1}, "key2": {"score": 0}}),
        ItemEvalResult.from_success({"key1": {"score": 0.5}, "key2": {"score": 0.5}}),
    ]
    eval_results = EvalResults(ie_schema=schema, results=results)

    model_dump = eval_results.model_dump()
    assert "stats" in model_dump
    assert "aggregated_scores" in model_dump
    assert "all_results" in model_dump

    aggregated_scores = eval_results.get_aggregated_scores()
    assert aggregated_scores["key1"]["mean_score"] == 0.75
    assert aggregated_scores["key1"]["std_score"] == approx(0.353, abs=1e-3)
    assert aggregated_scores["key2"]["mean_score"] == 0.25
    assert aggregated_scores["key2"]["std_score"] == approx(0.353, abs=1e-3)

    statistics = eval_results.get_statistics()
    assert statistics["total_docs"] == 2
    assert statistics["num_success_evaluations"] == 2
    assert statistics["num_parsing_errors"] == 0
    assert statistics["num_judge_errors"] == 0
    assert statistics["missing_keys"] == {}
    assert statistics["extra_keys"] == {}
    assert statistics["avg_missing_keys_when_success"] == 0
    assert statistics["avg_extra_keys_when_success"] == 0


def test_eval_results_missing_keys():
    schema = {"key1": {}, "key2": {}}
    results = [
        ItemEvalResult.from_success({"key1": {"score": 1}}, missing_keys=["key2"]),  # Missing key2
    ]
    eval_results = EvalResults(ie_schema=schema, results=results)

    statistics = eval_results.get_statistics()
    assert statistics["total_docs"] == 1
    assert statistics["num_success_evaluations"] == 1
    assert statistics["num_parsing_errors"] == 0
    assert statistics["num_judge_errors"] == 0
    assert statistics["missing_keys"]["key2"] == 1
    assert statistics["extra_keys"] == {}
    assert statistics["avg_missing_keys_when_success"] == 1.0
    assert statistics["avg_extra_keys_when_success"] == 0.0


def test_eval_results_extra_keys():
    schema = {"key1": {}}
    results = [
        ItemEvalResult.from_success(
            {"key1": {"score": 1}, "key2": {"score": 0}},
            extra_keys=["key2"],
        ),  # Extra key2
    ]
    eval_results = EvalResults(ie_schema=schema, results=results)

    statistics = eval_results.get_statistics()
    assert statistics["total_docs"] == 1
    assert statistics["num_success_evaluations"] == 1
    assert statistics["num_parsing_errors"] == 0
    assert statistics["num_judge_errors"] == 0
    assert statistics["missing_keys"] == {}
    assert statistics["extra_keys"]["key2"] == 1
    assert statistics["avg_missing_keys_when_success"] == 0.0
    assert statistics["avg_extra_keys_when_success"] == 1.0
