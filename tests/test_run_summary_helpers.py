import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from juddges.utils.json_schema import ie_schema_to_json_schema
from juddges.utils.run_record import (
    file_sha256,
    run_record_matches,
    write_json_atomic,
    write_run_record,
)

IE_SCHEMA = {
    "decision": {"type": "enum", "choices": ["Tak", "Nie"], "required": False},
    "court_type": {"type": "enum", "choices": ["A", "B"], "required": True},
    "judgment_date": {
        "type": "string",
        "description": "Data wyroku w formacie YYYY-MM-DD",
        "required": False,
    },
    "summary": {"type": "string", "description": "free text", "required": False},
    "evidence": {"type": "list", "items": {"type": "enum", "choices": ["x", "y"]}},
}


def test_ie_schema_to_json_schema():
    schema = ie_schema_to_json_schema(IE_SCHEMA)

    assert schema["required"] == list(IE_SCHEMA)  # all keys, schema order
    assert schema["additionalProperties"] is False
    props = schema["properties"]
    assert props["decision"] == {"enum": ["Tak", "Nie", None]}
    assert props["court_type"] == {"enum": ["A", "B"]}
    assert props["judgment_date"] == {"anyOf": [{"type": "string"}, {"type": "null"}]}
    # absent `required` means nullable (docstring), only `required: true` forbids null
    assert props["evidence"] == {
        "anyOf": [{"type": "array", "items": {"enum": ["x", "y"]}}, {"type": "null"}]
    }
    json.dumps(schema)


def test_ie_schema_to_json_schema_swiss_franc_config():
    schema_dict = yaml.safe_load(Path("configs/ie_schema/swiss_franc_loans.yaml").read_text())
    schema = ie_schema_to_json_schema(schema_dict)
    assert list(schema["properties"]) == list(schema_dict)


def test_ie_schema_to_json_schema_rejects_unknown_type():
    with pytest.raises(ValueError):
        ie_schema_to_json_schema({"x": {"type": "table"}})


def test_write_run_record_uses_null_for_unknown(tmp_path):
    split = tmp_path / "train.json"
    split.write_text("[]")

    path = write_run_record(
        output_dir=tmp_path,
        config={"a": Path("x")},
        seed=7,
        model_name="not-cached/model-that-does-not-exist",
        split_files={"train": split, "missing": tmp_path / "missing.json"},
        start_time=0.0,
    )

    record = json.loads(path.read_text())
    assert path.name == "run_record.json"
    assert record["seed"] == 7 and record["config"] == {"a": "x"}
    assert record["split_files_sha256"]["train"] == file_sha256(split)
    assert record["split_files_sha256"]["missing"] is None
    assert record["model"] == {"name": "not-cached/model-that-does-not-exist", "revision": None}
    assert set(record["library_versions"]) == {
        "torch",
        "vllm",
        "transformers",
        "peft",
        "trl",
        "bitsandbytes",
        "accelerate",
        "xgrammar",
    }
    assert record["adapter"] is None
    assert record["wall_clock_seconds"] > 0


def test_write_run_record_stores_adapter_and_extra(tmp_path):
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"weights")

    path = write_run_record(
        output_dir=tmp_path,
        config={},
        seed=1,
        model_name="not-cached/model-that-does-not-exist",
        split_files={},
        start_time=0.0,
        adapter_path=adapter_dir,
        extra={"guided_decoding_backend": "xgrammar"},
    )

    record = json.loads(path.read_text())
    assert record["adapter"] == {
        "path": str(adapter_dir.resolve()),
        "sha256": file_sha256(adapter_dir / "adapter_model.safetensors"),
    }
    assert record["guided_decoding_backend"] == "xgrammar"


def test_run_record_matches_requires_same_seed_and_config(tmp_path):
    assert not run_record_matches(tmp_path, 7, guided_json=True)  # no run record

    config = {"guided_json": True, "max_model_len": 8192, "max_docs": None}
    write_run_record(tmp_path, config, 7, "m/x", {}, start_time=0.0)

    assert run_record_matches(tmp_path, 7, guided_json=True, max_model_len=8192, max_docs=None)
    assert not run_record_matches(tmp_path, 8, guided_json=True)
    assert not run_record_matches(tmp_path, 7, guided_json=False)
    assert not run_record_matches(tmp_path, 7, max_model_len=4096)
    assert not run_record_matches(tmp_path, 7, max_docs=1)


def test_write_json_atomic_leaves_no_temp_file(tmp_path):
    path = tmp_path / "predictions.json"

    write_json_atomic([{"a": "ą"}], path, ensure_ascii=False)

    assert json.loads(path.read_text()) == [{"a": "ą"}]
    assert [f.name for f in tmp_path.iterdir()] == ["predictions.json"]


def _summarize_module():
    pytest.importorskip("typer")
    pytest.importorskip("pandas")
    pytest.importorskip("langchain_core")
    from scripts.sft import summarize_metrics

    return summarize_metrics


def test_bootstrap_ci_is_deterministic_and_brackets_mean():
    module = _summarize_module()
    values = np.array([1.0] * 30 + [0.0] * 70)

    ci = module.bootstrap_ci(values)

    assert ci == module.bootstrap_ci(values)  # fixed seed
    assert ci[0] < 0.3 < ci[1]
    assert module.bootstrap_ci([1.0, 1.0, 1.0]) == (1.0, 1.0)


def test_summarize_run_counts_parse_errors_and_exact_matches(tmp_path):
    module = _summarize_module()
    ie_schema = {k: IE_SCHEMA[k] for k in ["decision", "judgment_date", "summary"]}
    (tmp_path / "config.yaml").write_text(yaml.dump({"ie_schema": ie_schema}))
    gold = json.dumps({"decision": "Tak", "judgment_date": "2020-01-02", "summary": "s"})
    predictions = [
        {"answer": gold, "gold": gold, "finish_reason": "stop"},
        {
            "answer": json.dumps(
                {"decision": "Nie", "judgment_date": "02.01.2020", "summary": "s"}
            ),
            "gold": gold,
            "finish_reason": "stop",
        },
        {"answer": "I cannot answer", "gold": gold, "finish_reason": "length"},
    ]
    (tmp_path / "predictions.json").write_text(json.dumps(predictions))
    judge = {
        "parse_errors_scored_as_zero": True,
        "all_results": {
            "0": {"decision": 1, "summary": 1},
            "1": {"decision": 0, "summary": 1},
            "2": {"decision": 0, "summary": 0},
        },
    }
    (tmp_path / "llm_as_judge_some-judge.json").write_text(json.dumps(judge))

    result = module.summarize_run(tmp_path)

    assert result["num_docs"] == 3
    assert result["parsing_errors"] == 1
    assert result["finish_reason_length"] == 1
    assert result["per_field_exact_match"] == {"judgment_date": 2 / 3, "decision": 1 / 3}
    assert result["date_match_iso"] == 2 / 3  # "02.01.2020" is 2020-01-02 with dayfirst
    assert result["exact_match_categorical"] == 1 / 3
    assert result["judge_average"] == pytest.approx(0.5)
    assert result["judge_file"] == "llm_as_judge_some-judge.json"


def test_summarize_run_scores_missing_keys_as_zero(tmp_path):
    module = _summarize_module()
    ie_schema = {k: IE_SCHEMA[k] for k in ["decision", "judgment_date"]}
    (tmp_path / "config.yaml").write_text(yaml.dump({"ie_schema": ie_schema}))
    gold = json.dumps({"decision": None, "judgment_date": None})
    # truncated answer: langchain parses it partially, so it is not a parsing error
    truncated = {"answer": '{"decision": "Tak", "judgm', "gold": gold, "finish_reason": "length"}
    (tmp_path / "predictions.json").write_text(json.dumps([truncated]))

    result = module.summarize_run(tmp_path)

    assert result["parsing_errors"] == 0
    assert result["finish_reason_length"] == 1
    assert result["per_field_exact_match"] == {"judgment_date": 0.0, "decision": 0.0}


def test_normalize_iso_date_rejects_partial_dates():
    module = _summarize_module()

    assert module.normalize_iso_date("2020-01-02") == "2020-01-02"  # ISO is not day/month swapped
    assert module.normalize_iso_date("02.01.2020") == "2020-01-02"
    assert module.normalize_iso_date("2020-01") is None
    assert module.normalize_iso_date("January 2020") is None
    assert module.normalize_iso_date("2020") is None
    assert module.normalize_iso_date("2020-13-45") is None
    assert module.normalize_iso_date(None) is None
    assert module.normalize_iso_date(20200102) is None


def _write_run(tmp_path, ie_schema, golds, answers):
    (tmp_path / "config.yaml").write_text(yaml.dump({"ie_schema": ie_schema}))
    predictions = [
        {"answer": a, "gold": json.dumps(g), "finish_reason": "stop"}
        for a, g in zip(answers, golds)
    ]
    (tmp_path / "predictions.json").write_text(json.dumps(predictions))


def test_summarize_run_reports_nonnull_gold_variant_and_null_match(tmp_path):
    module = _summarize_module()
    ie_schema = {k: IE_SCHEMA[k] for k in ["decision", "judgment_date"]}
    golds = [
        {"decision": "Tak", "judgment_date": "2020-01-02"},
        {"decision": None, "judgment_date": None},
        {"decision": None, "judgment_date": None},
    ]
    answers = [
        json.dumps({"decision": "Tak", "judgment_date": "2020-01"}),  # partial date: no match
        json.dumps({"decision": None, "judgment_date": None}),  # null-null: match
        "not json",  # parse error: 0 even for null gold
    ]
    _write_run(tmp_path, ie_schema, golds, answers)

    result = module.summarize_run(tmp_path)

    assert result["per_field_exact_match"] == {"judgment_date": 1 / 3, "decision": 2 / 3}
    assert result["date_match_iso"] == 1 / 3
    assert result["date_match_iso_nonnull_gold"] == 0.0  # only document 0 has a gold date
    assert result["exact_match_categorical_nonnull_gold"] == 1.0


def test_summarize_run_counts_non_dict_answer_as_parse_error(tmp_path):
    module = _summarize_module()
    ie_schema = {k: IE_SCHEMA[k] for k in ["decision"]}
    _write_run(tmp_path, ie_schema, [{"decision": "Tak"}] * 2, ["[1, 2]", '"Tak"'])

    result = module.summarize_run(tmp_path)

    assert result["parsing_errors"] == 2
    assert result["per_field_exact_match"] == {"decision": 0.0}


def _judge_file(tmp_path, name, num_docs, flag=True):
    data = {"all_results": {str(i): {"decision": 1} for i in range(num_docs)}}
    if flag:
        data["parse_errors_scored_as_zero"] = True
    (tmp_path / f"llm_as_judge_{name}.json").write_text(json.dumps(data))


def test_summarize_run_selects_judge_by_name_and_checks_doc_count(tmp_path):
    module = _summarize_module()
    ie_schema = {k: IE_SCHEMA[k] for k in ["decision"]}
    _write_run(tmp_path, ie_schema, [{"decision": "Tak"}] * 2, ['{"decision": "Tak"}'] * 2)
    _judge_file(tmp_path, "a", 2)
    _judge_file(tmp_path, "b", 1)

    with pytest.raises(ValueError, match="Several judge files"):
        module.summarize_run(tmp_path)
    assert module.summarize_run(tmp_path, "a")["judge_file"] == "llm_as_judge_a.json"
    with pytest.raises(ValueError, match="scores 1 documents, expected 2"):
        module.summarize_run(tmp_path, "b")
    with pytest.raises(FileNotFoundError):
        module.summarize_run(tmp_path, "missing")


def test_judge_file_without_parse_error_flag_is_rejected(tmp_path):
    module = _summarize_module()
    _judge_file(tmp_path, "old", 1, flag=False)

    with pytest.raises(ValueError, match="parse_errors_scored_as_zero"):
        module.load_judge_file(tmp_path / "llm_as_judge_old.json")


def test_summarize_reads_scores_ngram_json_and_fails_when_empty(tmp_path):
    module = _summarize_module()
    run = tmp_path / "llm" / "seed_1"
    run.mkdir(parents=True)
    scores = {
        "aggregated_scores": {
            "summary": {"rouge1": {"mean_score": 0.5, "standard_error": 0.1}},
            "decision": {"f1": {"mean_score": 1.0, "standard_error": 0.0}},
        }
    }
    (run / "scores_ngram.json").write_text(json.dumps(scores))

    assert module.parse_ngram_metrics(run / "scores_ngram.json", llm="llm") == [
        {"llm": "llm", "summary/rouge1": 0.5, "decision/f1": 1.0}
    ]
    module.summarize(tmp_path, "scores_ngram.json", "ngram")
    assert (tmp_path / "metrics_ngram_summary.md").exists()

    with pytest.raises(RuntimeError, match="No judge results"):
        module.summarize(tmp_path, "llm_as_judge_x.json", "judge")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(RuntimeError, match="No runs"):
        module.summarize_extraction(empty)


def test_summarize_seeds_reports_mean_and_std_across_seeds():
    module = _summarize_module()
    base = {"date_match_iso_nonnull_gold": None, "exact_match_categorical_nonnull_gold": None}
    runs = [
        {
            "llm": "m",
            "run_dir": "m/p/s/seed_1",
            **base,
            "date_match_iso": 0.2,
            "exact_match_categorical": 0.5,
            "judge_average": None,
        },
        {
            "llm": "m",
            "run_dir": "m/p/s/seed_2",
            **base,
            "date_match_iso": 0.4,
            "exact_match_categorical": 0.5,
            "judge_average": None,
        },
        {
            "llm": "n",
            "run_dir": "n/p/s/seed_1",
            **base,
            "date_match_iso": 0.9,
            "exact_match_categorical": 0.1,
            "judge_average": None,
        },
    ]

    rows = {row["config_dir"]: row for row in module.summarize_seeds(runs)}

    assert rows["m/p/s"]["num_seeds"] == 2
    assert rows["m/p/s"]["date_match_iso"]["mean"] == pytest.approx(0.3)
    assert rows["m/p/s"]["date_match_iso"]["std"] == pytest.approx(0.1414213, rel=1e-4)
    assert rows["n/p/s"]["date_match_iso"] == {"mean": 0.9, "std": None}
    assert rows["m/p/s"]["judge_average"] == {"mean": None, "std": None}


def test_paired_bootstrap_uses_shared_document_indices():
    module = _summarize_module()
    zero_shot = np.array([0.0] * 50 + [1.0] * 50)
    sft = zero_shot + 0.2  # constant gain: paired CI has zero width, unpaired would not

    result = module.paired_bootstrap(zero_shot, sft)

    assert result["mean_diff"] == pytest.approx(0.2)
    assert result["ci95"] == pytest.approx((0.2, 0.2))
    with pytest.raises(ValueError):
        module.paired_bootstrap([1.0, 0.0], [1.0])
