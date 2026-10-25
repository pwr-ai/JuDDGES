import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from juddges.utils.json_schema import ie_schema_to_json_schema
from juddges.utils.run_record import file_sha256, write_run_record

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
    assert props["evidence"] == {"type": "array", "items": {"enum": ["x", "y"]}}
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
    assert set(record["library_versions"]) == {"torch", "vllm", "transformers", "peft", "trl"}
    assert record["wall_clock_seconds"] > 0


def _summarize_module():
    pytest.importorskip("typer")
    pytest.importorskip("pandas")
    pytest.importorskip("torchmetrics")
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
                {"decision": "Nie", "judgment_date": "2020-1-2", "summary": "s"}
            ),
            "gold": gold,
            "finish_reason": "stop",
        },
        {"answer": "I cannot answer", "gold": gold, "finish_reason": "length"},
    ]
    (tmp_path / "predictions.json").write_text(json.dumps(predictions))
    judge = {
        "all_results": {"0": {"decision": 1, "summary": 1}, "2": {"decision": 0, "summary": 0}}
    }
    (tmp_path / "llm_as_judge_some-judge.json").write_text(json.dumps(judge))

    result = module.summarize_run(tmp_path)

    assert result["num_docs"] == 3
    assert result["parsing_errors"] == 1
    assert result["finish_reason_length"] == 1
    assert result["per_field_exact_match"] == {"judgment_date": 2 / 3, "decision": 1 / 3}
    assert result["exact_match_date"] == 2 / 3  # "2020-1-2" parses to the same date
    assert result["exact_match_categorical"] == 1 / 3
    assert result["judge_average"] == 0.5
