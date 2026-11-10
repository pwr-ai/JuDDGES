import json

import pytest

pytest.importorskip("langchain_core")

from juddges.llm_as_judge.data_model import PredictionLoader  # noqa: E402


def test_json_answer_that_is_not_an_object_is_a_parse_error():
    schema = {"decision": {"type": "enum", "choices": ["Tak", "Nie"]}}
    gold = json.dumps({"decision": "Tak"})
    preds = [
        {"answer": answer, "gold": gold} for answer in ['["Tak"]', "42", '{"decision": "Tak"}']
    ]

    parsed = PredictionLoader.load_predictions(schema, preds)

    assert sorted(parsed.errors) == [0, 1]
    assert sorted(parsed.predictions) == [2]
    assert parsed.missing_keys[0] == [] and parsed.extra_keys[1] == []
