from juddges.evals.metrics import evaluate_boolean
from juddges.utils.json_schema import ie_schema_to_json_schema


def test_optional_boolean_field_accepts_null():
    schema = ie_schema_to_json_schema({"flag": {"type": "boolean", "required": False}})
    assert schema["properties"]["flag"] == {"anyOf": [{"type": "boolean"}, {"type": "null"}]}


def test_evaluate_boolean_matches_exactly_and_treats_null_as_value():
    assert evaluate_boolean(True, True) == {"match": 1}
    assert evaluate_boolean(None, False) == {"match": 0}
