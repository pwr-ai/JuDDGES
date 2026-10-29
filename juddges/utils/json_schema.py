from typing import Any


def ie_schema_to_json_schema(ie_schema: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Converts an information extraction schema (configs/ie_schema) to a JSON schema.

    Every key is required so that the model cannot drop fields. Only fields with `required: true`
    must have a value; fields with `required: false` or without the `required` property (as in
    `validate_output_structure`) accept null, which is the "not found" value requested by the
    prompts.
    """
    properties = {name: _field_schema(spec) for name, spec in ie_schema.items()}
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def _field_schema(spec: dict[str, Any]) -> dict[str, Any]:
    schema = _non_null_schema(spec)
    if spec.get("required", False):
        return schema
    if "enum" in schema:
        return {"enum": [*schema["enum"], None]}
    return {"anyOf": [schema, {"type": "null"}]}


def _non_null_schema(spec: dict[str, Any]) -> dict[str, Any]:
    field_type = spec["type"]
    if field_type == "enum":
        return {"enum": list(spec["choices"])}
    if field_type in ("string", "date"):
        return {"type": "string"}
    if field_type == "boolean":
        return {"type": "boolean"}
    if field_type in ("int", "integer"):
        return {"type": "integer"}
    if field_type == "number":
        return {"type": "number"}
    if field_type == "list":
        return {"type": "array", "items": _non_null_schema(spec["items"])}
    raise ValueError(f"Unsupported field type for JSON schema: {field_type}")
