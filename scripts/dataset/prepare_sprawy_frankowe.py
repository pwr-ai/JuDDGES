"""Converts the instruction-format `sprawy_frankowe` splits to the `swiss_franc_loans` layout.

Source records carry `prompt`, `context` and `output`. The `output` is fenced YAML produced by an
LLM extraction in December 2024 (silver labels, not expert-corrected gold). It follows the
42-field schema stored in `dataset_info.json`, which differs from `configs/ie_schema/swiss_franc_loans.yaml`
(44 fields, other key spellings). This script therefore also writes that schema as an IE schema
config and converts each `output` to a JSON string over the same keys; empty strings become null.
Only `context` and `output` are kept. `annotated.json` (the expert-corrected documents) is not
produced here. With a leakage report, `test_clean.json` keeps the test documents the report did
not flag.
"""

import json
import re
from pathlib import Path
from typing import Any

import typer
import yaml

from juddges.utils.misc import parse_yaml

FIELD_PATTERN = re.compile(
    r'^(?P<name>\S+): (?P<type>(?:list of )?enum \[.*?\]|[^,]+?), description: "(?P<description>.*?)", example: '
)


def parse_schema(schema_text: str) -> dict[str, dict[str, Any]]:
    schema: dict[str, dict[str, Any]] = {}
    for line in schema_text.splitlines():
        if not line.strip():
            continue
        match = FIELD_PATTERN.match(line)
        if match is None:
            raise ValueError(f"Cannot parse schema line: {line!r}")
        spec = _parse_type(match["type"])
        spec["description"] = match["description"]
        spec["required"] = False
        schema[match["name"]] = spec
    return schema


def _parse_type(type_text: str) -> dict[str, Any]:
    if type_text.startswith("list of enum"):
        return {"type": "list", "items": _parse_type(type_text.removeprefix("list of "))}
    if type_text.startswith("enum"):
        choices = type_text[type_text.index("[") + 1 : type_text.rindex("]")].split(", ")
        return {"type": "enum", "choices": choices}
    if type_text.startswith("date"):
        return {"type": "date"}
    if type_text in ("string", "boolean"):
        return {"type": type_text}
    raise ValueError(f"Unsupported type: {type_text!r}")


def convert_output(output: str, schema: dict[str, dict[str, Any]]) -> str:
    data = parse_yaml(output)
    unknown = set(data) - set(schema)
    if unknown:
        raise ValueError(f"Keys outside the schema: {sorted(unknown)}")
    converted = {name: _empty_to_null(data.get(name)) for name in schema}
    return json.dumps(converted, ensure_ascii=False)


def _empty_to_null(value: Any) -> Any:
    if value == "" or value == []:
        return None
    return value


def main(
    source_dir: Path = typer.Option(Path("data/datasets/pl/sprawy_frankowe")),
    target_dir: Path = typer.Option(Path("data/datasets/pl/swiss_franc_loans")),
    schema_file: Path = typer.Option(Path("configs/ie_schema/swiss_franc_loans_2024.yaml")),
    leakage_report: Path | None = typer.Option(
        None, help="check_split_leakage.py report; flagged test documents go to no test_clean.json"
    ),
) -> None:
    info = json.loads((source_dir / "dataset_info.json").read_text(encoding="utf-8"))
    schema = parse_schema(info["schema"])
    schema_file.write_text(
        yaml.safe_dump(schema, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    typer.echo(f"schema: {len(schema)} fields -> {schema_file}")

    target_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        records = json.loads((source_dir / f"{split}.jsonl").read_text(encoding="utf-8"))
        rows = [
            {"context": r["context"], "output": convert_output(r["output"], schema)}
            for r in records
        ]
        (target_dir / f"{split}.json").write_text(
            json.dumps(rows, ensure_ascii=False), encoding="utf-8"
        )
        typer.echo(f"{split}: {len(rows)} records -> {target_dir / f'{split}.json'}")
        if split == "test" and leakage_report is not None:
            report = json.loads(leakage_report.read_text(encoding="utf-8"))
            flagged = {f["index"] for f in report["flagged"] if f["split"] == "test"}
            clean = [r for i, r in enumerate(rows) if i not in flagged]
            (target_dir / "test_clean.json").write_text(
                json.dumps(clean, ensure_ascii=False), encoding="utf-8"
            )
            typer.echo(f"test_clean: {len(clean)} records ({len(flagged)} flagged removed)")


if __name__ == "__main__":
    typer.run(main)
