# Export annotations to a dataset

There are two export scripts depending on which workflow you ran.

| Source of annotations | Script |
|---|---|
| Human-reviewed in Label Studio (workflow A or B) | [`scripts/label_studio/export_annotated_dataset.py`](../../scripts/label_studio/export_annotated_dataset.py) |
| LLM-only parquet (workflow C) | [`scripts/annotation/export_annotated_dataset.py`](../../scripts/annotation/export_annotated_dataset.py) |

Both produce the same output format:

- `dataset.json` — a JSON array of `{"context": <text>, "output": <pydantic-json>}` objects.
- `schema.yaml` — human-readable schema rendered by `Schema.get_schema_string()` ([`label_studio_toolkit/schemas/utils.py`](../schemas/utils.py)).

## A/B — Export from a Label Studio JSON export

### Step 1. Export from the Label Studio UI

In the project, click **Export** → choose **JSON** (full export). This is Label Studio's standard [JSON export format](https://labelstud.io/guide/export.html#Label-Studio-JSON-format-of-annotated-tasks), which contains each task's `data`, `annotations`, and `predictions` arrays. Save the file locally.

> **JSON vs JSON-MIN.** The script supports both shapes. With `--use-preannotation`, it reads from `predictions[0].result` (LLM output saved as a prediction — JSON format). Without it, it reads the human's submitted annotation values from the top-level entry fields (JSON-MIN format).

### Step 2. Run the export script

```bash
python scripts/label_studio/export_annotated_dataset.py \
  --input-file path/to/label-studio-export.json \
  --output-path data/label_studio/exports/personal_rights \
  --schema personal_rights
```

Flags:
- `--input-file` (required) — the JSON file exported from the LS UI.
- `--schema` (required) — schema to validate against. One of: `personal_rights`, `swiss_frank`, `en_appealcourt`. Add a new entry to `SCHEMA_MAP` in the script to support your own schema.
- `--output-path` — directory for `dataset.json` + `schema.yaml`. Default: `data/label_studio/exports`.
- `--use-preannotation` — read from `predictions` instead of human annotations. Default: `False`.

## C — Export an LLM-only parquet

For workflow C, you skip the Label Studio UI entirely.

```bash
python scripts/annotation/export_annotated_dataset.py \
  --input-file path/to/test_annotations.parquet \
  --output-path path/to/output_dir \
  --schema swiss_frank
```

Flags:
- `--input-file` (required) — parquet produced by `annotate_data.py`.
- `--output-path` (required) — output directory.
- `--schema` (required) — one of: `swiss_frank`, `en_appealcourt`. Add a new entry to `SCHEMA_MAP` in the script to support your own schema.

Output:
- `<input_file_stem>_dataset.json`
- `schema.yaml`

## Output format

The output format is a JSON array where each object has two keys:

```json
[
  {
    "context": "Sąd Okręgowy w Warszawie ...",
    "output": "{\"naruszenie_dobr_osobistych\": \"Tak\", \"podstawa_prawna\": [\"23 KC\", \"24 KC\"], ...}"
  }
]
```

`output` is a JSON string (the result of `model.model_dump_json()`), which makes the file safe to read with any JSONL/JSON tooling and easy to load back into the schema:

```python
schema_cls.model_validate_json(row["output"])
```

## Next

- [add-new-task.md](add-new-task.md) — define your own schema and form template.
