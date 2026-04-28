# Upload preannotated data and annotate in Label Studio

[`scripts/label_studio/upload_with_preannotation.py`](../../scripts/label_studio/upload_with_preannotation.py) takes a preannotation parquet (the output of [`annotate_data.py`](../../scripts/annotation/annotate_data.py)) and:

1. creates the Label Studio project (if it doesn't exist),
2. sets the project's label interface from your XML form template,
3. creates one task per row (text → `data.text`),
4. instantiates the schema with the row's annotation columns and pushes it as a prediction on the task.

Annotators then review and correct the predictions in the Label Studio UI.

## Config

[`configs/upload_with_preannotation.yaml`](../../configs/upload_with_preannotation.yaml) — example for the Swiss Franc task. Key fields:

```yaml
ls_base_url: ${oc.env:LABEL_STUDIO_BASE_URL}
ls_api_key: ${oc.env:LABEL_STUDIO_API_KEY}
project_name: "Juddges Project: Swiss Frank v2"
text_field: "text"
label_interface: label_studio_toolkit/form_templates/swiss_frank.xml

model_version: gpt-4.1-2025-04-14
data_path: data/analysis/sprawy_frankowe/annotations/gpt-4.1-2025-04-14/test_annotations.parquet

filter_fields:
  sprawa_frankowiczow: Tak

annotation_schema: label_studio_toolkit.schemas.swiss_frank.SwissFrancJudgmentAnnotation

model:
  request_cache_db: .langchain_annotation_cache.db
```

| Field | Meaning |
|---|---|
| `project_name` | Title of the LS project. Created if missing; otherwise reused (tasks are appended). |
| `label_interface` | Path to the XML template that defines the form. Field names in the XML must match schema field names exactly. |
| `data_path` | Parquet produced by `annotate_data.py`. |
| `text_field` | Column with the document text. Becomes `task.data.<text_field>`. |
| `filter_fields` | `{column: value}` pairs applied as `df[df[col] == value]` before upload. Use to upload only a subset (e.g. only rows where the LLM said `sprawa_frankowiczow == "Tak"`). |
| `model_version` | Stored on each prediction so you can distinguish multiple LLM runs in the LS UI. |
| `annotation_schema` | Pydantic class — must match the form. |

## Run

```bash
python scripts/label_studio/upload_with_preannotation.py
```

To upload a different parquet / project:

```bash
python scripts/label_studio/upload_with_preannotation.py \
  project_name="My Project: Personal Rights" \
  label_interface=label_studio_toolkit/form_templates/personal_rights.xml \
  data_path=path/to/your_annotations.parquet \
  annotation_schema=label_studio_toolkit.schemas.personal_rights.PersonalRightsAnnotation \
  filter_fields='{naruszenie_dobr_osobistych: Tak}'
```

## Re-running upload safely

The script creates new tasks every time it runs and does not deduplicate. If you re-run with the same data, you'll get duplicate tasks. Either:

- delete the project's tasks from the LS UI before re-running, or
- create a new `project_name` for each re-run.

## Reviewing in the Label Studio UI

1. Open `LABEL_STUDIO_BASE_URL` in your browser and go to the project.
2. Open any task — the LLM prediction appears prefilled in the form.
3. Correct the fields where the LLM was wrong, then click **Submit** (or **Update** for a previously submitted task).
4. Use the project filters/views to split work across annotators.

The form layout (collapsible sections, choices, text areas) is entirely controlled by the XML template — see [`label_studio_toolkit/form_templates/swiss_frank.xml`](../form_templates/swiss_frank.xml) for a reference.

## Notes on the field-name contract

The XML form's `name="..."` attribute on each `Choices` / `TextArea` / `Number` element must equal the corresponding Pydantic field. `LabelStudioClient.create_prediction` looks up controls by field name via `label_interface.get_control(control_key)` ([`label_studio_toolkit/api/client.py:58`](../api/client.py#L58)) — a mismatch raises at upload time.

## Next

- [export.md](export.md) — once humans have reviewed the tasks, export them to a structured dataset.
