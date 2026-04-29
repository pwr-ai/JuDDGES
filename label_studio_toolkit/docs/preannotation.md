# Preannotation with an LLM

Two scripts produce LLM predictions:

- [`scripts/annotation/annotate_data.py`](../../scripts/annotation/annotate_data.py) — async batch annotation of a **parquet or HuggingFace dataset**, output is a parquet ready for Label Studio upload.
- [`scripts/label_studio/preannotate.py`](../../scripts/label_studio/preannotate.py) — annotate **tasks already in a Label Studio project**, predictions are pushed back to LS.

Both call [`LangChainOpenAIAnnotator`](../annotator.py), which wraps `langchain-openai`'s `ChatOpenAI` with `with_structured_output(schema, method="json_schema")`. Output is validated against the Pydantic schema you select.

## Common configuration

Every run is configured by a Hydra config under [`configs/`](../../configs/) with a `defaults` block that composes:

- a **model** config from [`configs/api_llm/`](../../configs/api_llm/) — defines `model.name`, `model.endpoint`, `model.request_cache_db`,
- a **prompt** config from [`configs/prompt/`](../../configs/prompt/) — defines `prompt.template` (must contain `{text}`),
- the **annotation_schema** — fully qualified Python path to a Pydantic model.

`OPENAI_API_KEY` must be set in `.env`. LLM calls are cached in a SQLite file (`model.request_cache_db`, defaults to `.langchain_annotation_cache.db`) so re-runs of the same input are free.

## `annotate_data.py` — annotate a dataset

Reads a dataset, calls the LLM concurrently (semaphore = 15), writes a parquet with the annotation columns plus `text`, `LLM`, `schema`, `language`.

### Configs that ship

- [`configs/annotate_data.yaml`](../../configs/annotate_data.yaml) — Polish CHF (Swiss Franc) example, parquet input.
- [`configs/annotate_data_en_appealcourt.yaml`](../../configs/annotate_data_en_appealcourt.yaml) — English appeal court example, HuggingFace input.

### Run

```bash
python scripts/annotation/annotate_data.py
```

By default, the example configs have `skip_train: true` and `skip_test: true` (annotate nothing). Override them:

```bash
python scripts/annotation/annotate_data.py skip_train=false skip_test=false
```

### Common overrides

```bash
# pick a different model config
python scripts/annotation/annotate_data.py model=../api_llm/gpt_5

# point at your own data
python scripts/annotation/annotate_data.py \
  dataset.train.path=path/to/train.parquet \
  dataset.test.path=path/to/test.parquet \
  output_dir=path/to/output

# use a HuggingFace dataset instead
python scripts/annotation/annotate_data.py \
  dataset.type=hf dataset.path=YourOrg/your-dataset \
  text_field=context
```

### Inputs

- `dataset.type`: `"parquet"` or `"hf"`.
- `dataset.train.path` / `dataset.test.path`: parquet file paths (when type is `parquet`).
- `dataset.path`: HF dataset id (when type is `hf`).
- `text_field`: column / field with the text to annotate (default `text`, `context` for the appeal-court example).
- `language`: passed to `async_annotate` and stored in output (used by some prompts via `{language}`).
- `annotation_schema`: e.g. `label_studio_toolkit.schemas.swiss_frank.SwissFrancJudgmentAnnotation`.

### Outputs

In `output_dir`:

- `train_annotations.parquet` (unless `skip_train=true`)
- `test_annotations.parquet` (unless `skip_test=true`)

Each row contains:
- one column per schema field (the LLM's structured prediction),
- `text` — the original text,
- `LLM` — the `model_version` string,
- `schema` — the JSON schema dump (for traceability),
- `language` — the language used.

This parquet is the input to [upload-and-annotate.md](upload-and-annotate.md) (workflow A) or to the LLM-only export script (workflow C).

## `preannotate.py` — preannotate tasks already in Label Studio

Iterates `client.get_tasks()` for the configured project, annotates each task's text with the LLM, and pushes a prediction back via `client.push_prediction(...)`. Use this when tasks were imported through the Label Studio UI (CSV upload) rather than via the toolkit.

### Config

[`configs/preannotate_label_studio.yaml`](../../configs/preannotate_label_studio.yaml). Key fields:

- `ls_base_url`, `ls_api_key` — read from `.env` (`LABEL_STUDIO_BASE_URL`, `LABEL_STUDIO_API_KEY`).
- `project_name` — the existing Label Studio project to annotate. The project is created if missing, but it must already contain tasks for any predictions to land.
- `text_field` — the key under `task.data` that holds the text (default `text`).
- `model_version` — version string stored on the prediction (defaults to `${model.name}`).
- `annotation_schema` — Python path to the Pydantic schema (must match the project's label interface).

### Run

```bash
python scripts/label_studio/preannotate.py
```

### Common overrides

```bash
# switch project + schema in one command
python scripts/label_studio/preannotate.py \
  project_name="My Project: Personal Rights" \
  annotation_schema=label_studio_toolkit.schemas.personal_rights.PersonalRightsAnnotation \
  prompt=annotate_personal_rights
```

### Outputs

Predictions appear under each task in the Label Studio UI, grouped by `model_version`. After human review you export the project from the UI — see [export.md](export.md).

## Adding a new model

Drop a YAML into [`configs/api_llm/`](../../configs/api_llm/):

```yaml
# configs/api_llm/gpt_4_1.yaml
name: gpt-4.1-2025-04-14
endpoint: null
request_cache_db: .langchain_annotation_cache.db
```

Then point any annotation config at it via the `defaults` list or a CLI override:

```bash
python scripts/annotation/annotate_data.py model=../api_llm/gpt_4_1
```

`name` is passed to `ChatOpenAI(model=...)`; any model id supported by your OpenAI-compatible endpoint works.

## Next

- [upload-and-annotate.md](upload-and-annotate.md) — push the parquet to Label Studio and review.
- [export.md](export.md) — export the final dataset.
