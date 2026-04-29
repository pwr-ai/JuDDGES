# Label Studio Toolkit

A generic framework for LLM-assisted annotation of legal documents in [Label Studio](https://labelstud.io/).

You define an annotation task as a Pydantic schema plus a Label Studio XML form. The toolkit takes care of:

- batch-annotating raw text with an LLM (preannotation),
- creating a Label Studio project, uploading tasks, attaching predictions,
- exporting human-reviewed annotations into a structured dataset.

Three example tasks ship with the toolkit and serve as references for adding your own.

## Architecture

```
                          ┌─────────────────────────────┐
                          │   your annotation task      │
                          │  ┌──────────┐ ┌──────────┐  │
                          │  │ Pydantic │ │ LS XML   │  │
                          │  │  schema  │ │ template │  │
                          │  └────┬─────┘ └────┬─────┘  │
                          └───────┼────────────┼────────┘
                                  │            │
            text data ─────► annotate_data.py ─┘
            (parquet/HF)         │
                                 ▼
                       *_annotations.parquet
                                 │
                                 ▼
                  upload_with_preannotation.py
                                 │
                                 ▼
                ┌───────── Label Studio ──────────┐
                │   tasks + LLM predictions       │
                │   reviewed/corrected by humans  │
                └─────────────┬───────────────────┘
                              │ (export JSON)
                              ▼
              scripts/label_studio/export_annotated_dataset.py
                              │
                              ▼
                     dataset.json + schema.yaml
```

## Components

| Path | What it is |
|---|---|
| [`label_studio_toolkit/schemas/`](../schemas/) | Pydantic models for annotation outputs. Three example schemas: `swiss_frank`, `personal_rights`, `en_appealcourt`. Add your own. |
| [`label_studio_toolkit/form_templates/`](../form_templates/) | Label Studio XML form templates rendering schemas as collapsible-section UIs. Two examples ship: `swiss_frank.xml`, `personal_rights.xml`. Add your own. |
| [`label_studio_toolkit/api/client.py`](../api/client.py) | `LabelStudioClient` — wrapper around `label-studio-sdk` for projects, tasks, and predictions. |
| [`label_studio_toolkit/annotator.py`](../annotator.py) | `LangChainOpenAIAnnotator` — wraps an OpenAI ChatGPT model into a structured-output chain producing Pydantic instances. |
| [`scripts/annotation/annotate_data.py`](../../scripts/annotation/annotate_data.py) | Async batch LLM annotation of a parquet/HF dataset → `*_annotations.parquet`. |
| [`scripts/label_studio/upload_with_preannotation.py`](../../scripts/label_studio/upload_with_preannotation.py) | Upload preannotated rows to Label Studio as tasks with predictions. |
| [`scripts/label_studio/preannotate.py`](../../scripts/label_studio/preannotate.py) | Add LLM predictions to tasks that already exist in a Label Studio project. |
| [`scripts/label_studio/export_annotated_dataset.py`](../../scripts/label_studio/export_annotated_dataset.py) | Convert a Label Studio JSON export to `dataset.json` + `schema.yaml`. |
| [`scripts/annotation/export_annotated_dataset.py`](../../scripts/annotation/export_annotated_dataset.py) | Convert an LLM-annotation parquet directly to `dataset.json` (no human review). |
| [`configs/`](../../configs/) | Hydra configs for every script (`preannotate_label_studio.yaml`, `upload_with_preannotation.yaml`, `annotate_data*.yaml`) and prompt templates under `configs/prompt/`. |

## Documentation

Read in order:

1. [setup.md](setup.md) — install Label Studio (self-hosted or cloud), set env vars, install Python deps.
2. [workflows.md](workflows.md) — the three end-to-end workflows and when to pick each.
3. [preannotation.md](preannotation.md) — generate LLM predictions with `annotate_data.py` and `preannotate.py`.
4. [upload-and-annotate.md](upload-and-annotate.md) — upload preannotated tasks and review them in the Label Studio UI.
5. [export.md](export.md) — export human-reviewed annotations to a structured dataset.
6. [add-new-task.md](add-new-task.md) — the main extensibility guide: define your own schema, form template, prompt, and config.

## Requirements

- Python 3.11
- A running Label Studio instance (self-hosted or `https://app.heartex.com/`)
- `OPENAI_API_KEY` for the OpenAI annotator
- Toolkit dependencies are installed with the project (`pip install -e .` from repo root)
