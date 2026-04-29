# Workflows

The toolkit supports three end-to-end workflows. Pick one based on whether you want human review and where your tasks come from.

## A — LLM preannotation + human review (recommended)

You have raw text data. The LLM produces a first-pass annotation, humans review and correct it in Label Studio, then you export the corrected dataset.

```
parquet/HF dataset
       │
       ▼
annotate_data.py  ───► *_annotations.parquet
       │
       ▼
upload_with_preannotation.py  ───► Label Studio project (tasks + predictions)
       │
       ▼
   human review in LS UI
       │
       ▼
LS export (JSON)  ───► scripts/label_studio/export_annotated_dataset.py  ───► dataset.json
```

Use this when:
- you have a clean text dataset (one text per row),
- you want high-quality labels (human-verified),
- the task is well-defined enough that an LLM gives a useful starting point.

Detailed instructions:
- [preannotation.md](preannotation.md) — running `annotate_data.py`.
- [upload-and-annotate.md](upload-and-annotate.md) — running `upload_with_preannotation.py` and reviewing in the UI.
- [export.md](export.md) — converting the LS export to a dataset.

## B — Preannotate tasks already in Label Studio

You already have a Label Studio project with tasks (e.g. you imported a CSV through the UI), and you want to add LLM predictions to them.

```
existing LS project (tasks already imported)
       │
       ▼
preannotate.py  ───► predictions attached to each task
       │
       ▼
   human review in LS UI
       │
       ▼
LS export (JSON)  ───► scripts/label_studio/export_annotated_dataset.py  ───► dataset.json
```

Use this when:
- annotators already started working in Label Studio without preannotation,
- you imported data through the LS UI and don't have it as a parquet,
- you want to add predictions to a subset of tasks based on Label Studio filters.

Detailed instructions:
- [preannotation.md](preannotation.md) — running `preannotate.py`.
- [export.md](export.md) — converting the LS export to a dataset.

## C — LLM-only (no human review)

You want a fully synthetic dataset annotated only by the LLM, e.g. for fine-tuning a smaller model or for fast iteration on the schema.

```
parquet/HF dataset
       │
       ▼
annotate_data.py  ───► *_annotations.parquet
       │
       ▼
scripts/annotation/export_annotated_dataset.py  ───► dataset.json
```

Use this when:
- you don't need human verification (e.g. training data, prototyping, schema iteration),
- you trust the LLM enough for the task,
- you don't want to pay the human-annotation cost.

Detailed instructions:
- [preannotation.md](preannotation.md) — running `annotate_data.py`.
- [export.md](export.md) — converting the parquet to a dataset.

## Decision matrix

| You have... | You want... | Use workflow |
|---|---|---|
| parquet/HF dataset | high-quality, human-verified labels | **A** |
| LS project with tasks | LLM predictions on those tasks for review | **B** |
| parquet/HF dataset | synthetic LLM-only labels | **C** |
| nothing yet | start fresh on a new task | first read [add-new-task.md](add-new-task.md) |
