# JuDDGES Documentation

Documentation for the **JuDDGES** project (Judicial Decision Data Gathering, Encoding, and Sharing) — a multilingual research platform for collecting, processing, and analyzing court decisions with NLP and LLM techniques.

The docs are organized using the [**Diátaxis framework**](https://diataxis.fr/) into four categories matched to user intent:

| Section | Use when you want to… |
|---|---|
| **[Tutorials](tutorials/)** | …learn by doing — guided, step-by-step lessons |
| **[How-To Guides](how-to/)** | …accomplish a specific task on a real problem |
| **[Reference](reference/)** | …look up exact specifications, schemas, or APIs |
| **[Explanation](explanation/)** | …understand the architecture and design decisions |

---

## Start here

- New to JuDDGES? Begin with the [Getting Started Tutorial](tutorials/GETTING_STARTED.md) (~30 min).
- Want the big picture? Read the [Project Overview](explanation/architecture/PROJECT_OVERVIEW.md) and [System Architecture](explanation/architecture/SYSTEM_ARCHITECTURE.md).
- Want to run the pipeline? Jump to [Quick start](#quick-start) below.

---

## Tutorials

Guided lessons that take you from zero to working code.

**Main series** (recommended order):

1. [Tutorial 1 — Your First Legal Document Analysis](tutorials/tutorial-01-first-legal-document-analysis.md) (beginner, 30–60 min)
2. [Tutorial 2 — Working with Legal Document Embeddings](tutorials/tutorial-02-embeddings.md) (intermediate, 45 min)
3. [Tutorial 3 — Fine-tuning Your First Legal LLM](tutorials/tutorial-03-model-finetuning.md) (advanced, 60+ min)
4. [Tutorial 4 — Advanced Information Extraction](tutorials/tutorial-04-advanced-extraction.md) (advanced, 45 min)
5. [Tutorial 5 — End-to-End Legal Analysis System](tutorials/tutorial-05-end-to-end-project.md) (expert, 90 min)

**Topic-focused tutorials**:

- [Getting Started](tutorials/GETTING_STARTED.md) — Quick 30-minute setup and first run
- [Gemini Extraction](tutorials/GEMINI_EXTRACTION.md) — Information extraction with Google Gemini
- [Langfuse Setup](tutorials/LANGFUSE_SETUP.md) — Observability for LLM pipelines
- [Git LFS Setup](tutorials/GIT_LFS_SETUP.md) — Large-file storage configuration

---

## How-To Guides

Task-oriented recipes for common problems. See the [full how-to index](how-to/).

**Data acquisition**:

- [Polish court data acquisition](how-to/data-acquisition/data_acquisition_pl_court.md)
- [NSA (Supreme Administrative Court) data acquisition](how-to/data-acquisition/data_acquisition_nsa.md)

**Embeddings & Weaviate**:

- [Embed and ingest to Weaviate](how-to/embeddings/embeddings_embed_and_ingest_weaviate.md)
- [Deploy Weaviate](how-to/embeddings/embeddings_deploy_weaviate.md)
- [Step-by-step ingestion guide](how-to/embeddings/INGESTION_GUIDE.md)
- [Universal dataset ingestion](how-to/embeddings/UNIVERSAL_INGESTION.md)
- [Streaming ingester](how-to/embeddings/STREAMING_INGESTER.md)
- [Optimize Weaviate ingestion](how-to/optimize-weaviate-ingestion.md)
- [Weaviate schema management](how-to/weaviate-schema-management.md)

**Information extraction**:

- [Iterative schema refinement](how-to/extraction/iterative_refinment.md)
- [Avoid reprocessing documents](how-to/extraction/avoid-reprocessing-documents.md)
- [Extraction storage setup](how-to/extraction-storage-setup.md)

**Infrastructure**:

- [Weaviate backup & restore](how-to/infrastructure/weaviate-backup-restore.md)

**Visualization (UMAP)**:

- [Overview](how-to/visualization/UMAP_VISUALIZATION.md) · [Sampling](how-to/visualization/UMAP_SAMPLING.md) · [Apply coordinates](how-to/visualization/UMAP_APPLY_COORDS.md)

**Documentation contribution**:

- [Contributing to documentation](how-to/documentation/CONTRIBUTING_TO_DOCS.md)

**Troubleshooting**:

- [Gemini API troubleshooting](how-to/troubleshooting/GEMINI_API_TROUBLESHOOTING.md)

---

## Reference

Authoritative specifications and API surface.

**API reference** (auto-generated from docstrings):

- [API index](reference/api/index.md)
- [Data loaders](reference/api/data/loaders.md) · [Data index](reference/api/data/index.md)
- [LLM factory](reference/api/llm/factory.md)
- [Gemini extraction chain](reference/api/extraction/gemini_chain.md)
- [Evaluation metrics](reference/api/evals/metrics.md)
- [Text chunker](reference/api/preprocessing/text_chunker.md)
- [Weaviate cursor pagination](reference/api/weaviate_cursor_pagination.md)
- [Raw text quick reference](reference/api/raw_text_quick_reference.md)

**Schemas**:

- [Document schema mapping](reference/schemas/DOCUMENT_SCHEMA_MAPPING.md) · [Dataset → Weaviate mapping](reference/schemas/dataset_weaviate_mapping.md)
- [Judgment extraction schema](reference/schemas/extraction_schema_judgments.md) · [Tax interpretations](reference/schemas/extraction_schema_tax_interpretations.md) · [Gemini schema](reference/schemas/gemini_extraction_schema.md)
- [LLM field extraction schema (YAML)](reference/schemas/llm_field_extraction_schema.yaml)

**Pipelines**:

- [DVC pipeline](reference/pipelines/DVC_PIPELINE.md) · [pipelines index](reference/pipelines/README.md)

**Standards**:

- [Documentation style guide](reference/STYLE_GUIDE.md)

---

## Explanation

Background and design rationale.

**Architecture**:

- [Project overview](explanation/architecture/PROJECT_OVERVIEW.md) — what JuDDGES is and what it does
- [System architecture](explanation/architecture/SYSTEM_ARCHITECTURE.md) — high-level component view
- [Data flow pipeline](explanation/architecture/DATA_FLOW_PIPELINE.md) — end-to-end data journey
- [Component relationships](explanation/architecture/COMPONENT_RELATIONSHIPS.md) — module dependencies
- [Model training flow](explanation/architecture/MODEL_TRAINING_FLOW.md) — training and inference pipelines
- [Weaviate integration](explanation/architecture/WEAVIATE_INTEGRATION.md) — vector database design

**Design topics**:

- [Extraction storage architecture](explanation/extraction/EXTRACTION_STORAGE_ARCHITECTURE.md)
- [Structured output implementation](explanation/structured-output-implementation.md)
- [Weaviate vs. document-DB metadata architecture](explanation/weaviate-vs-document-db-metadata-architecture.md)

---

## Quick start

```bash
# 1. Set up environment
./setup.sh                # or setup.bat on Windows

# 2. Acquire data
python scripts/data_acquisition/pl_court_data_pipeline.py

# 3. Generate embeddings and ingest to Weaviate
dvc repro embed
python scripts/embed/ingest_to_weaviate.py

# 4. Fine-tune and evaluate
dvc repro sft
dvc repro predict
dvc repro evaluate
```

See the [Getting Started Tutorial](tutorials/GETTING_STARTED.md) for the full setup walkthrough.

---

## License & citation

- **Code**: MIT License (see [`LICENSE`](../LICENSE))
- **Data**: CC-BY 4.0 · **Models**: Apache 2.0 · **Documentation**: CC-BY 4.0

If you use JuDDGES in research, please cite the project repository at <https://github.com/pwr-ai/JuDDGES>.

---

## Contributing

- Code & issues: <https://github.com/pwr-ai/JuDDGES>
- Documentation: see the [contribution guide](how-to/documentation/CONTRIBUTING_TO_DOCS.md) and [style guide](reference/STYLE_GUIDE.md)
