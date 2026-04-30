# JuDDGES — Open-Science Evaluation Form (Q&A)

Compact, copy-paste-friendly companion to the deep-linked report at [`docs/open-science/index.md`](index.md). Each question below mirrors the standard FAIR4RS / EOSC software-sustainability evaluation form, answered against the current state of the repository.

Last verified against the repository: **2026-04-30**. Repository: <https://github.com/pwr-ai/JuDDGES>.

---

## Project context

**Q: Briefly describe the context and purpose.**

**A:** JuDDGES is an end-to-end research codebase for working with Polish (and to a lesser extent English) legal-judgment data. It covers six tightly integrated capabilities: (1) **data acquisition** from the Polish Common Courts API and the National Administrative Court (NSA); (2) **storage and semantic retrieval** in a Weaviate vector database (`legal_documents`, `document_chunks` collections); (3) **automated dataset analysis** producing descriptive statistics so every published dataset ships with a transparent profile; (4) **schema-driven information extraction** running LLM inference over the corpus against a user-defined Pydantic schema (vLLM and OpenAI-compatible back-ends); (5) **fine-tuning and evaluation** of Llama 3.1/3.2, Mistral, Bielik (Polish), Phi-4 via PEFT/LoRA on Unsloth, with both n-gram metrics and an LLM-as-judge protocol; (6) a **Human-in-the-Loop annotation toolkit** on Label Studio — annotation tasks are declared as Pydantic schemas + XML form templates, and the toolkit automates LLM preannotation, task and prediction upload, human review, and export as `dataset.json` + `schema.yaml`. Three reference annotation tasks ship in the repo (Swiss Franc, personal rights, English appeal-court). The codebase exists so the JuDDGES project can publish high-quality Polish legal datasets and reproducible model artefacts on the JuDDGES Hugging Face organisation, and can be reused by anyone reproducing the data-collection workflow, refreshing datasets, extracting structured information under a custom schema, or fine-tuning an LLM for a domain-specific task.

---

## DOCUMENTATION — License & accessibility

**Q: How can the repository be accessed by third parties?**

**A:** Public, registration-free GitHub at <https://github.com/pwr-ai/JuDDGES>. **Code** is licensed under **Apache 2.0** (`LICENSE`). **Documentation** is licensed separately under **CC BY 4.0** (`LICENSE-docs`). **Datasets** published to the Hugging Face organisation are CC BY 4.0; **fine-tuned models** are released under OpenRAIL-M. The four-way licensing model (code / docs / data / models) is declared in `pyproject.toml`'s `license` field and on each Hugging Face card's YAML metadata.

**Q: What type of documentation is available, provided with the project and delivered under the same conditions?**

**A:** Multi-layer documentation under the same open licenses: (a) top-level project overview, install steps (automated via `setup.sh` / `setup.bat` and manual via `uv` + `pyproject.toml`), and quick-start commands; (b) a `docs/` directory organised under the **Diátaxis** framework — `tutorials/`, `how-to/`, `reference/`, `explanation/` — built and deployed as a Material-for-MkDocs site by the `.github/workflows/docs-build-deploy.yaml` CI pipeline (with separate PR-preview and quality-checks workflows); (c) a dedicated open-science / FAIR4RS report at `docs/open-science/index.md` (rendered at the clean URL `/open-science/`); (d) executable Jupyter notebooks in `nbs/` and `dev_notebooks/`; (e) a sub-toolkit doc set in `label_studio_toolkit/docs/` (setup, workflows, preannotation, upload-and-annotate, export, add-new-task); (f) Hydra YAML configurations in `configs/` act as living reference for every published experiment.

**Q: Does the documentation describe how to use/build/deploy/install the project?**

**A:** **No web application is shipped** — JuDDGES is a research codebase composed of CLI scripts, DVC pipelines, and a Label Studio integration, not a hosted web service. Installation is documented two ways: automated (`setup.sh` / `setup.bat`) and manual (`uv venv .venv && uv pip install -e .` against `pyproject.toml` + `requirements.txt`). Build and developer-workflow targets are exposed via the `Makefile` (`make install`, `make install_unsloth`, `make fix`, `make check`, `make check-types`, `make test`, `make all`). Pipeline execution is documented as DVC stages declared in `dvc.yaml` (e.g. `dvc repro predict_raw_vllm`, `dvc repro predict_swiss_franc_loans_on_fine_tuned_vllm`, `dvc repro evaluate`, `dvc repro evaluate_llm_as_judge`), with matrix expansion over `(model × dataset × seed)`. External services run via Docker Compose: Weaviate (`weaviate/docker-compose.yml`), an extraction stack (`docker-compose.extraction.yml`), and an LLM+Postgres optimised stack (`docker-compose-llm-postgres-optimized.yml`). Deployment of the Label Studio annotation UI is documented in `label_studio_toolkit/docs/setup.md`.

---

## TESTING — Sample data & parameters

**Q: Are sample data and/or parameters that can be used to test the project available with the source code?**

**A:** Yes, in four categories. (a) **Automated tests** under `tests/` (pytest + coverage via `make test`, full quality sweep via `make all`): preprocessing tests (`tests/preprocessing/`), extraction tests (`tests/extraction/`), evaluation tests (`tests/evals/`), LLM-as-judge tests (`tests/llm_as_judge/`), and a Weaviate embedding/ingestion integration suite at `tests/embeddings/` with its own README, `conftest.py`, and `run_tests.py`. (b) **Sample data** committed under `data/sample_data/` — 100-row and 10-row CSV samples (`judgements-100-sample.csv`, `judgements-100-sample-with-retrieved-informations.csv`, `judgements-konfiskata-100-sample.csv`, `judgements-10-konfiskata-sample-with-retrieved-informations.csv`) — sufficient to exercise embedding, extraction, and evaluation end-to-end. Each sample is also DVC-tracked via `.dvc` pointer files. (c) **Use-case example scripts** under `scripts/` and `examples/`, including the instruct-dataset builders in `scripts/dataset/` and the Weaviate ingestion script `scripts/embed/ingest_to_weaviate.py`. (d) **Reference annotation tasks** in `label_studio_toolkit/` — all three tasks (Swiss Franc, personal rights, English appeal-court) ship with both a Pydantic schema and a Label Studio XML form template, wired through Hydra configs `configs/preannotate_label_studio.yaml`, `configs/upload_with_preannotation.yaml`, and `configs/annotate_data_en_appealcourt.yaml`.

---

## INTEROPERABILITY — Standard I/O formats

**Q: Do you use existing and standard input/output formats?**

**A:** Yes. **Datasets** stored as **Parquet** and distributed via the Hugging Face `datasets` library; **CSV** and **JSON** for tabular and metadata exports. **Configuration** in **YAML** (Hydra-structured, in `configs/`). **Dependencies** declared in `requirements.txt`, `pyproject.toml`, and a fully resolved `uv.lock`. **Vector data** persisted in **Weaviate** with predefined collections (`legal_documents`, `document_chunks`) and deterministic UUIDs for deduplication. **Embeddings** generated with the publicly available `sdadas/mmlw-roberta-large` model, so they are regenerable. The **annotation toolkit** consumes Parquet/HF datasets, calls any **OpenAI-compatible REST API** for LLM preannotation (works with OpenAI, vLLM, Ollama, LiteLLM, or self-hosted endpoints — see `label_studio_toolkit/api/client.py`), uses **Label Studio's standard JSON task format** for upload/review, and exports human-corrected annotations as a portable pair: **`dataset.json`** + **`schema.yaml`**.

---

## VERSIONING — Source-code version control

**Q: Do you use a version control system?**

**A:** Yes. **Git** + **GitHub** at <https://github.com/pwr-ai/JuDDGES>. Standard branch + PR review workflow with **pre-commit hooks** (`.pre-commit-config.yaml`, invoked via `make fix` / `make check`) enforcing formatting, linting, type-checking, markdown lint, and a custom spell-check before merge. **Continuous integration** under `.github/workflows/`: a Python test/quality pipeline (`python.yaml`), a docs build-and-deploy pipeline (`docs-build-deploy.yaml`), a docs PR-preview pipeline (`docs-pr-preview.yaml`), and a docs quality-checks pipeline (`docs-quality-checks.yaml`).

---

## REPRODUCIBILITY — Releases

**Q: Do you provide releases of your software?**

**A:** Yes — three layers. (a) **GitHub Releases backed by a Zenodo persistent DOI**: the project is archived on Zenodo with concept DOI [`10.5281/zenodo.19911970`](https://doi.org/10.5281/zenodo.19911970) (always resolves to the latest version) and a v0.1.0 version DOI [`10.5281/zenodo.19911971`](https://doi.org/10.5281/zenodo.19911971); plus a research-reproducibility Git tag `neurips_v0.1` capturing the SFT experiments on `pl-swiss-franc-loans`. (b) **DVC pipeline tracking**: `dvc.yaml` declares every stage end-to-end (preprocessing, embedding, instruct-dataset construction, SFT, raw and fine-tuned prediction, n-gram evaluation, LLM-as-judge evaluation), with matrix expansion fanning out across `(model × dataset × seed)`; `dvc.lock` records exact inputs/parameters/hashes/outputs so any reported artefact can be reproduced with `dvc repro <stage>`. (c) **Sample data version-tracked** through DVC `.dvc` pointer files.

**Q: How do you define language-specific dependencies of your project and their version?**

**A:** Three layers: a `requirements.txt` for runtime dependencies, a `pyproject.toml` describing the package and its optional install groups, and a fully resolved `uv.lock` pinning every transitive dependency to an exact version for byte-identical environment reconstruction. Recommended path uses **`uv`** (`uv venv .venv && uv pip install -e .`); `make install` is provided for `pip`-based workflows; `make install_unsloth` provisions a dedicated conda environment for fine-tuning. Required CUDA version (12.4 by default) is documented alongside install instructions. External services (Weaviate, Label Studio, optional Postgres/LLM stacks) are pinned via `docker compose` files for reproducible deployment without polluting the host system.

**Q: Do you state how to report bugs and/or usability problems by the software user(s)?**

**A:** Yes. Users are directed to the **GitHub Issues tracker** at <https://github.com/pwr-ai/JuDDGES/issues>. Four templated issue forms are shipped under `.github/ISSUE_TEMPLATE/`: `bug_report.yml`, `feature_request.yml`, `documentation.yml`, plus a `config.yml` that disables blank issues and links to GitHub Discussions and Security Advisories. PRs are reviewed against `.github/PULL_REQUEST_TEMPLATE.md` (summary, linked issue, test plan, pre-merge checklist). Contribution conventions are documented in `CONTRIBUTING.md`, and a coordinated vulnerability-disclosure policy is documented in `SECURITY.md` (with a private contact channel via GitHub Security Advisories + email backup, and a 5/14/90-day SLA).

**Q: Do you state how to report bugs and/or usability problems by the web app user(s)?**

**A:** **Not applicable** — JuDDGES does not ship a hosted web application; it is a research codebase that runs locally or on a user-controlled compute environment. The Label Studio UI used by the annotation toolkit is a third-party component whose own bug-reporting channels apply to UI defects; issues specific to the JuDDGES integration with Label Studio are reported on the same GitHub Issues tracker.

---

## RECOGNITION — Citation information

**Q: Do you include citation information (i.e. how to cite your software in the form of citation.cff, codemeta.json or bibtex)?**

**A:** Yes — **all four canonical formats**, all carrying the same software-author roster and pointing at the same reference publication:

1. **`CITATION.cff`** (Citation File Format v1.2.0) at the repository root — consumed by GitHub's "Cite this repository" widget, Zenodo, Zotero, Mendeley, OpenAIRE.
2. **`codemeta.json`** (CodeMeta v3.0 JSON-LD) at the repository root — consumed by HAL, OpenAIRE, Software Heritage, re3data.
3. **Zenodo persistent DOI** — concept DOI [`10.5281/zenodo.19911970`](https://doi.org/10.5281/zenodo.19911970) (always-latest) and version DOI [`10.5281/zenodo.19911971`](https://doi.org/10.5281/zenodo.19911971) (v0.1.0); badge rendered in `docs/index.md`.
4. **Copy-pasteable BibTeX** in the open-science report.

**Software-author roster** (eleven authors across five institutions): Łukasz Augustyniak, Jakub Binkowski, Albert Sawczyn, Tomasz Kajdanowicz (Wrocław University of Science and Technology); Michał Bernaczyk (University of Wrocław); Krzysztof Kamiński (Court of Appeal, Wrocław); Santosh Tirunagari, David Windridge, Mandeep K. Dhami (Middlesex University); Chérifa Boukacem-Zeghmouri, Candice Fillaud (Université Claude Bernard Lyon 1).

**Reference paper:** _"Bridging AI and Law: A Scalable Multi-Agent Platform for Quantitative Legal Analytics Across Millions of Documents"_ (Augustyniak et al., 2026), Bridge between AI and Law workshop, pp. 207–214. <https://openreview.net/forum?id=hWjsyTSWrY>. Note the BAIL paper's authorship (eleven WUST authors) is intentionally distinct from the broader software-collaboration roster above — software-author lists evolve after publication.
