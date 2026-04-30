# JuDDGES — Open-Science & FAIR4RS Report

Software-sustainability and reproducibility report for the **JuDDGES** codebase, structured as direct answers to standard open-science evaluation criteria (FAIR4RS / EOSC software-sustainability).

Repository: <https://github.com/pwr-ai/JuDDGES>
Last verified against the repository: **2026-04-30**.

> Looking for a blank version of this form to use on a different project? See the reusable [`template.md`](https://github.com/pwr-ai/JuDDGES/blob/master/docs/open-science/template.md) in this directory.

---

## 1. Project context

**Q: Briefly describe the context and purpose.**

**A:** **JuDDGES** is an end-to-end research codebase for working with Polish (and to a lesser extent English) legal-judgment data. It covers six tightly integrated capabilities:

1. **Data acquisition** from the Polish Common Courts API and the National Administrative Court (NSA), implemented under [`scripts/nsa/`](https://github.com/pwr-ai/JuDDGES/tree/master/scripts/nsa) and the data loaders in [`juddges/data/`](https://github.com/pwr-ai/JuDDGES/tree/master/juddges/data).
2. **Storage and semantic retrieval** of judgments in a Weaviate vector database (`legal_documents`, `document_chunks` collections), with an ingestion pipeline ([`scripts/embed/ingest_to_weaviate.py`](https://github.com/pwr-ai/JuDDGES/tree/master/scripts/embed)) and a Docker-Compose deployment in [`weaviate/`](https://github.com/pwr-ai/JuDDGES/tree/master/weaviate).
3. **Automated dataset analysis** producing descriptive statistics and quality reports under [`scripts/analytics/`](https://github.com/pwr-ai/JuDDGES/tree/master/scripts/analytics) and notebooks in [`nbs/`](https://github.com/pwr-ai/JuDDGES/tree/master/nbs), so every published dataset ships with a transparent statistical profile.
4. **Schema-driven information extraction** running LLM inference over the corpus against a user-defined Pydantic schema ([`juddges/models/`](https://github.com/pwr-ai/JuDDGES/tree/master/juddges/models), [`scripts/extraction/`](https://github.com/pwr-ai/JuDDGES/tree/master/scripts/extraction)), supporting both local vLLM and OpenAI-compatible API back-ends.
5. **Fine-tuning and evaluation** of Llama 3.1/3.2, Mistral, Bielik (Polish), Phi-4 via PEFT/LoRA on Unsloth ([`scripts/sft/`](https://github.com/pwr-ai/JuDDGES/tree/master/scripts/sft)), with both n-gram metrics and an LLM-as-judge protocol ([`juddges/evaluation/`](https://github.com/pwr-ai/JuDDGES/tree/master/juddges/evaluation)).
6. A **Human-in-the-Loop annotation toolkit** on Label Studio in [`label_studio_toolkit/`](https://github.com/pwr-ai/JuDDGES/tree/master/label_studio_toolkit) — annotation tasks declared as Pydantic schemas + XML form templates; the toolkit automates LLM preannotation, task and prediction upload, human review and correction, and export of the corrected annotations as a structured dataset (`dataset.json` + `schema.yaml`). Three reference tasks ship in the repo: Swiss Franc loan cases ([`schemas/swiss_frank.py`](https://github.com/pwr-ai/JuDDGES/blob/master/label_studio_toolkit/schemas/swiss_frank.py) + [`form_templates/swiss_frank.xml`](https://github.com/pwr-ai/JuDDGES/blob/master/label_studio_toolkit/form_templates/swiss_frank.xml)), personal rights ([`schemas/personal_rights.py`](https://github.com/pwr-ai/JuDDGES/blob/master/label_studio_toolkit/schemas/personal_rights.py) + [`form_templates/personal_rights.xml`](https://github.com/pwr-ai/JuDDGES/blob/master/label_studio_toolkit/form_templates/personal_rights.xml)), and English appeal-court ([`schemas/en_appealcourt.py`](https://github.com/pwr-ai/JuDDGES/blob/master/label_studio_toolkit/schemas/en_appealcourt.py) + [`form_templates/en_appealcourt.xml`](https://github.com/pwr-ai/JuDDGES/blob/master/label_studio_toolkit/form_templates/en_appealcourt.xml), driven by [`configs/annotate_data_en_appealcourt.yaml`](https://github.com/pwr-ai/JuDDGES/blob/master/configs/annotate_data_en_appealcourt.yaml)).

The codebase exists so the JuDDGES project can publish high-quality Polish legal datasets and reproducible model artefacts on the [JuDDGES Hugging Face organisation](https://huggingface.co/JuDDGES). It can be reused by anyone reproducing the data-collection workflow, refreshing datasets with newly published judgments, extracting structured information under a custom schema, or fine-tuning an LLM for a domain-specific task.

---

## 2. Evaluation criteria

### DOCUMENTATION — License & accessibility

**Q: How can the repository be accessed by third parties?**

**A:** Public, registration-free GitHub at <https://github.com/pwr-ai/JuDDGES>. Four-way licensing model:

- **Code** under **[Apache 2.0](https://github.com/pwr-ai/JuDDGES/blob/master/LICENSE)** (also declared in [`pyproject.toml`](https://github.com/pwr-ai/JuDDGES/blob/master/pyproject.toml) `license` field).
- **Documentation** under **[CC BY 4.0](https://github.com/pwr-ai/JuDDGES/blob/master/LICENSE-docs)**.
- **Datasets** on Hugging Face under **CC BY 4.0** (declared in each dataset card's YAML metadata).
- **Fine-tuned models** under **OpenRAIL-M** (declared in each model card's YAML metadata).

**Q: What type of documentation is available, provided with the project and delivered under the same conditions?**

**A:** Multi-layer documentation under the same open licenses:

- (a) Top-level project overview, install steps (automated via [`setup.sh`](https://github.com/pwr-ai/JuDDGES/blob/master/setup.sh) / [`setup.bat`](https://github.com/pwr-ai/JuDDGES/blob/master/setup.bat) and manual via [`uv`](https://github.com/astral-sh/uv) + [`pyproject.toml`](https://github.com/pwr-ai/JuDDGES/blob/master/pyproject.toml)), plus quick-start commands.
- (b) [`docs/`](https://github.com/pwr-ai/JuDDGES/tree/master/docs) directory organised under the **Diátaxis** framework — [`tutorials/`](https://github.com/pwr-ai/JuDDGES/tree/master/docs/tutorials), [`how-to/`](https://github.com/pwr-ai/JuDDGES/tree/master/docs/how-to), [`reference/`](https://github.com/pwr-ai/JuDDGES/tree/master/docs/reference), [`explanation/`](https://github.com/pwr-ai/JuDDGES/tree/master/docs/explanation) — built and deployed as a Material-for-MkDocs site by [`.github/workflows/docs-build-deploy.yaml`](https://github.com/pwr-ai/JuDDGES/blob/master/.github/workflows/docs-build-deploy.yaml) (with separate PR-preview and quality-checks workflows).
- (c) This open-science / FAIR4RS report at [`docs/open-science/index.md`](https://github.com/pwr-ai/JuDDGES/blob/master/docs/open-science/index.md), rendered at the clean URL `/open-science/`.
- (d) Executable Jupyter notebooks in [`nbs/`](https://github.com/pwr-ai/JuDDGES/tree/master/nbs) and [`dev_notebooks/`](https://github.com/pwr-ai/JuDDGES/tree/master/dev_notebooks).
- (e) Sub-toolkit docs in [`label_studio_toolkit/docs/`](https://github.com/pwr-ai/JuDDGES/tree/master/label_studio_toolkit/docs): [`setup`](https://github.com/pwr-ai/JuDDGES/blob/master/label_studio_toolkit/docs/setup.md), [`workflows`](https://github.com/pwr-ai/JuDDGES/blob/master/label_studio_toolkit/docs/workflows.md), [`preannotation`](https://github.com/pwr-ai/JuDDGES/blob/master/label_studio_toolkit/docs/preannotation.md), [`upload-and-annotate`](https://github.com/pwr-ai/JuDDGES/blob/master/label_studio_toolkit/docs/upload-and-annotate.md), [`export`](https://github.com/pwr-ai/JuDDGES/blob/master/label_studio_toolkit/docs/export.md), [`add-new-task`](https://github.com/pwr-ai/JuDDGES/blob/master/label_studio_toolkit/docs/add-new-task.md).
- (f) Hydra YAML configurations in [`configs/`](https://github.com/pwr-ai/JuDDGES/tree/master/configs) act as living reference for every published experiment.

**Q: Does the documentation describe how to use/build/deploy/install the project?**

**A:** **No web application is shipped** — JuDDGES is a research codebase composed of CLI scripts, DVC pipelines, and a Label Studio integration, not a hosted web service. The accompanying documentation therefore focuses on what is actually delivered:

- **Installation** is documented two ways: automated via [`setup.sh`](https://github.com/pwr-ai/JuDDGES/blob/master/setup.sh) / [`setup.bat`](https://github.com/pwr-ai/JuDDGES/blob/master/setup.bat), and manual via `uv venv .venv && uv pip install -e .` against [`pyproject.toml`](https://github.com/pwr-ai/JuDDGES/blob/master/pyproject.toml) + [`requirements.txt`](https://github.com/pwr-ai/JuDDGES/blob/master/requirements.txt).
- **Build / dev workflows** through the project [`Makefile`](https://github.com/pwr-ai/JuDDGES/blob/master/Makefile): `make install`, `make install_unsloth`, `make fix`, `make check`, `make check-types`, `make test`, `make all`.
- **Pipeline execution** as DVC stages in [`dvc.yaml`](https://github.com/pwr-ai/JuDDGES/blob/master/dvc.yaml) (e.g. `dvc repro predict_raw_vllm`, `dvc repro predict_swiss_franc_loans_on_fine_tuned_vllm`, `dvc repro evaluate`, `dvc repro evaluate_llm_as_judge`), with matrix expansion over `(model × dataset × seed)`.
- **External services** via Docker Compose: Weaviate ([`weaviate/`](https://github.com/pwr-ai/JuDDGES/tree/master/weaviate)), an extraction stack ([`docker-compose.extraction.yml`](https://github.com/pwr-ai/JuDDGES/blob/master/docker-compose.extraction.yml)), and an LLM+Postgres optimised stack ([`docker-compose-llm-postgres-optimized.yml`](https://github.com/pwr-ai/JuDDGES/blob/master/docker-compose-llm-postgres-optimized.yml)).
- **Label Studio annotation UI** deployment is documented in [`label_studio_toolkit/docs/setup.md`](https://github.com/pwr-ai/JuDDGES/blob/master/label_studio_toolkit/docs/setup.md).

---

### TESTING — Sample data & parameters

**Q: Are sample data and/or parameters that can be used to test the project available with the source code?**

**A:** Yes, in four categories.

- (a) **Automated tests** under [`tests/`](https://github.com/pwr-ai/JuDDGES/tree/master/tests) (pytest + coverage via `make test`, full sweep via `make all`): preprocessing tests ([`tests/preprocessing/`](https://github.com/pwr-ai/JuDDGES/tree/master/tests/preprocessing)), extraction tests ([`tests/extraction/`](https://github.com/pwr-ai/JuDDGES/tree/master/tests/extraction)), evaluation tests ([`tests/evals/`](https://github.com/pwr-ai/JuDDGES/tree/master/tests/evals)), an LLM-as-judge suite ([`tests/llm_as_judge/`](https://github.com/pwr-ai/JuDDGES/tree/master/tests/llm_as_judge)), and a Weaviate embedding/ingestion integration suite ([`tests/embeddings/`](https://github.com/pwr-ai/JuDDGES/tree/master/tests/embeddings) with [`README.md`](https://github.com/pwr-ai/JuDDGES/blob/master/tests/embeddings/README.md), [`conftest.py`](https://github.com/pwr-ai/JuDDGES/blob/master/tests/embeddings/conftest.py), [`run_tests.py`](https://github.com/pwr-ai/JuDDGES/blob/master/tests/embeddings/run_tests.py)).
- (b) **Sample data** committed under [`data/sample_data/`](https://github.com/pwr-ai/JuDDGES/tree/master/data/sample_data) — 100-row and 10-row CSV samples ([`judgements-100-sample.csv`](https://github.com/pwr-ai/JuDDGES/blob/master/data/sample_data/judgements-100-sample.csv), [`judgements-100-sample-with-retrieved-informations.csv`](https://github.com/pwr-ai/JuDDGES/blob/master/data/sample_data/judgements-100-sample-with-retrieved-informations.csv), [`judgements-konfiskata-100-sample.csv`](https://github.com/pwr-ai/JuDDGES/blob/master/data/sample_data/judgements-konfiskata-100-sample.csv), [`judgements-10-konfiskata-sample-with-retrieved-informations.csv`](https://github.com/pwr-ai/JuDDGES/blob/master/data/sample_data/judgements-10-konfiskata-sample-with-retrieved-informations.csv)) — sufficient to exercise embedding, extraction, and evaluation end-to-end. Each sample is also DVC-tracked via `.dvc` pointer files.
- (c) **Use-case example scripts** under [`scripts/`](https://github.com/pwr-ai/JuDDGES/tree/master/scripts) and [`examples/`](https://github.com/pwr-ai/JuDDGES/tree/master/examples), notably the instruct-dataset builders in [`scripts/dataset/`](https://github.com/pwr-ai/JuDDGES/tree/master/scripts/dataset) and the Weaviate ingestion script [`scripts/embed/ingest_to_weaviate.py`](https://github.com/pwr-ai/JuDDGES/tree/master/scripts/embed).
- (d) **Reference annotation tasks** in [`label_studio_toolkit/`](https://github.com/pwr-ai/JuDDGES/tree/master/label_studio_toolkit) — all three tasks ship with both a Pydantic schema and a Label Studio XML form, wired through [`configs/preannotate_label_studio.yaml`](https://github.com/pwr-ai/JuDDGES/blob/master/configs/preannotate_label_studio.yaml), [`configs/upload_with_preannotation.yaml`](https://github.com/pwr-ai/JuDDGES/blob/master/configs/upload_with_preannotation.yaml), and [`configs/annotate_data_en_appealcourt.yaml`](https://github.com/pwr-ai/JuDDGES/blob/master/configs/annotate_data_en_appealcourt.yaml).

---

### INTEROPERABILITY — Standard I/O formats

**Q: Do you use existing and standard input/output formats?**

**A:** Yes. **Datasets** stored as **Parquet** and distributed via the Hugging Face `datasets` library; **CSV** and **JSON** for tabular and metadata exports (samples under [`data/sample_data/`](https://github.com/pwr-ai/JuDDGES/tree/master/data/sample_data)). **Configuration** in **YAML** ([Hydra-structured](https://github.com/pwr-ai/JuDDGES/tree/master/configs)). **Dependencies** declared in [`requirements.txt`](https://github.com/pwr-ai/JuDDGES/blob/master/requirements.txt), [`pyproject.toml`](https://github.com/pwr-ai/JuDDGES/blob/master/pyproject.toml), and a fully resolved [`uv.lock`](https://github.com/pwr-ai/JuDDGES/blob/master/uv.lock). **Vector data** persisted in **Weaviate** with predefined collections (`legal_documents`, `document_chunks`) and deterministic UUIDs for deduplication. **Embeddings** generated with the publicly available `sdadas/mmlw-roberta-large` model so they are regenerable. The **annotation toolkit** consumes Parquet/HF datasets, calls any **OpenAI-compatible REST API** for LLM preannotation (works with OpenAI, vLLM, Ollama, LiteLLM, or self-hosted endpoints — see [`label_studio_toolkit/api/client.py`](https://github.com/pwr-ai/JuDDGES/blob/master/label_studio_toolkit/api/client.py)), uses **Label Studio's standard JSON task format** for upload/review (see [`scripts/label_studio/upload_with_preannotation.py`](https://github.com/pwr-ai/JuDDGES/blob/master/scripts/label_studio/upload_with_preannotation.py)), and exports human-corrected annotations as a portable pair: **`dataset.json`** + **`schema.yaml`** ([`scripts/label_studio/export_annotated_dataset.py`](https://github.com/pwr-ai/JuDDGES/blob/master/scripts/label_studio/export_annotated_dataset.py)).

---

### VERSIONING — Source-code version control

**Q: Do you use a version control system?**

**A:** Yes. **Git** + **GitHub** at <https://github.com/pwr-ai/JuDDGES>. Standard branch + PR review workflow with **pre-commit hooks** ([`.pre-commit-config.yaml`](https://github.com/pwr-ai/JuDDGES/blob/master/.pre-commit-config.yaml), invoked via `make fix` / `make check`) enforcing formatting, linting, type-checking, markdown lint, and a custom spell-check before merge. **Continuous integration** under [`.github/workflows/`](https://github.com/pwr-ai/JuDDGES/tree/master/.github/workflows): a Python test/quality pipeline ([`python.yaml`](https://github.com/pwr-ai/JuDDGES/blob/master/.github/workflows/python.yaml)), a docs build-and-deploy pipeline ([`docs-build-deploy.yaml`](https://github.com/pwr-ai/JuDDGES/blob/master/.github/workflows/docs-build-deploy.yaml)), a docs PR-preview pipeline ([`docs-pr-preview.yaml`](https://github.com/pwr-ai/JuDDGES/blob/master/.github/workflows/docs-pr-preview.yaml)), and a docs quality-checks pipeline ([`docs-quality-checks.yaml`](https://github.com/pwr-ai/JuDDGES/blob/master/.github/workflows/docs-quality-checks.yaml)).

---

### REPRODUCIBILITY — Releases

**Q: Do you provide releases of your software?**

**A:** Yes — three layers.

- (a) **GitHub Releases backed by a Zenodo persistent DOI**: archived on Zenodo with concept DOI [`10.5281/zenodo.19911970`](https://doi.org/10.5281/zenodo.19911970) (always-latest) and v0.1.0 version DOI [`10.5281/zenodo.19911971`](https://doi.org/10.5281/zenodo.19911971); plus a research-reproducibility Git tag [`neurips_v0.1`](https://github.com/pwr-ai/JuDDGES/tree/neurips_v0.1) capturing the SFT experiments on `pl-swiss-franc-loans`.
- (b) **DVC pipeline tracking** via [`dvc.yaml`](https://github.com/pwr-ai/JuDDGES/blob/master/dvc.yaml) (preprocessing, embedding, instruct-dataset construction, SFT, raw and fine-tuned prediction, n-gram and LLM-as-judge evaluation), with matrix expansion over `(model × dataset × seed)`. The lockfile [`dvc.lock`](https://github.com/pwr-ai/JuDDGES/blob/master/dvc.lock) records exact inputs/parameters/hashes/outputs so any reported artefact can be reproduced with `dvc repro <stage>`.
- (c) **Sample data** version-tracked through DVC `.dvc` pointer files under [`data/sample_data/`](https://github.com/pwr-ai/JuDDGES/tree/master/data/sample_data).

**Q: How do you define language-specific dependencies of your project and their version?**

**A:** Three layers, all committed to the repository:

- A [`requirements.txt`](https://github.com/pwr-ai/JuDDGES/blob/master/requirements.txt) for runtime dependencies.
- A [`pyproject.toml`](https://github.com/pwr-ai/JuDDGES/blob/master/pyproject.toml) describing the package and its optional install groups.
- A fully resolved [`uv.lock`](https://github.com/pwr-ai/JuDDGES/blob/master/uv.lock) lockfile pinning every transitive dependency to an exact version for byte-identical environment reconstruction.

Recommended path uses **[`uv`](https://github.com/astral-sh/uv)** (`uv venv .venv && uv pip install -e .`); `make install` is provided for `pip`-based workflows; `make install_unsloth` provisions a dedicated conda environment for fine-tuning. Required CUDA version (12.4 by default) is documented alongside install instructions. External services (Weaviate, Label Studio, optional Postgres/LLM stacks) are pinned via `docker compose` files: [`docker-compose.yml`](https://github.com/pwr-ai/JuDDGES/blob/master/docker-compose.yml), [`docker-compose.extraction.yml`](https://github.com/pwr-ai/JuDDGES/blob/master/docker-compose.extraction.yml), [`docker-compose-llm-postgres-optimized.yml`](https://github.com/pwr-ai/JuDDGES/blob/master/docker-compose-llm-postgres-optimized.yml).

**Q: Do you state how to report bugs and/or usability problems by the software user(s)?**

**A:** Yes. Users are directed to the **GitHub Issues tracker** at <https://github.com/pwr-ai/JuDDGES/issues>. Four templated issue forms ship under [`.github/ISSUE_TEMPLATE/`](https://github.com/pwr-ai/JuDDGES/tree/master/.github/ISSUE_TEMPLATE): [`bug_report.yml`](https://github.com/pwr-ai/JuDDGES/blob/master/.github/ISSUE_TEMPLATE/bug_report.yml), [`feature_request.yml`](https://github.com/pwr-ai/JuDDGES/blob/master/.github/ISSUE_TEMPLATE/feature_request.yml), [`documentation.yml`](https://github.com/pwr-ai/JuDDGES/blob/master/.github/ISSUE_TEMPLATE/documentation.yml), plus a [`config.yml`](https://github.com/pwr-ai/JuDDGES/blob/master/.github/ISSUE_TEMPLATE/config.yml) that disables blank issues and links to GitHub Discussions and Security Advisories. PRs are reviewed against [`.github/PULL_REQUEST_TEMPLATE.md`](https://github.com/pwr-ai/JuDDGES/blob/master/.github/PULL_REQUEST_TEMPLATE.md). Contribution conventions live in [`CONTRIBUTING.md`](https://github.com/pwr-ai/JuDDGES/blob/master/CONTRIBUTING.md); a coordinated vulnerability-disclosure policy lives in [`SECURITY.md`](https://github.com/pwr-ai/JuDDGES/blob/master/SECURITY.md) (private channel via GitHub Security Advisories + email backup, 5/14/90-day SLA).

**Q: Do you state how to report bugs and/or usability problems by the web app user(s)?**

**A:** **Not applicable** — JuDDGES does not ship a hosted web application; it is a research codebase that runs locally or on a user-controlled compute environment. The Label Studio UI used by the annotation toolkit is a third-party component whose own bug-reporting channels apply to UI defects; issues specific to the JuDDGES integration with Label Studio are reported on the same [GitHub Issues tracker](https://github.com/pwr-ai/JuDDGES/issues).

---

### RECOGNITION — Citation information

**Q: Do you include citation information (i.e. how to cite your software in the form of citation.cff, codemeta.json or bibtex)?**

**A:** Yes — **all four canonical formats**, all carrying the same software-author roster and pointing at the same reference publication:

1. **[`CITATION.cff`](https://github.com/pwr-ai/JuDDGES/blob/master/CITATION.cff)** — Citation File Format v1.2.0 at the repository root. Consumed by GitHub's "Cite this repository" widget, Zenodo, Zotero, Mendeley, OpenAIRE.
2. **[`codemeta.json`](https://github.com/pwr-ai/JuDDGES/blob/master/codemeta.json)** — CodeMeta v3.0 JSON-LD (`@context: https://w3id.org/codemeta/3.0`, `@type: SoftwareSourceCode`). Consumed by HAL, OpenAIRE, Software Heritage, re3data.
3. **Zenodo persistent DOI** — concept DOI [`10.5281/zenodo.19911970`](https://doi.org/10.5281/zenodo.19911970) (always-latest) and version DOI [`10.5281/zenodo.19911971`](https://doi.org/10.5281/zenodo.19911971) (v0.1.0); badge rendered in [`docs/index.md`](https://github.com/pwr-ai/JuDDGES/blob/master/docs/index.md).
4. **Copy-pasteable BibTeX** (below).

**Software-author roster** (eleven authors across five institutions, reflecting the expanded research collaboration behind the codebase): Łukasz Augustyniak, Jakub Binkowski, Albert Sawczyn, Tomasz Kajdanowicz (Wrocław University of Science and Technology); Michał Bernaczyk (University of Wrocław); Krzysztof Kamiński (Court of Appeal, Wrocław); Santosh Tirunagari, David Windridge, Mandeep K. Dhami (Middlesex University); Chérifa Boukacem-Zeghmouri, Candice Fillaud (Université Claude Bernard Lyon 1).

**Reference paper:** _"Bridging AI and Law: A Scalable Multi-Agent Platform for Quantitative Legal Analytics Across Millions of Documents"_ (Augustyniak et al., 2026), Bridge between AI and Law workshop, pp. 207–214. <https://openreview.net/forum?id=hWjsyTSWrY>. Note the BAIL paper's authorship (eleven WUST authors) is intentionally distinct from the broader software-collaboration roster above — software-author lists evolve after publication.

```bibtex
@inproceedings{augustyniak2026bridging,
  author    = {Lukasz Augustyniak and Jakub Binkowski and Albert Sawczyn and Kamil Tagowski and Denis Janiak and Mateusz Bystroński and Grzegorz Piotrowski and Michal Bernaczyk and Krzysztof Kamiński and Adrian Szymczak and Tomasz Jan Kajdanowicz},
  booktitle = {Bridge between Artificial Intelligence and Law},
  pages     = {207--214},
  title     = {Bridging {AI} and Law: A Scalable Multi-Agent Platform for Quantitative Legal Analytics Across Millions of Documents},
  url       = {https://openreview.net/forum?id=hWjsyTSWrY},
  year      = {2026}
}
```

---

## 3. Open-science checklist — current status

The verifications above against the repository as of **2026-04-30** identified twelve open-science items beyond the FAIR4RS / EOSC software-sustainability baseline. **9 of 12 are now in place** (including a Zenodo-minted persistent DOI); two require off-repo actions on accounts the project owner controls (Software Heritage, OpenSSF Best Practices); one is intentionally deferred.

| # | Item | Principle | Status | Where it lives |
|---|---|---|---|---|
| 1 | `CITATION.cff` (CFF v1.2.0) | FAIR4RS R1.2 — machine-readable citation metadata | ✅ **Done** | [`CITATION.cff`](https://github.com/pwr-ai/JuDDGES/blob/master/CITATION.cff) |
| 2 | `codemeta.json` (CodeMeta v3.0 JSON-LD) | FAIR4RS R1.2 — cross-platform research-software metadata | ✅ **Done** | [`codemeta.json`](https://github.com/pwr-ai/JuDDGES/blob/master/codemeta.json) |
| 3 | GitHub Release + Zenodo DOI | FAIR4RS F1 — globally unique persistent identifier | ✅ **Done** | Concept DOI [`10.5281/zenodo.19911970`](https://doi.org/10.5281/zenodo.19911970); version DOI [`10.5281/zenodo.19911971`](https://doi.org/10.5281/zenodo.19911971); badge in [`docs/index.md`](https://github.com/pwr-ai/JuDDGES/blob/master/docs/index.md) |
| 4 | `CONTRIBUTING.md` | EOSC sustainability — contributor onboarding | ✅ **Done** | [`CONTRIBUTING.md`](https://github.com/pwr-ai/JuDDGES/blob/master/CONTRIBUTING.md) |
| 5 | `CODE_OF_CONDUCT.md` | EOSC community-health | ⏭ **Intentionally deferred** | n/a — to be revisited by project owner |
| 6 | `SECURITY.md` | EOSC sustainability — coordinated vulnerability disclosure | ✅ **Done** | [`SECURITY.md`](https://github.com/pwr-ai/JuDDGES/blob/master/SECURITY.md) |
| 7 | `.github/ISSUE_TEMPLATE/` (4 files) | EOSC sustainability — triage hygiene | ✅ **Done** | [`bug_report.yml`](https://github.com/pwr-ai/JuDDGES/blob/master/.github/ISSUE_TEMPLATE/bug_report.yml), [`feature_request.yml`](https://github.com/pwr-ai/JuDDGES/blob/master/.github/ISSUE_TEMPLATE/feature_request.yml), [`documentation.yml`](https://github.com/pwr-ai/JuDDGES/blob/master/.github/ISSUE_TEMPLATE/documentation.yml), [`config.yml`](https://github.com/pwr-ai/JuDDGES/blob/master/.github/ISSUE_TEMPLATE/config.yml) |
| 8 | `.github/PULL_REQUEST_TEMPLATE.md` | EOSC sustainability — PR review hygiene | ✅ **Done** | [`PULL_REQUEST_TEMPLATE.md`](https://github.com/pwr-ai/JuDDGES/blob/master/.github/PULL_REQUEST_TEMPLATE.md) |
| 9 | Software Heritage SWHID | FAIR4RS F1 — archival permanence beyond GitHub | ⚠ **Pending — external action** (see below) | n/a |
| 10 | `en_appealcourt.xml` Label Studio form | Reproducibility — symmetry with the other reference annotation tasks | ✅ **Done** | [`label_studio_toolkit/form_templates/en_appealcourt.xml`](https://github.com/pwr-ai/JuDDGES/blob/master/label_studio_toolkit/form_templates/en_appealcourt.xml) |
| 11 | `docs/index.md` landing page | Documentation accessibility — clears `mkdocs build --strict` warning | ✅ **Done** | [`docs/index.md`](https://github.com/pwr-ai/JuDDGES/blob/master/docs/index.md) |
| 12 | OpenSSF Best Practices badge / Scorecard | Third-party software-sustainability indicator | ⚠ **Pending — external action** (depends on #3) | n/a |

### Pending external actions

Two items can only be closed off-repo on accounts the project owner controls. Neither is a blocker; both materially improve archival permanence and third-party signalling.

1. **Software Heritage SWHID (#9).** Steps: (a) open <https://archive.softwareheritage.org/save/>, (b) submit `https://github.com/pwr-ai/JuDDGES`, (c) wait for the archival job to complete (typically minutes), (d) copy the resulting `swh:1:dir:…` SWHID and add it to [`CITATION.cff`](https://github.com/pwr-ai/JuDDGES/blob/master/CITATION.cff) under the existing `identifiers:` block (with `type: swh`), and to [`codemeta.json`](https://github.com/pwr-ai/JuDDGES/blob/master/codemeta.json) as an additional `identifier`. Effort: ≈ 10 minutes.
2. **OpenSSF Best Practices badge (#12).** Steps: (a) sign in at <https://www.bestpractices.dev/> with GitHub OAuth, (b) register `pwr-ai/JuDDGES`, (c) work through the criteria checklist (most items already pass), (d) embed the resulting _passing_ / _silver_ / _gold_ badge in [`docs/index.md`](https://github.com/pwr-ai/JuDDGES/blob/master/docs/index.md). Optionally enable the [OpenSSF Scorecard GitHub Action](https://github.com/ossf/scorecard-action) for an automated weekly score. Effort: ≈ 30–45 minutes.

### Recently completed

- **Zenodo DOI (#3)** — completed 2026-04-30. GitHub _Release_ + Zenodo integration was wired up; the deposition exposes a concept DOI [`10.5281/zenodo.19911970`](https://doi.org/10.5281/zenodo.19911970) (always-latest) and a version DOI [`10.5281/zenodo.19911971`](https://doi.org/10.5281/zenodo.19911971) for the v0.1.0 snapshot. Both are recorded in [`CITATION.cff`](https://github.com/pwr-ai/JuDDGES/blob/master/CITATION.cff) (top-level `doi:` field + `identifiers:` block) and in [`codemeta.json`](https://github.com/pwr-ai/JuDDGES/blob/master/codemeta.json) (`identifier` array); the Zenodo DOI badge is rendered in [`docs/index.md`](https://github.com/pwr-ai/JuDDGES/blob/master/docs/index.md).

Total remaining effort once external accounts are available: **≈ 45 minutes**.
