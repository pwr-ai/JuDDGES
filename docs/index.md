# JuDDGES Documentation

End-to-end codebase for acquiring, embedding, fine-tuning, annotating, and evaluating Polish legal judgments with Large Language Models.

<div>
  <a href="https://doi.org/10.5281/zenodo.19911970"><img src="https://zenodo.org/badge/761820962.svg" alt="Zenodo DOI"></a>
  <img src="https://img.shields.io/badge/code_license-Apache_2.0-blue.svg" alt="Code license: Apache 2.0">
  <img src="https://img.shields.io/badge/docs_license-CC_BY_4.0-lightgrey.svg" alt="Docs license: CC BY 4.0">
  <a href="https://huggingface.co/JuDDGES"><img src="https://img.shields.io/badge/🤗-JuDDGES-yellow.svg" alt="HuggingFace organisation"></a>
  <a href="https://github.com/pwr-ai/JuDDGES"><img src="https://img.shields.io/badge/GitHub-pwr--ai/JuDDGES-181717.svg?logo=github" alt="GitHub repository"></a>
</div>

## What JuDDGES does

JuDDGES is a research codebase that takes Polish legal judgments end-to-end: it acquires raw documents from the Polish Common Courts API and the National Administrative Court (NSA), generates multilingual legal embeddings with `sdadas/mmlw-roberta-large` for storage in Weaviate, and runs schema-driven information extraction with Pydantic schemas on top of fine-tuned Large Language Models (Llama 3.1/3.2, Mistral, Bielik for Polish, Phi-4) trained via PEFT/LoRA with Unsloth. A Human-in-the-Loop annotation toolkit on Label Studio supports iterative dataset curation, and the entire pipeline — preprocessing, embedding, supervised fine-tuning, prediction, and evaluation — is reproducibly tracked with DVC. All datasets, code, and trained models are openly published: see the [GitHub repository](https://github.com/pwr-ai/JuDDGES) and the [Hugging Face organisation](https://huggingface.co/JuDDGES).

## Quick start

Clone the repository and install the package into a fresh virtual environment.

```bash
git clone https://github.com/pwr-ai/JuDDGES.git
cd JuDDGES
uv venv .venv && source .venv/bin/activate
uv pip install -e .
```

Run the full quality + test sweep with `make all`.

## Documentation map

| Section | Use this when |
| --- | --- |
| [Tutorials](tutorials/index.md) | learning JuDDGES from scratch |
| [How-to guides](how-to/index.md) | accomplishing a specific task (ingesting a dataset, running fine-tuning, exporting annotations) |
| [API reference](reference/api/index.md) | looking up a specific function, class, or configuration field |
| [Explanation](explanation/index.md) | understanding the architecture, data-flow, or research motivation |
| [Open Science](open-science/index.md) | citation, licensing, reproducibility, FAIR4RS compliance |

## Project structure (one-glance)

- `juddges/` — library code
- `scripts/` — CLI entry points (data ingestion, training, evaluation)
- `configs/` — Hydra configurations (datasets, models, pipelines)
- `dvc.yaml` — pipeline definition
- `label_studio_toolkit/` — HITL annotation toolkit (Pydantic schemas + Label Studio integration)
- `tests/` — pytest suite
- `docs/` — this documentation site

## Citation

If you use JuDDGES in academic work, please cite the BAIL 2026 paper (_Bridging AI and Law: A Scalable Multi-Agent Platform for Quantitative Legal Analytics Across Millions of Documents_). The full BibTeX, `CITATION.cff`, and `codemeta.json` metadata are documented in [Open Science → Recognition](open-science/index.md#2-evaluation-criteria).

## Contributing

Contributions are welcome — see [CONTRIBUTING.md](https://github.com/pwr-ai/JuDDGES/blob/master/CONTRIBUTING.md) for the development workflow. The project follows the [Contributor Covenant v2.1](https://github.com/pwr-ai/JuDDGES/blob/master/CODE_OF_CONDUCT.md). For vulnerability disclosure, see [SECURITY.md](https://github.com/pwr-ai/JuDDGES/blob/master/SECURITY.md).

## License

Code: [Apache 2.0](https://github.com/pwr-ai/JuDDGES/blob/master/LICENSE) — Documentation: [CC BY 4.0](https://github.com/pwr-ai/JuDDGES/blob/master/LICENSE-docs).
