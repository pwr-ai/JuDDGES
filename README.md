# JuDDGES

![python-3.11](https://img.shields.io/badge/Python-3.11-blue)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://pwr-ai.github.io/JuDDGES/)
[![Documentation Build](https://github.com/pwr-ai/JuDDGES/actions/workflows/docs-build-deploy.yaml/badge.svg)](https://github.com/pwr-ai/JuDDGES/actions/workflows/docs-build-deploy.yaml)
[![Documentation Quality](https://github.com/pwr-ai/JuDDGES/actions/workflows/docs-quality-checks.yaml/badge.svg)](https://github.com/pwr-ai/JuDDGES/actions/workflows/docs-quality-checks.yaml)

> JuDDGES stands for Judicial Decision Data Gathering, Encoding, and Sharing.

JuDDGES makes judicial decisions across multiple legal systems easier to gather, structure, and analyze at scale. The project combines modern Natural Language Processing with Human-In-The-Loop machine learning to turn unstructured court documents into research-ready data, with an initial focus on criminal court records from Poland and England & Wales.

By tackling the language, format, and access barriers that have historically held back empirical legal research, JuDDGES gives researchers, public institutions, and policymakers a shared foundation for studying judicial decision-making across jurisdictions. All software, datasets, and models are released openly so the community can extend them, reproduce results, and build the most comprehensive legal research repository in Europe.

![baner](https://raw.githubusercontent.com/pwr-ai/JuDDGES/bffb1d75ba7c78f101fc94bd9086499886b2c128/nbs/images/baner.png)

## What you can do with JuDDGES

- **Build legal datasets** from raw court records (Polish and English) using the ingestion and preprocessing pipelines.
- **Run information extraction** with open LLMs (Bielik, Llama, Mistral, Phi) to pull structured fields out of judgments.
- **Fine-tune and evaluate** models on legal extraction tasks using DVC-tracked, reproducible pipelines.
- **Search semantically** across legal documents through Weaviate-backed vector indexes built on multilingual legal embeddings.
- **Reuse open data and models** published on the [JuDDGES Hugging Face organization](https://huggingface.co/JuDDGES).

## Installation

The project requires **Python 3.11**. Pick whichever installation path fits your workflow.

### Option 1: UV (recommended)

[UV](https://github.com/astral-sh/uv) is a fast Python package installer and resolver.

**Automated setup:**

- Linux/macOS:

  ```bash
  chmod +x setup.sh
  ./setup.sh
  ```

- Windows:

  ```
  setup.bat
  ```

**Manual setup:**

```bash
# Install UV if you don't have it yet
pip install uv

# Create and activate a virtual environment
uv venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate            # Windows

# Install the project in editable mode
uv pip install -e .
```

### Option 2: Make (legacy)

- Install dependencies (Python 3.11+): `make install`
- Install Unsloth for fine-tuning and evaluation (run inside a conda environment): `make install_unsloth`

## Usage

### Dataset creation

Step-by-step instructions for assembling datasets from raw court records live in [scripts/README.md](scripts/README.md).

### Inference, fine-tuning, and evaluation

Inference, fine-tuning, and evaluation are all defined as stages in [`dvc.yaml`](dvc.yaml) (see the [DVC user guide](https://dvc.org/doc/user-guide) for background). Many stages are configured as a matrix and run for combinations of parameters such as models and random seeds. Most scripts are configured with [Hydra](https://github.com/facebookresearch/hydra); a few simpler ones (for example n-gram evaluation) take command-line arguments instead. The sections below show how to reproduce each stage and where to find its configuration.

> [!NOTE]
> The commands below assume all dependencies are installed and a GPU with at least 40 GB of VRAM is available.

> [!TIP]
> To add a new model from the Hugging Face Hub or from a local checkpoint/adapter, drop a new config into `configs/model/`.

> [!TIP]
> To run a single combination from the DVC matrix, call the stage by its full name, for example `predict@Bielik-7B-Instruct-v0.1-42`. Use `dvc stage list <stage_name>` to see what is available.

#### Inference

- Configuration: `configs/predict.yaml`
- Environment variables:
  - `CUDA_VISIBLE_DEVICES` — GPU device ID
  - `NUM_PROC` — parallel worker count
- Command:

  ```bash
  CUDA_VISIBLE_DEVICES=0 NUM_PROC=10 dvc repro predict
  ```

- Output: structured information extracted by the LLM.

#### Fine-tuning

- Configuration: `configs/fine_tuning.yaml`
- Environment variables:
  - `CUDA_VISIBLE_DEVICES` — GPU device ID
  - `NUM_PROC` — parallel worker count
- Command:

  ```bash
  CUDA_VISIBLE_DEVICES=0 NUM_PROC=10 dvc repro sft
  ```

- Output: trained LLM adapter.

#### Evaluation

1. **N-gram-based evaluation**
   - Configuration: command-line arguments (no separate config file)
   - Command:

     ```bash
     dvc repro evaluate
     ```

   - Input: LLM-extracted information (see [Inference](#inference)).
   - Output: metrics.

2. **LLM-as-judge evaluation**
   - Configuration: `configs/llm_judge.yaml`
   - Command:

     ```bash
     dvc repro evaluate_llm_as_judge
     ```

   - Input: LLM-extracted information (see [Inference](#inference)).
   - Output: metrics.

## Documentation

Full documentation is published at [https://pwr-ai.github.io/JuDDGES/](https://pwr-ai.github.io/JuDDGES/) and is organized following the [Diátaxis framework](https://diataxis.fr/):

- **Getting Started** — installation and quickstart guides
- **Tutorials** — step-by-step walkthroughs for common workflows
- **How-To Guides** — focused recipes for specific tasks
- **API Reference** — module-level reference generated from the codebase
- **Explanation** — architectural overviews and research context

### Contributing to documentation

To contribute, see [Contributing to Documentation](docs/how-to/documentation/CONTRIBUTING_TO_DOCS.md) for detailed instructions on:

- setting up a local documentation environment,
- writing and formatting content,
- running quality checks locally, and
- understanding the CI/CD pipeline.

The documentation site is built with MkDocs Material and deployed automatically to GitHub Pages whenever changes land on `main`.

## Project details

JuDDGES is organized into four Work Packages (WPs) covering everything from project management to open-science engagement with early-career researchers.

### WP1: Project Management (24 months)

Ensures the project completes on time and on budget. Covers administrative, scientific, and technological management; quality, innovation, and risk management; ethical and legal considerations; and open-science facilitation.

### WP2: Gathering and Human Encoding of Judicial Decision Data (22 months)

Builds the data foundation. Activities include collecting legal case records and judgments, developing the coding scheme, training human coders, supplying human-coded data to WP3, supporting human-in-the-loop coding, and preparing data for open release through WP4.

### WP3: NLP and HITL Machine Learning Methodological Development (24 months)

Bridges machine learning (led by WUST and MUHEC) with open-science facilitation (ELICO). Develops baseline information extraction, intelligent inference methods for legal corpora, and an active-learning annotation tool driven by human-in-the-loop methods.

### WP4: Open Science Practices and Engaging Early-Career Researchers (12 months)

Implements the call's open-science policy and engages early-career researchers (ECRs). Provides open access to publication data and software, disseminates and exploits project results, and promotes the project and its findings.

Each WP defines specific tasks pursued collaboratively across project partners in service of the wider JuDDGES mission.

## Acknowledgements

JuDDGES is a collaboration between:

1. Wroclaw University of Science and Technology (Poland)
2. Middlesex University London (UK)
3. University of Lyon 1 (France)

## License

This project uses different licenses for different artifacts to align with open-science best practices:

| Artifact | License |
|---|---|
| Source code (this repository) | [Apache License 2.0](LICENSE) |
| Documentation (this repository, including the mkdocs site) | [Creative Commons Attribution 4.0 International (CC BY 4.0)](LICENSE-docs) |
| Datasets published on [Hugging Face Hub](https://huggingface.co/JuDDGES) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| Fine-tuned models published on [Hugging Face Hub](https://huggingface.co/JuDDGES) | [OpenRAIL-M](https://huggingface.co/blog/open_rail) |

Dataset and model licenses are declared in the YAML metadata of their respective Hugging Face Hub cards.

## Citation

If you find this research useful, please cite our paper:

```bibtex
@inproceedings{augustyniak2026bridging,
  author = {Lukasz Augustyniak and Jakub Binkowski and Albert Sawczyn and Kamil Tagowski and Denis Janiak and Mateusz Bystro{\'n}ski and Grzegorz Piotrowski and Michal Bernaczyk and Krzysztof Kami{\'n}ski and Adrian Szymczak and Tomasz Jan Kajdanowicz},
  booktitle = {Bridge between Artificial Intelligence and Law},
  pages = {207--214},
  title = {Bridging {AI} and Law: A Scalable Multi-Agent Platform for Quantitative Legal Analytics Across Millions of Documents},
  url = {https://openreview.net/forum?id=hWjsyTSWrY},
  year = {2026}
}
```
