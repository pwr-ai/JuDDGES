# Contributing to JuDDGES

Thank you for your interest in contributing to JuDDGES, a research codebase for Polish legal-document analysis. We welcome contributions of all kinds, from bug reports and documentation improvements to new annotation tasks and model integrations.

## Code of Conduct

This project adheres to the Contributor Covenant v2.1. By participating, you agree to uphold its terms; please read [./CODE_OF_CONDUCT.md](./CODE_OF_CONDUCT.md) before engaging with the community.

## Where to start

- Browse open issues, especially those labelled `good first issue` and `help wanted`.
- For broader questions, use GitHub Discussions.
- For project goals and FAIR/open-science context, see the Open Science docs at `docs/open-science/`.

## Development environment setup

Python 3.10+ is required. CUDA 12.4 is required for any GPU-backed work (embeddings, fine-tuning, inference).

Recommended path using `uv`:

```bash
git clone https://github.com/pwr-ai/JuDDGES.git
cd JuDDGES
uv venv .venv
source .venv/bin/activate
uv pip install -e .
```

Automated path:

```bash
./setup.sh        # Linux/macOS
setup.bat         # Windows
```

Fine-tuning extras (creates a dedicated conda environment):

```bash
make install_unsloth
```

Install the pre-commit hooks once after cloning:

```bash
pre-commit install
```

## Running the test and quality suite

- `make fix` — auto-format and auto-fix lint.
- `make check` — verify formatting and linting (read-only).
- `make check-types` — run mypy on `juddges/`, `scripts/`, `tests/`.
- `make test` — pytest with coverage.
- `make all` — full sweep; run this before opening a pull request.

## Where contributions live

- Library code → `juddges/`
- CLI scripts and DVC stage drivers → `scripts/`
- Hydra configurations → `configs/`
- Tests → `tests/`
- DVC pipeline definition → `dvc.yaml`
- HITL annotation toolkit → `label_studio_toolkit/`
- Documentation (Diataxis-aligned: `tutorials/`, `how-to/`, `reference/`, `explanation/`, `open-science/`) → `docs/`, built with MkDocs (`mkdocs.yml`)

## Pull request process

1. Fork the repository and create a feature branch off `master` with a descriptive name (e.g. `feat/swiss-franc-extraction`).
2. Make focused commits — one logical change per commit; reference the issue number where applicable.
3. Run `make all` and confirm everything passes locally.
4. If you change Python APIs or CLI behaviour, update the matching documentation under `docs/` (Diataxis-aligned).
5. If you change the DVC pipeline, run `dvc repro <stage>` for the affected stage and commit the regenerated `dvc.lock`.
6. If you add an annotation task to `label_studio_toolkit/`, update `label_studio_toolkit/docs/add-new-task.md` and add the corresponding Hydra configs under `configs/`.
7. If you change the author list or release a new version, update `CITATION.cff` AND `codemeta.json` together so they remain consistent.
8. Open a pull request, fill in the PR template, link the issue, and request review.

## Adding a new annotation task

The HITL annotation toolkit is designed to be extended. See [`label_studio_toolkit/docs/add-new-task.md`](./label_studio_toolkit/docs/add-new-task.md) for the full extensibility guide. In short, each new task needs (a) a Pydantic schema in `label_studio_toolkit/schemas/`, (b) a Label Studio form template in `label_studio_toolkit/form_templates/`, and (c) a Hydra preannotation config in `configs/`.

## Reporting a vulnerability

Please do NOT open public issues for security matters; follow the responsible-disclosure process described in [./SECURITY.md](./SECURITY.md).

## License of contributions

By contributing to JuDDGES, you agree that your contributions will be licensed under the Apache License 2.0, matching the project.
