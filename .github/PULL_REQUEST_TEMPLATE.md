## Summary

<!-- 1–3 sentences on what changed and why. Focus on the *why*, not the *what* — the diff already shows the what. -->

## Linked issue

<!-- "Closes #123" or "Related to #123" — please link the issue this PR addresses. -->

## Type of change

- [ ] Bug fix
- [ ] New feature / enhancement
- [ ] Documentation update
- [ ] DVC pipeline change (`dvc.yaml` / `dvc.lock` updated)
- [ ] Annotation toolkit change (`label_studio_toolkit/`)
- [ ] CI / build change (`.github/workflows/`, `Makefile`, Docker)
- [ ] Refactor (no functional change)
- [ ] Breaking change — describe the migration path under "Test plan"

## Test plan

<!--
Bullet list. Mention the exact commands you ran:
- `make all` — passed locally
- For DVC pipeline changes, which `dvc repro <stage>` did you execute?
- For annotation-toolkit changes, which task config did you exercise?
- For data-acquisition changes, on which sample file under `data/sample_data/` did you verify?
-->

-

## Documentation

- [ ] Updated relevant docs in `docs/` (Diátaxis-aligned: `tutorials/`, `how-to/`, `reference/`, `explanation/`, `open-science/`)
- [ ] Updated annotation-toolkit docs in `label_studio_toolkit/docs/` if applicable
- [ ] Updated `CITATION.cff` AND `codemeta.json` together if author list or version changed
- [ ] N/A — change does not affect documentation

## Pre-merge checklist

- [ ] `make all` passes locally
- [ ] No secrets, credentials, or `.env` files committed
- [ ] New dependencies added to `pyproject.toml` / `requirements.txt` and `uv.lock` regenerated
- [ ] Pre-commit hooks ran without errors
- [ ] PR title is concise and descriptive (under 70 characters)

---

*Reviewers: see [`CONTRIBUTING.md`](../CONTRIBUTING.md) for review conventions.*
