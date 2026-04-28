# nbs/ — Archived exploratory notebooks

The notebooks in this directory are an archive from the nbdev-era exploration phase of the JuDDGES project. They are **preserved intentionally** for reference, future revival, or extraction of useful examples — but they are **not part of the published documentation** at <https://pwr-ai.github.io/JuDDGES/>.

## Why they're parked here

When the project moved from nbdev to MkDocs Material (organized via the [Diátaxis](https://diataxis.fr/) framework) for documentation, the active documentation surface became `docs/`. The notebooks here pre-date that transition and contain a mix of:

- Dataset descriptions and exploratory analyses
- Workshop demos and presentations
- Graph analysis prototypes
- Model and embedding experiments

They were never imported into the `docs/` tree because the published documentation was rewritten from scratch around the new architecture, not migrated.

## What's still actively wired into the codebase

Two notebook-derived artefacts are NOT just parked content — they are referenced by live code and pipelines. If you delete or move either of these, update the corresponding Python and DVC references first:

- `nbs/Dataset Cards/01_Dataset_Description_Raw.ipynb` is loaded by `scripts/dataset/push_raw_dataset.py` and `scripts/dataset/pl_court_data_pipeline.py`, and consumed by the `raw_dataset_readme` stage in `dvc.yaml`.
- `nbs/images/baner.png` is referenced from the project `README.md` and from the Streamlit dashboard at `juddges/dashboards/project_info.py`.

Everything else under `nbs/` is inert from the codebase's point of view.

## How notebooks are excluded from the docs build

Several layers keep `nbs/` out of the published documentation, so contributors don't need to add anything ad hoc:

- `mkdocs.yml` only renders entries listed in its `nav:` block; no `nbs/` entries exist there.
- The `mkdocstrings` plugin generates API reference from `juddges/**/*.py` only; it never sees notebooks.
- `.github/workflows/docs-build-deploy.yaml` and `.github/workflows/docs-quality-checks.yaml` both restrict their `paths:` triggers to `docs/**`, `mkdocs.yml`, and related config — edits under `nbs/` do not start a docs build or quality run.
- The `cspell` and `markdownlint-cli2` pre-commit hooks are scoped to `^docs/.*\.md$` (with `cspell` also covering the root `README.md`), so notebook-adjacent markdown is not lint-checked.

## Reviving a notebook into the docs site

If you want to convert a notebook into a published tutorial or how-to:

1. Pick the appropriate Diátaxis quadrant under `docs/` — `docs/tutorials/` for learning-oriented walkthroughs, `docs/how-to/` for task-focused recipes, `docs/explanation/` for conceptual deep-dives, or `docs/reference/` for specification material.
2. Translate the notebook to Markdown (or keep the `.ipynb` and add `mkdocs-jupyter` to the `mkdocs.yml` plugin list — currently not enabled).
3. Add the new file to the `nav:` block in `mkdocs.yml`.
4. Run `mkdocs build --strict` locally to verify the navigation, links, and any code examples.

## Why we didn't delete them

Open-science research repositories benefit from being able to point at their historical exploration phase: it makes prior results reproducible, lets reviewers and downstream users trace decisions, and keeps references stable in papers, slides, and issue threads. Git history preserves the notebooks either way, but keeping them at a stable filesystem path means external links continue to resolve.
