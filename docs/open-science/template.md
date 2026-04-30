# Open-Science Software-Sustainability Report — Template

> **How to use this template**
>
> 1. Copy this file into your project as `docs/open-science/index.md` (or `OPEN_SCIENCE.md` at the repository root).
> 2. Replace every `[PLACEHOLDER]` and every _italic filling-hint_ with your project-specific answer.
> 3. Refresh the **Last verified** date in the header on each revision.
> 4. Use full GitHub deep links (e.g. `https://github.com/<org>/<repo>/blob/<branch>/path/to/file`) on every file/directory you reference — the report should be readable as a self-contained PDF without any cross-references to a `README.md`.
> 5. Once filled, the document doubles as your submission for FAIR4RS (Findable / Accessible / Interoperable / Reusable for Research Software) and EOSC software-sustainability evaluations.
>
> A worked example for a real research codebase is at <https://github.com/pwr-ai/JuDDGES/blob/master/docs/open-science/index.md>.
>
> **Frameworks referenced in this template**
>
> - [FAIR4RS Principles](https://www.rd-alliance.org/group/fair-research-software-fair4rs-wg/outcomes/fair-principles-research-software-fair4rs)
> - [EOSC Software Sustainability](https://www.eosc.eu/) evaluation criteria
> - [Diátaxis](https://diataxis.fr/) documentation framework (Tutorials / How-To / Reference / Explanation)
> - [Citation File Format (CFF) v1.2.0](https://citation-file-format.github.io/)
> - [CodeMeta v3.0](https://codemeta.github.io/) JSON-LD schema
> - [Contributor Covenant](https://www.contributor-covenant.org/) Code of Conduct
> - [OpenSSF Best Practices](https://www.bestpractices.dev/) and [OpenSSF Scorecard](https://github.com/ossf/scorecard)
> - [Software Heritage](https://www.softwareheritage.org/) archival service

---

<!-- markdownlint-disable-next-line MD025 -->
# [PROJECT NAME] — [one-line subtitle, e.g. "data gathering, model training and annotation (PL)"]

Software-sustainability and reproducibility report for the **[PROJECT NAME]** codebase, structured as direct answers to standard open-science evaluation criteria (FAIR4RS / EOSC software-sustainability).

Repository: <[REPOSITORY URL]>
Last verified against the repository: **[YYYY-MM-DD]**.

## 1. Project context and purpose

| Aspect | Description |
|---|---|
| **Briefly describe the context and purpose** | _2–4 paragraphs answering: What does the codebase do? Which problem does it solve? Who is the intended audience? What are the main capabilities? Number the capabilities `(1)`, `(2)`, … and deep-link the relevant subdirectories. Why was the repository created? Who can reuse it and for what? Avoid pointing the reader to `README.md` — the answer must be self-contained._ <br/><br/>**Example skeleton:** "The **[PROJECT NAME]** repository provides an end-to-end codebase for [DOMAIN]. It covers [N] tightly integrated capabilities. (1) **[CAPABILITY 1]** implemented under [`path/to/dir/`](URL). (2) **[CAPABILITY 2]** … (N) **[CAPABILITY N]** … The repository was created so [PROJECT/INSTITUTION] can [PURPOSE]. It can be reused by anyone wishing to [USE-CASE A], [USE-CASE B], or [USE-CASE C]." |

## 2. Evaluation criteria

| Criterion | Sub-question | Answer |
|---|---|---|
| **DOCUMENTATION — License & accessibility** | How can the repository be accessed by third parties? | _State the public URL. List all applicable licenses with deep links to each license file. If you ship code + documentation + data + model weights separately, declare each license separately (e.g. **[Apache-2.0](LICENSE)** for code, **[CC BY 4.0](LICENSE-docs)** for documentation, dataset license declared in Hugging Face dataset card YAML, model license declared in Hugging Face model card YAML). Mention registration requirements (ideally: none)._ |
| | What type of documentation is available, provided with the [project type] and delivered under the same conditions? | _List documentation layers: (a) top-level project overview + installation + quick-start in the repository root; (b) organised `docs/` directory — recommend the **Diátaxis** four-folder structure (`tutorials/`, `how-to/`, `reference/`, `explanation/`); (c) static documentation site config (e.g. [`mkdocs.yml`](URL), [`docs.yaml`](URL)) plus the CI workflow that builds and deploys it; (d) executable notebooks for exploratory analyses; (e) sub-toolkit-specific docs if applicable; (f) configurations (Hydra/argparse) treated as living reference. Deep-link each._ |
| | Does the documentation describe how to use/build/deploy/install the [project type]? | _Cover: (a) installation — automated path (e.g. [`setup.sh`](URL)) and manual path (e.g. `uv venv`, `pip install -e .`); (b) build/lint/test targets exposed via [`Makefile`](URL) / `just` / scripts; (c) pipeline execution if any (DVC / Snakemake / Nextflow stage commands); (d) external services run via Docker Compose ([`docker-compose.yml`](URL)). If you do not ship a web application, state explicitly: "No web application is shipped — [PROJECT] is a [research codebase / library / CLI tool] composed of …"._ |
| **TESTING — Sample data & parameters** | Are sample data and/or parameters that can be used to test the [project type] available with the source code? | _List four categories. (a) **Automated tests** — location ([`tests/`](URL)), runner (`pytest`, `cargo test`, etc.), aggregate target (`make test`, `make all`), CI workflow that runs them. List sub-suites if relevant. (b) **Sample data** committed to the repository — paths and a short description of each artefact; mention whether sample data is also tracked through DVC `.dvc` pointer files. (c) **Use-case example scripts** under `scripts/` or `examples/`. (d) **Reference configurations / runnable examples** that exercise the full pipeline on a small subset._ |
| **INTEROPERABILITY — Standard I/O formats** | Do you use existing and standard input/output formats? | _List the formats: **Parquet**, **CSV**, **JSON**, **YAML**, **HDF5**, **NetCDF**, **TFRecord**, etc. — whatever applies. Note distribution channels (Hugging Face `datasets`, Zenodo, Figshare). Note standard APIs consumed (e.g. **OpenAI-compatible REST**, gRPC, GraphQL). Note dependency-declaration formats ([`requirements.txt`](URL), [`pyproject.toml`](URL), [`Cargo.toml`](URL), [`package.json`](URL), [`environment.yml`](URL)). Note any vector DB / object store and the schema used. Avoid bespoke formats unless you also publish their spec._ |
| **VERSIONING — Source-code version control** | Do you use a version control system? | _State Git/Mercurial + hosting platform URL. Mention pre-commit hooks ([`.pre-commit-config.yaml`](URL)) and the branch + pull-request review workflow. List each CI workflow under [`.github/workflows/`](URL) by purpose (test/quality, docs build/deploy, docs PR preview, docs quality checks, security scan, etc.)._ |
| **REPRODUCIBILITY — Releases** | Do you provide releases of your software? | _Cover: (a) **Versioned Git tags / GitHub Releases / Zenodo DOI** — link the canonical reproducibility tag if you have one (e.g. `neurips_v0.1`). (b) **Pipeline tracking** with DVC / Snakemake / Nextflow / similar — link [`dvc.yaml`](URL) and [`dvc.lock`](URL) (or equivalents); call out matrix expansion if you use it. (c) **Sample data version-tracking** through DVC `.dvc` pointer files. The lockfile records exact inputs/parameters/hashes/outputs so any reported artefact can be reproduced end-to-end with `dvc repro <stage>`._ |
| | How do you define language-specific dependencies of your [project type] and their version? | _List dependency files at the language level: [`requirements.txt`](URL), [`pyproject.toml`](URL) + [`uv.lock`](URL) for Python, [`Cargo.toml`](URL) + [`Cargo.lock`](URL) for Rust, [`package.json`](URL) + lockfile for Node, [`environment.yml`](URL) for Conda, etc. State the recommended package manager and any pinned system-level dependency (CUDA, glibc, etc.). State that external services are pinned via Docker Compose files for reproducibility without polluting the host system._ |
| **BUG REPORTING (software users)** | Do you state how to report bugs and/or usability problems by the software user(s)? | _Direct users to [`<repository URL>/issues`](URL) and explain the standard tooling used (labels, milestones, GitHub Discussions). Reference [`CONTRIBUTING.md`](URL) for contribution conventions and [`SECURITY.md`](URL) for coordinated vulnerability disclosure._ |
| **BUG REPORTING (web-app users)** | Do you state how to report bugs and/or usability problems by the web app user(s)? | _If you ship a hosted web application: in-app feedback form, status page, support email._ <br/><br/>_If not: state "**Not applicable** — [PROJECT] does not ship a hosted web application; it is a [research codebase / CLI tool / library] that runs locally or on a user-controlled compute environment."_ |
| **RECOGNITION — Citation information** | Do you include citation information (i.e. how to cite your software in the form of citation.cff, codemeta.json or bibtex)? | _Provide all three canonical formats with deep links: (a) [`CITATION.cff`](URL) — Citation File Format v1.2.0, consumed by GitHub's "Cite this repository" widget, Zenodo, Zotero, Mendeley, OpenAIRE; (b) [`codemeta.json`](URL) — CodeMeta v3.0 JSON-LD, consumed by HAL, OpenAIRE, Software Heritage, re3data; (c) a copy-pasteable BibTeX block embedded in this report. Cite the reference paper if any (title, venue, authors, URL). State persistent DOI status (minted via Zenodo, or "pending" with a forward-pointer to § 3)._ |

## 3. Open-science checklist — current status

The verifications above identify **twelve** open-science items beyond the FAIR4RS / EOSC software-sustainability baseline. Track each item's status as **✅ Done** (file present) / **⚠ Pending external action** (requires off-repo account) / **⏭ Intentionally deferred** / **❌ Not done** below.

| # | Item | Principle | Status | Where it lives |
|---|---|---|---|---|
| 1 | `CITATION.cff` (CFF v1.2.0) | FAIR4RS R1.2 — machine-readable citation metadata | [STATUS] | [`CITATION.cff`](URL) or n/a |
| 2 | `codemeta.json` (CodeMeta v3.0 JSON-LD) | FAIR4RS R1.2 — cross-platform research-software metadata | [STATUS] | [`codemeta.json`](URL) or n/a |
| 3 | GitHub Release + Zenodo DOI | FAIR4RS F1 — globally unique persistent identifier | [STATUS] | n/a — see external-action checklist |
| 4 | `CONTRIBUTING.md` | EOSC sustainability — contributor onboarding | [STATUS] | [`CONTRIBUTING.md`](URL) or n/a |
| 5 | `CODE_OF_CONDUCT.md` | EOSC community-health | [STATUS] | [`CODE_OF_CONDUCT.md`](URL) or n/a |
| 6 | `SECURITY.md` | EOSC sustainability — coordinated vulnerability disclosure | [STATUS] | [`SECURITY.md`](URL) or n/a |
| 7 | `.github/ISSUE_TEMPLATE/` (`bug_report.yml`, `feature_request.yml`, `documentation.yml`, `config.yml`) | EOSC sustainability — triage hygiene | [STATUS] | [`.github/ISSUE_TEMPLATE/`](URL) or n/a |
| 8 | `.github/PULL_REQUEST_TEMPLATE.md` | EOSC sustainability — PR review hygiene | [STATUS] | [`.github/PULL_REQUEST_TEMPLATE.md`](URL) or n/a |
| 9 | Software Heritage SWHID | FAIR4RS F1 — archival permanence beyond the host platform | [STATUS] | n/a — see external-action checklist |
| 10 | Domain-specific runnable examples (each example ships matching schema **and** form/config) | Reproducibility — symmetric reference assets | [STATUS] | [example dir URL] or n/a |
| 11 | `docs/index.md` landing page (eliminates `mkdocs build --strict` warnings) | Documentation accessibility | [STATUS] | [`docs/index.md`](URL) or n/a |
| 12 | OpenSSF Best Practices badge / OpenSSF Scorecard | Third-party software-sustainability indicator | [STATUS] | n/a — see external-action checklist |

### Pending external actions

For items requiring off-repo actions on accounts the project owner controls:

1. **Zenodo DOI (#3).** A Git tag is not a citable artefact; a GitHub _Release_ + Zenodo integration mints a persistent DOI for every published version. Steps: (a) sign in to <https://zenodo.org/> with GitHub OAuth, (b) on <https://zenodo.org/account/settings/github/> flip the `<org>/<repo>` switch to **On**, (c) on GitHub, promote your reproducibility tag to a _Release_ (Releases → Draft a new release → choose tag → write release notes → Publish), (d) cut a fresh `vX.Y.Z` _Release_ for the current state, (e) once Zenodo mints the DOI, populate the `doi:` field in `CITATION.cff` and the `identifier` field in `codemeta.json`, and add the DOI badge to `docs/index.md`. **Effort: ≈ 30 min.**
2. **Software Heritage SWHID (#9).** Software Heritage provides a permanent SWHID for every commit, independent of the host platform. Steps: (a) open <https://archive.softwareheritage.org/save/>, (b) submit your repository URL, (c) wait for the archival job to complete, (d) copy the resulting `swh:1:dir:…` SWHID and add it to `CITATION.cff` (under `identifiers:` of `type: swh`) and to `codemeta.json` (as an additional `identifier` entry). **Effort: ≈ 10 min.**
3. **OpenSSF Best Practices badge (#12).** Widely used as a third-party indicator of software-sustainability hygiene. Steps: (a) sign in to <https://www.bestpractices.dev/> with GitHub OAuth, (b) register the project, (c) work through the criteria checklist (most items already pass once #1, #4, #6, #7, #8 land), (d) embed the resulting _passing_ / _silver_ / _gold_ badge in `docs/index.md`. Optionally enable the [OpenSSF Scorecard GitHub Action](https://github.com/ossf/scorecard-action) for an automated weekly score. **Effort: ≈ 30–45 min.**

### Items already met (no action required)

_Once filled, list everything verified as already in place — license declarations, public hosting, Diátaxis docs, CI pipelines, pytest suite, interoperable I/O formats, Git + pre-commit, DVC tracking, dependency lockfiles, Docker Compose pinning, GitHub Issues tracker, BibTeX citation, etc. This list also doubles as the project's open-science evidence summary for funder reports._

---

## Appendix A — Filling guidance

### Tone and style

- Be **self-contained** — do not refer the reader to `README.md` or any other repository file for context. The answer in each cell must be readable on its own.
- Use **British or American English** consistently throughout.
- Use **full GitHub deep links** (`/blob/<branch>/` for files, `/tree/<branch>/` for directories) on every file/directory referenced. Avoid relative paths — they break when the file is rendered outside the repository.
- Refresh the **Last verified** date on every revision.
- Keep each cell's answer **focused on what is actually delivered** — if you do not ship a web application, do not invent web-application content; state "Not applicable" and explain.

### Common gotchas

| Pitfall | Better approach |
|---|---|
| Single license declaration when the project ships code + docs + data + model weights | Declare each license separately, with deep links to each license file (and dataset/model card YAML metadata). |
| Pointing at `README.md` for installation instructions | Inline the installation command(s) directly in the cell, with a deep link to the manifest (`pyproject.toml`, `Cargo.toml`, etc.). |
| Listing `pytest` without naming any sub-suite | Enumerate sub-suites with deep links — preprocessing tests, integration tests, etc. — so reviewers can see test breadth at a glance. |
| Treating a Git tag as a release | A Git tag is not a citable artefact. Promote it to a GitHub _Release_ and wire the Zenodo–GitHub webhook to mint a DOI. |
| Listing `BibTeX only` under "Citation information" | Ship the **citation triple**: BibTeX (in this report) + `CITATION.cff` (CFF v1.2.0) + `codemeta.json` (CodeMeta v3.0 JSON-LD). |
| Empty `.github/ISSUE_TEMPLATE/` directory | Ship four files at minimum: `bug_report.yml`, `feature_request.yml`, `documentation.yml`, `config.yml` (the last disables blank issues and links to Discussions / Security Advisories). |
| Asymmetric example assets (some examples have config, others don't) | Make examples symmetric — each example ships matching schema **and** form/config, so they all work as runnable references. |

### When to update this report

- Every release that changes license, dependency manifest, CI workflow, or external-service Compose file.
- Every six months as a freshness check (refresh the **Last verified** date, re-run the checks).
- Whenever an external action lands (Zenodo DOI minted, SWHID assigned, OpenSSF badge granted) — update the corresponding row in the § 3 checklist and add the badge to `docs/index.md`.

## Appendix B — Suggested rollout order

If you are starting from scratch, the twelve items in § 3 are loosely ordered for maximum payoff per minute spent:

1. **Day 1 (≈ 2 hours, items #1, #2, #5, #6, #11)** — pure-content additions with no infrastructure dependencies: `CITATION.cff`, `codemeta.json`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, `docs/index.md`.
2. **Day 2 (≈ 1.5 hours, items #4, #7, #8)** — contributor onboarding: `CONTRIBUTING.md`, four issue templates, PR template.
3. **Day 3 (≈ 1 hour, items #3, #9)** — wire the Zenodo–GitHub integration, cut the first GitHub _Release_, archive on Software Heritage. After this, a citable DOI and a permanent SWHID exist for the project.
4. **Day 4 (≈ 1 hour, items #10, #12)** — symmetrise the example assets, then submit the OpenSSF Best Practices self-assessment.

Total estimated effort: **≈ 5–6 hours of one engineer's time.**
