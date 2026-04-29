# Security Policy

## Supported versions

JuDDGES is a research codebase. The only supported branch is `master` (soon to be the default public branch). No long-term support releases are maintained; security fixes land on `master` and are tagged as patch releases.

## Reporting a vulnerability

**Private disclosure (preferred)**: Email `aisolutions@lukaszaugustyniak.com` with the subject line `[SECURITY] JuDDGES — <brief description>`. We aim to acknowledge within 48 hours and provide a fix or mitigation plan within 14 days.

**GitHub Security Advisory**: Once the repository is public, you may also open a private advisory via the repository's **Security** tab (Security > Advisories > New draft security advisory). This is the recommended channel for CVE assignment.

Please do not open a public GitHub issue for security vulnerabilities until a fix has been prepared.

## Licensing posture

- **Source code**: Apache 2.0 (see `LICENSE`)
- **Documentation and dataset cards**: CC BY 4.0

## Known accepted-risk packages (unpatched CVEs)

The packages below carry open CVEs for which **no patched version exists upstream** as of 2026-04-29. They have been reviewed, accepted as tolerable given the mitigating factors listed, and documented here in lieu of an upgrade.

---

### `mlflow` — CVE-2026-0545 (Critical)

| Field | Value |
|---|---|
| CVE | CVE-2026-0545 |
| Severity | Critical |
| Vulnerable range | `<= 3.10.1` (all current releases) |
| Fix version | None available upstream |
| Dependabot alerts | 39 |

**Description**: MLflow's FastAPI job endpoints (`/ajax-api/3.0/jobs/*`) do not enforce authentication, allowing any network-reachable client to trigger arbitrary job execution.

**Why we cannot upgrade**: No patched release exists in the MLflow ecosystem as of the date of this audit. The CVE was filed against the latest stable release; upstream tracking issue is open.

**Mitigating factors**:
- MLflow is used exclusively in **offline fine-tuning pipelines** running on internal HPC infrastructure (PWr e-Science cluster). The MLflow tracking server is never exposed to a public network or the internet.
- CI/CD does not spin up MLflow servers; it only reads logged metrics.
- Contributors running MLflow locally should bind it to `127.0.0.1` only (`mlflow server --host 127.0.0.1`).

**Accepted by**: Lukasz Augustyniak (`aisolutions@lukaszaugustyniak.com`), 2026-04-29

---

### `ray` (transitive) — CVE-2025-34351 (Critical)

| Field | Value |
|---|---|
| CVE | CVE-2025-34351 |
| Severity | Critical |
| Vulnerable range | `<= 2.52.0` (all current releases) |
| Fix version | None available upstream |
| Dependabot alerts | 8 |
| Relationship | Transitive (pulled in by `vllm`) |

**Description**: Ray's new token-based authentication mechanism is disabled by default. Any process that can reach the Ray dashboard port can submit tasks or access the cluster without credentials.

**Why we cannot upgrade**: No patched release exists. The Ray project has acknowledged the issue but has not shipped a default-secure configuration in any stable release. Upgrading `ray` transitively via `vllm` does not resolve the underlying configuration default.

**Mitigating factors**:
- Ray is a transitive dependency introduced by `vllm`, which itself is in the `full` optional-dependency group. It is **not** installed in the base environment.
- Ray clusters are only used during GPU fine-tuning jobs on an air-gapped HPC cluster. The Ray dashboard is not exposed beyond the job's private VLAN.
- Contributors should ensure Ray dashboards are never bound to public interfaces. The recommended invocation is `ray start --head --dashboard-host=127.0.0.1`.
- No Ray cluster is used in any public-facing inference or API serving capacity.

**Accepted by**: Lukasz Augustyniak (`aisolutions@lukaszaugustyniak.com`), 2026-04-29

---

## DVC remote endpoint disclosure

`.dvc/config` references an internal e-Science Cloud MinIO endpoint (`https://s3min.e-science.pl`, bucket `s3min-tkajdanowicz-1724771116`). No credentials are stored in this file; AWS access keys belong in `.dvc/config.local` which is covered by `.gitignore`. The endpoint is authentication-walled. This disclosure is accepted as a configuration-topology leak with no direct exploitability; it will be moved to `.dvc/config.local` in a follow-up cleanup PR.

## CVE upgrade history

| Date | Packages upgraded | CVEs addressed |
|---|---|---|
| 2026-04-29 | `vllm`, `torch`, `langchain-core`, `deepdiff`, `langchain-text-splitters`, `langchain-community`, `authlib`, `h11` | CVE-2026-22778, CVE-2025-32434, CVE-2025-68664, CVE-2025-58367, CVE-2025-6985, CVE-2025-6984, CVE-2026-27962, CVE-2025-43859 |
