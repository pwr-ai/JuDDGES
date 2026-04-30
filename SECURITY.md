# Security Policy

## Supported versions

| Version | Supported |
| --- | --- |
| `master` (latest commit) | yes |
| `0.1.0` (research-reproducibility tag `neurips_v0.1`) | yes — security fixes only |
| Older commits | no — please update |

## Reporting a vulnerability

- Primary channel: GitHub Private Vulnerability Reporting at https://github.com/pwr-ai/JuDDGES/security/advisories/new (this is the preferred method — it gives us a private channel and an audit trail).
- Backup channel: email **aisolutions@lukaszaugustyniak.com** with the subject prefix `[JuDDGES SECURITY]`.
- **Do NOT** open public GitHub issues for security matters. If you have already done so by mistake, close the issue and use the channels above.

## What to include in your report

- A clear description of the vulnerability and its impact
- Step-by-step reproduction instructions or a minimal proof-of-concept
- The affected version, commit SHA, or release tag
- The environment in which you reproduced it (OS, Python version, CUDA version where relevant)
- Any suggested mitigation or patch
- Whether you would like public credit when the advisory is published

## What to expect

- Initial acknowledgement within **5 working days**
- A status update or triage outcome within **14 working days**
- A coordinated disclosure timeline negotiated case-by-case (default: up to **90 days** between report and public disclosure, shorter if the issue is actively exploited)

## Scope

- **In scope:** code in this repository (`juddges/`, `scripts/`, `label_studio_toolkit/`), DVC pipeline definitions in `dvc.yaml`, the Docker Compose configurations shipped with the repo
- **Out of scope:** third-party Python dependencies (please report upstream and let us know), Label Studio itself (report to https://github.com/HumanSignal/label-studio/security), Weaviate (report to https://github.com/weaviate/weaviate/security), models hosted on Hugging Face (report to the respective model authors)

## Public disclosure

- Once a fix is merged and a release is cut, a GitHub Security Advisory will be published at https://github.com/pwr-ai/JuDDGES/security/advisories
- Reporters will be credited in the advisory unless they request otherwise

## Defensive practices for users

- Pin to a specific release tag rather than `master` for production use
- Review `pyproject.toml` and `uv.lock` before upgrading dependencies
- Do not commit `.env`, credentials, or API keys; the project ships a `.env.example` template
- When ingesting third-party legal documents, treat the text as untrusted input — apply the project's pseudonymisation utilities before publishing derivatives
