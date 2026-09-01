# JuDDGES-PL Corpus Characterization (§5)

This directory holds the EDA pipeline that produces the descriptive
statistics, tables, and figures for §5 ("Corpus characterization") of the
NeurIPS 2026 JuDDGES-PL paper.

## Prerequisites

```bash
pip install datasets pandas numpy matplotlib tqdm
```

A Hugging Face token is required only for gated datasets; the two JuDDGES
datasets are public, so `huggingface-cli login` is optional.

## How to run

```bash
# Single dataset
python corpus_characterization.py --dataset pl-court
python corpus_characterization.py --dataset pl-nsa

# Both datasets + cross-branch comparison (recommended for the paper)
python corpus_characterization.py --all

# Smoke-test on 5k docs per dataset
python corpus_characterization.py --all --sample-size 5000

# Override output dir
python corpus_characterization.py --all --output-dir /tmp/juddges-eda
```

The script streams rows via `datasets.load_dataset(..., streaming=True)`, so a
laptop with ~8 GB of RAM is sufficient. For each dataset the pipeline
materialises the streamed rows once (bounded by `--sample-size`) and feeds the
same in-memory list to every stat group; this avoids paying the network cost
six times. Set `--sample-size` if you want hard latency caps; otherwise the
script will stream the full split.

## Expected runtime

Numbers below are rough estimates on a residential 100 Mbps connection,
single-threaded. They are dominated by HF streaming I/O, not by the Python
work.

| Dataset                       | Sample (5k) | Full stream            |
| ----------------------------- | ----------- | ---------------------- |
| `JuDDGES/pl-court-raw-enriched` | ~3-5 min    | ~30-60 min (≈100K-1M rows) |
| `JuDDGES/pl-nsa-enriched`     | ~3-5 min    | ~2-6 h (≈1M-10M rows)  |
| `cross_branch_comparison`     | ~1 min      | ~10-30 min             |

## Output layout

```
eda/
├── corpus_characterization.py
├── README.md
└── output/
    ├── tables/
    │   ├── {dataset}_coverage_by_court_level.csv
    │   ├── {dataset}_coverage_by_year.csv
    │   ├── {dataset}_coverage_by_type.csv
    │   ├── {dataset}_length_distribution.csv
    │   ├── {dataset}_section_presence.csv
    │   ├── {dataset}_top30_statutes.csv
    │   ├── {dataset}_case_citations_top50.csv
    │   ├── {dataset}_extraction_field_stats.csv
    │   ├── {dataset}_pseudonym_audit.csv
    │   ├── {dataset}_pseudonym_by_court.csv
    │   ├── {dataset}_pseudonym_by_year.csv
    │   └── cross_branch_vocab.csv
    ├── figures/
    │   ├── {dataset}_year_timeline.{png,pgf}
    │   ├── {dataset}_length_hist.{png,pgf}
    │   └── {dataset}_top30_statutes.{png,pgf}
    └── summary.json
```

`summary.json` is the headline-numbers dump that gets pasted into the paper to
replace placeholders such as "hundreds of thousands".

## Mapping outputs to the paper

| Paper section                   | Stat function                | Artifact                                                                          |
| ------------------------------- | ---------------------------- | --------------------------------------------------------------------------------- |
| §5.1 Coverage                   | `coverage_stats`             | `coverage_by_*` CSVs, `year_timeline.png`                                         |
| §5.2 Document structure         | `document_structure_stats`   | `length_distribution.csv`, `section_presence.csv`, `length_hist.png`              |
| §5.3 Citation patterns          | `citation_patterns`          | `top30_statutes.csv/.png`, `case_citations_top50.csv`                             |
| §5.4 Extraction-field stats     | `extraction_field_stats`     | `extraction_field_stats.csv`                                                      |
| §5.5 Pseudonymization fidelity  | `pseudonymization_audit`     | `pseudonym_audit.csv`, `pseudonym_by_court.csv`, `pseudonym_by_year.csv`          |
| §5.6 Cross-branch comparison    | `cross_branch_comparison`    | `cross_branch_vocab.csv` (uses `top30_statutes.csv` written by §5.3)              |

## Reproducibility notes

- `RANDOM_STATE=42` is pinned for reservoir sampling, stratified sampling, and
  any `np.random` use.
- Each stat group is wrapped in `try/except`; one failing group does not abort
  the run, and the failure is recorded in `summary.json` under
  `{"error": ...}`.
- Library versions are logged at startup.

## Field-name TODOs

The Croissant JSON-LDs do not expose every column unambiguously. The
following are flagged as `TODO(user)` in `corpus_characterization.py` and
should be confirmed against a live row dump before running on the full
corpora:

- `pl-nsa-enriched` exposes no plain `date` column in its Croissant; the year
  is currently extracted from `docket_number` / `judgment_id`.
- `legalBases` (court) and `extracted_legal_bases` (NSA) are typed as
  `sc:Text` but treated as JSON-encoded lists. `_as_list` decodes both shapes.
