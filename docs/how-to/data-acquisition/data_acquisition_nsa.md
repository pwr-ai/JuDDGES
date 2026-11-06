# NSA Data Acquisition

The `scripts/nsa` directory contains scripts that scrape judgments from the NSA archive
(<https://orzeczenia.nsa.gov.pl>), keep the progress in plain files (no database) and produce the
parquet dataset consumed by `push_to_repo.py`.

## Requirements

1. **Python environment** with the project dependencies (run the scripts from the repository root,
   preferably in a Docker container).
2. **Data storage path** – `NSA_DATA_PATH` from `juddges.settings` (`data/datasets/nsa`). Files are
   written under:
   - `state/document_lists.jsonl` – append-only, one row per scraped date
     `{date, num_docs, document_ids, scraped_at}`; the latest row per date wins.
   - `state/documents.parquet` – one row per document: `doc_id, judgment_date, first_seen_at,
     page_downloaded_at, has_reasons, reasons_found_at, last_reasons_check_at,
     reasons_check_count, listed_dates`. `listed_dates` holds every date on whose listing the
     document appeared (normally one). Rewritten atomically (temp file + fsync + rename).
   - `state/pages/pages_<timestamp>.parquet` – append-only chunks `{doc_id, page, downloaded_at}`;
     the latest page version per document wins.
   - `dataset/data_{i}.parquet` – extracted dataset (output of `extract_data_from_pages.py`).

   `state/` is git-ignored.
3. **Proxy** – not required. All scripts accept an optional `--proxy-address
   http://user:password@host:port`.

## Politeness defaults

The scripts use a single scraper, single thread, and wait a random 5–10 s after every request
(about one request per 7.5 s on average). Change it with `--wait-min` / `--wait-max`. The wait is
applied after every request attempt, including failed ones. Failed requests are retried at most 5
times with a growing delay (10 s, 20 s, 40 s, 80 s, plus jitter), and urllib3 adds at most 2
retries (backoff 10 s, `Retry-After` respected) for 429/5xx statuses.

Error handling: a date or a document that still fails after those retries is logged and skipped
(it stays pending and is retried on the next run). After `--max-consecutive-errors` (default 5)
such failures in a row a script stops; `check_reasons.py` counts failed document downloads within
a date and failed dates, and stops the whole run in both cases. Progress is saved regularly (also
on Ctrl-C), so re-running resumes where it stopped.

Only one script can use a state directory at a time: each takes an exclusive lock on
`state/.lock` and exits immediately with an error message if another one holds it.

## Sentencja first, uzasadnienie later

NSA first publishes a judgment with only the "Sentencja" section. The "Uzasadnienie" (reasons) is
added to the same document page weeks or months later. A downloaded page is therefore flagged
(`has_reasons`, detected from the page by `juddges.data.nsa.extractor.has_reasons`) and the flagged
documents are re-checked on a schedule by `check_reasons.py`, instead of re-downloading everything.

## Scripts

### `scrap_documents_list.py`

Scrapes document IDs for each date (newest first) and registers new documents in
`documents.parquet`. Dates already present are skipped, unless they are within the refresh window.
State is saved after every date; a failed date is logged and skipped.

| Option | Default | Description |
| --- | --- | --- |
| `--start-date` | `1981-01-01` | First date to scrape |
| `--end-date` | two weeks ago (Poland) | Last date to scrape |
| `--refresh-days-back` | `30` | Re-scrape dates not older than this |
| `--wait-min`, `--wait-max` | `5`, `10` | Delay after every request (s) |
| `--proxy-address` | none | Optional proxy |
| `--max-consecutive-errors` | `5` | Stop after N errors in a row |
| `--state-dir` | `NSA_DATA_PATH/state` | State directory |
| `--log-file` | none | Log file |

### `download_document_pages.py`

Downloads pages of documents that have none yet (newest judgment dates first), sets `has_reasons`
(and `reasons_found_at` when true). Pages and state are flushed every `--flush-every` pages (default
50). Options: `--limit`, `--flush-every`, plus the wait / proxy / error / state / log options above.

### `check_reasons.py`

Re-checks documents with `has_reasons == False` whose `judgment_date` is within `--max-age-days`
(default 730). Documents of one judgment date are due when the time since their
`last_reasons_check_at` (or `page_downloaded_at` if never checked) reaches the `--schedule` interval
for their `reasons_check_count`; the default schedule is `30,60,90,180,365` days, and after its end
the last interval is reused. The interval is measured from the previous check, so the default
schedule checks about 30, 90, 180, 360 and 725 days after the page was downloaded (cumulative
sums), and then every 365 days until the judgment is older than `--max-age-days`.

For each due date the cheaper strategy is chosen:

- **search** – ask NSA for the date's judgments "with reasons" and re-download only the flagged
  documents that appear in the result. The number `n` of judgments with reasons is fetched first
  (2 requests); if it is 0 nothing else is requested. Otherwise the search costs another
  `2 + ceil(n/10)` requests (submit the form, then fetch the result pages). Search is chosen when
  that is fewer requests than the number of flagged documents;
- **direct** – re-download every flagged document of that date otherwise. Dates with 2 or fewer
  flagged documents go straight to this strategy without the count.

Documents that were listed on more than one date (NSA listings sometimes shift between dates, so
the same ID appears on two neighbouring days) are always downloaded directly, because absence from
a date's search result does not prove that they have no reasons.

Updates `has_reasons`, `reasons_found_at`, `last_reasons_check_at`, `reasons_check_count` and stores
the new page versions. Pages are stored before the state that refers to them, every 50 pages and
when the run is interrupted. A document whose download failed is not marked as checked, so it is
retried on the next run.

`--dry-run` prints a table (dates, flagged counts, strategy, estimated requests) without any
network access. As `n` is unknown offline, the estimate uses the number of all known documents of
the date as an upper bound for the search cost.

### `extract_data_from_pages.py`

Takes the latest page of every document and writes `NSA_DATA_PATH/dataset/data_{i}.parquet`
(about `--rows-per-file` documents each, default 50000, all held in memory at once) using
`NSADataExtractor`. Existing `data_*.parquet` files in the output directory are removed after
confirmation. Options: `--n-jobs`, `--rows-per-file`, `--state-dir`, `--output-path`, `--log-file`.

## Recommended run order

First run:

```bash
python scripts/nsa/scrap_documents_list.py
python scripts/nsa/download_document_pages.py
python scripts/nsa/extract_data_from_pages.py
python scripts/nsa/push_to_repo.py
```

The first two steps take long (one request per 5–10 s) but can be interrupted and re-run.

Periodic "what's new" run:

```bash
python scripts/nsa/scrap_documents_list.py   # skips known dates, re-scrapes the last 30 days
python scripts/nsa/download_document_pages.py
python scripts/nsa/check_reasons.py --dry-run   # inspect the plan first
python scripts/nsa/check_reasons.py
python scripts/nsa/extract_data_from_pages.py
python scripts/nsa/push_to_repo.py
```
