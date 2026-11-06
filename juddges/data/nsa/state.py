"""File-based state of the NSA scraping pipeline (no database).

Layout of the state directory (default: ``NSA_DATA_PATH / "state"``):

- ``document_lists.jsonl`` - append-only, one row per scraped date
  ``{date, num_docs, document_ids, scraped_at}``; the latest row per date wins.
- ``documents.parquet`` - one row per document (see ``DOCUMENT_COLUMNS``), rewritten atomically.
- ``pages/pages_<timestamp>.parquet`` - append-only chunks ``{doc_id, page, downloaded_at}``;
  the latest version of a page per document wins.
"""

import fcntl
import json
import os
from collections.abc import Iterable, Iterator
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from loguru import logger

from juddges.settings import NSA_DATA_PATH

DEFAULT_STATE_DIR = NSA_DATA_PATH / "state"

DOCUMENT_COLUMNS = {
    "doc_id": "object",
    "judgment_date": "object",  # YYYY-MM-DD
    "first_seen_at": "datetime64[ns]",
    "page_downloaded_at": "datetime64[ns]",
    "has_reasons": "boolean",  # NA until the page is downloaded
    "reasons_found_at": "datetime64[ns]",
    "last_reasons_check_at": "datetime64[ns]",
    "reasons_check_count": "int64",
    # All dates on whose NSA listing the document appeared (normally exactly one). More than one
    # means the listing shifted between dates, so the judgment date is not reliable.
    "listed_dates": "object",
}


def empty_documents() -> pd.DataFrame:
    return pd.DataFrame({col: pd.Series(dtype=dtype) for col, dtype in DOCUMENT_COLUMNS.items()})


def _index_by_doc_id(docs: pd.DataFrame) -> pd.DataFrame:
    docs = docs.set_index("doc_id", drop=False)
    docs.index.name = None
    return docs


def register_documents(
    docs: pd.DataFrame, judgment_date: str, doc_ids: Iterable[str], now: datetime
) -> pd.DataFrame:
    """Add documents not yet known; for known ones only record that they were listed on this date."""
    doc_ids = list(dict.fromkeys(doc_ids))
    for doc_id in doc_ids:
        if doc_id in docs.index and judgment_date not in docs.at[doc_id, "listed_dates"]:
            docs.at[doc_id, "listed_dates"] = [*docs.at[doc_id, "listed_dates"], judgment_date]
    new_ids = [d for d in doc_ids if d not in docs.index]
    if not new_ids:
        return docs
    new = empty_documents()
    new["doc_id"] = new_ids
    new["judgment_date"] = judgment_date
    new["first_seen_at"] = pd.Timestamp(now)
    new["reasons_check_count"] = 0
    new["listed_dates"] = [[judgment_date] for _ in new_ids]
    new = new.astype(DOCUMENT_COLUMNS)
    return _index_by_doc_id(pd.concat([docs, new]) if len(docs) else new)


class StateLocked(RuntimeError):
    """Another process is using the same state directory."""


def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write to a temp file in the same directory, fsync it and rename over `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    with tmp.open("wb") as f:
        df.to_parquet(f, index=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class NSAState:
    def __init__(self, state_dir: Path | None = None) -> None:
        self.state_dir = Path(state_dir) if state_dir else DEFAULT_STATE_DIR
        self.lists_path = self.state_dir / "document_lists.jsonl"
        self.documents_path = self.state_dir / "documents.parquet"
        self.pages_dir = self.state_dir / "pages"
        self.lock_path = self.state_dir / ".lock"
        self._lock_file = None

    # --- locking --------------------------------------------------------------------------
    def acquire_lock(self) -> None:
        """Take an exclusive lock on the state directory until `release_lock` / process exit."""
        self.state_dir.mkdir(parents=True, exist_ok=True)
        lock_file = self.lock_path.open("w")
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock_file.close()
            raise StateLocked(
                f"State directory {self.state_dir} is in use by another NSA script "
                f"(lock file: {self.lock_path}). Wait for it to finish."
            ) from None
        self._lock_file = lock_file

    def release_lock(self) -> None:
        if self._lock_file is not None:
            self._lock_file.close()  # closing releases the flock
            self._lock_file = None

    # --- document lists -------------------------------------------------------------------
    def append_document_list(self, date: str, document_ids: list[str], now: datetime) -> None:
        row = {
            "date": date,
            "num_docs": len(document_ids),
            "document_ids": document_ids,
            "scraped_at": now.isoformat(),
        }
        self.state_dir.mkdir(parents=True, exist_ok=True)
        with self.lists_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def load_document_lists(self) -> dict[str, dict]:
        """Latest row per date."""
        rows: dict[str, dict] = {}
        if self.lists_path.exists():
            with self.lists_path.open(encoding="utf-8") as f:
                for line_no, line in enumerate(f, start=1):
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                        rows[row["date"]] = row
                    except (json.JSONDecodeError, KeyError):
                        # e.g. a line cut short by a crash during append
                        logger.warning(f"Skipping malformed line {line_no} in {self.lists_path}")
        return rows

    # --- documents ------------------------------------------------------------------------
    def load_documents(self) -> pd.DataFrame:
        """Documents indexed by doc_id (the column is kept as well)."""
        if not self.documents_path.exists():
            return _index_by_doc_id(empty_documents())
        docs = pd.read_parquet(self.documents_path)
        docs["listed_dates"] = [list(x) for x in docs["listed_dates"]]  # arrays -> lists
        return _index_by_doc_id(docs.astype(DOCUMENT_COLUMNS))

    def save_documents(self, docs: pd.DataFrame) -> None:
        """Atomic write: temp file in the same directory + rename."""
        _atomic_write_parquet(docs.reset_index(drop=True), self.documents_path)

    # --- pages ----------------------------------------------------------------------------
    def append_pages(self, pages: list[dict]) -> None:
        """Store a chunk of ``{doc_id, page, downloaded_at}`` rows as a new file."""
        if not pages:
            return
        self.pages_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S%f")
        _atomic_write_parquet(pd.DataFrame(pages), self.pages_dir / f"pages_{stamp}.parquet")

    def iter_latest_pages(self, min_rows: int = 0) -> Iterator[pd.DataFrame]:
        """Yield DataFrames with only the pages that are the latest version of their doc.

        Chunks are concatenated until they have at least `min_rows` rows (the last batch may have
        fewer). If two versions have the same `downloaded_at`, the one from the later chunk wins.
        """
        files = sorted(self.pages_dir.glob("pages_*.parquet"))
        meta = pd.concat(
            [
                pd.read_parquet(f, columns=["doc_id", "downloaded_at"]).assign(file=f.name)
                for f in files
            ]
            or [pd.DataFrame(columns=["doc_id", "downloaded_at", "file"])]
        )
        latest = meta.sort_values(["downloaded_at", "file"]).drop_duplicates("doc_id", keep="last")
        latest_per_file = latest.groupby("file")["doc_id"].apply(set).to_dict()
        batch: list[pd.DataFrame] = []
        n_rows = 0
        for f in files:
            keep = latest_per_file.get(f.name)
            if keep:
                df = pq.read_table(f).to_pandas()
                batch.append(df[df["doc_id"].isin(keep)])
                n_rows += len(batch[-1])
            if batch and n_rows >= min_rows:
                yield pd.concat(batch).reset_index(drop=True)
                batch, n_rows = [], 0
        if batch:
            yield pd.concat(batch).reset_index(drop=True)
