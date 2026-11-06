"""File-based state of the NSA scraping pipeline (no database).

Layout of the state directory (default: ``NSA_DATA_PATH / "state"``):

- ``document_lists.jsonl`` - append-only, one row per scraped date
  ``{date, num_docs, document_ids, scraped_at}``; the latest row per date wins.
- ``documents.parquet`` - one row per document (see ``DOCUMENT_COLUMNS``), rewritten atomically.
- ``pages/pages_<timestamp>.parquet`` - append-only chunks ``{doc_id, page, downloaded_at}``;
  the latest version of a page per document wins.
"""

import json
import os
from collections.abc import Iterable, Iterator
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

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
    """Add documents not yet known (existing rows are left untouched)."""
    new_ids = [d for d in dict.fromkeys(doc_ids) if d not in docs.index]
    if not new_ids:
        return docs
    new = empty_documents()
    new["doc_id"] = new_ids
    new["judgment_date"] = judgment_date
    new["first_seen_at"] = pd.Timestamp(now)
    new["reasons_check_count"] = 0
    new = new.astype(DOCUMENT_COLUMNS)
    return _index_by_doc_id(pd.concat([docs, new]) if len(docs) else new)


class NSAState:
    def __init__(self, state_dir: Path | None = None) -> None:
        self.state_dir = Path(state_dir) if state_dir else DEFAULT_STATE_DIR
        self.lists_path = self.state_dir / "document_lists.jsonl"
        self.documents_path = self.state_dir / "documents.parquet"
        self.pages_dir = self.state_dir / "pages"

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

    def load_document_lists(self) -> dict[str, dict]:
        """Latest row per date."""
        rows: dict[str, dict] = {}
        if self.lists_path.exists():
            with self.lists_path.open(encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        row = json.loads(line)
                        rows[row["date"]] = row
        return rows

    # --- documents ------------------------------------------------------------------------
    def load_documents(self) -> pd.DataFrame:
        """Documents indexed by doc_id (the column is kept as well)."""
        if not self.documents_path.exists():
            return _index_by_doc_id(empty_documents())
        return _index_by_doc_id(pd.read_parquet(self.documents_path).astype(DOCUMENT_COLUMNS))

    def save_documents(self, docs: pd.DataFrame) -> None:
        """Atomic write: temp file in the same directory + rename."""
        self.state_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.documents_path.with_suffix(".parquet.tmp")
        docs.reset_index(drop=True).to_parquet(tmp, index=False)
        os.replace(tmp, self.documents_path)

    # --- pages ----------------------------------------------------------------------------
    def append_pages(self, pages: list[dict]) -> None:
        """Store a chunk of ``{doc_id, page, downloaded_at}`` rows as a new file."""
        if not pages:
            return
        self.pages_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S%f")
        path = self.pages_dir / f"pages_{stamp}.parquet"
        tmp = path.with_suffix(".parquet.tmp")
        pd.DataFrame(pages).to_parquet(tmp, index=False)
        os.replace(tmp, path)

    def iter_latest_pages(self) -> Iterator[pd.DataFrame]:
        """Yield one DataFrame per chunk with only the pages that are the latest of their doc."""
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
        for f in files:
            keep = latest_per_file.get(f.name)
            if keep:
                df = pq.read_table(f).to_pandas()
                yield df[df["doc_id"].isin(keep)].reset_index(drop=True)
