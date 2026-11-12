"""Streaming, resumable ingest of a whole HF parquet dataset into the v2 collections.

A *unit* is a slice of `unit_rows` rows of one parquet row group (`file:row_group:start`). Units
are read one at a time, so the dataset is never held in memory. A JSON checkpoint records every
finished unit; a restart skips them. Object ids are deterministic, so a unit that was cut off
half-way is simply written again (overwrite, no duplicates).

Pipeline per unit: rows are mapped in the main process, their texts are chunked in a process
pool (tokenizer-bound) while the previous unit is embedded. Theses are embedded and the unit's
judgments imported (and flushed) first, because Weaviate rejects a chunk whose `parent_judgment`
does not exist yet. Chunk passages then go to TEI with several requests in flight, and each TEI
response goes straight into a Weaviate batch, so only a few batches of vectors are alive at a
time. A unit is checkpointed only after both batches were flushed without failed objects or
references.
"""

import json
import os
import time
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger

from juddges.data.weaviate_v2.ingest import (
    chunk_properties,
    court_chunk_text,
    judgment_properties,
    make_splitter,
    nsa_chunk_text,
    nsa_judgment_properties,
    split_judgment,
)
from juddges.data.weaviate_v2.schema import (
    JUDGMENT_CHUNKS_COLLECTION,
    JUDGMENTS_COLLECTION,
    PARENT_JUDGMENT_REF,
    JudgmentVector,
    chunk_uuid,
    judgment_uuid,
)
from juddges.settings import VectorName

Row = Mapping[str, Any]


# --- datasets ------------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetSpec:
    """How the rows of one HF dataset become `Judgments` properties and text to chunk."""

    repo_id: str
    columns: tuple[str, ...]
    judgment: Callable[[Row, datetime], dict[str, Any]]
    chunk_text: Callable[[Row], str]


DATASETS: dict[str, DatasetSpec] = {
    "pl-court-raw": DatasetSpec(
        repo_id="JuDDGES/pl-court-raw",
        columns=(
            "judgment_id",
            "docket_number",
            "judgment_date",
            "publication_date",
            "court_id",
            "department_id",
            "judgment_type",
            "presiding_judge",
            "judges",
            "legal_bases",
            "keywords",
            "full_text",
            "court_name",
            "department_name",
            "extracted_legal_bases",
            "references",
            "thesis",
            "country",
            "court_type",
        ),
        judgment=judgment_properties,
        chunk_text=court_chunk_text,
    ),
    "pl-nsa": DatasetSpec(
        repo_id="JuDDGES/pl-nsa",
        columns=(
            "judgment_id",
            "docket_number",
            "judgment_type",
            "judgment_date",
            "court_name",
            "judges",
            "presiding_judge",
            "keywords",
            "decision",
            "extracted_legal_bases",
            "thesis",
            "full_text",
            "country",
        ),
        judgment=nsa_judgment_properties,
        chunk_text=nsa_chunk_text,
    ),
}


def parquet_files(repo_id: str, local_files_only: bool = True) -> list[str]:
    """Parquet files of a dataset snapshot (from the HF cache unless downloading is allowed)."""
    from huggingface_hub import snapshot_download

    snapshot = snapshot_download(
        repo_id,
        repo_type="dataset",
        allow_patterns=["data/*.parquet"],
        local_files_only=local_files_only,
    )
    return sorted(str(p) for p in Path(snapshot).glob("data/*.parquet"))


# --- units ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class Unit:
    """`num_rows` rows from `start` within one row group of one parquet file."""

    file: str
    row_group: int
    start: int
    num_rows: int

    @property
    def key(self) -> str:
        return f"{Path(self.file).name}:{self.row_group}:{self.start}"


def plan_units(files: Sequence[str], unit_rows: int) -> list[Unit]:
    """All units of the files, from parquet metadata only."""
    units: list[Unit] = []
    for f in files:
        meta = pq.ParquetFile(f).metadata
        for rg in range(meta.num_row_groups):
            n = meta.row_group(rg).num_rows
            units.extend(
                Unit(f, rg, start, min(unit_rows, n - start)) for start in range(0, n, unit_rows)
            )
    return units


def _row_group_ranges(
    pf: pq.ParquetFile, rg: int, ranges: Sequence[tuple[int, int]], columns: Sequence[str]
) -> Iterator[pa.Table]:
    """Tables of the `(start, num_rows)` ranges (ascending) of a row group, read in one pass."""
    batches = pf.iter_batches(
        batch_size=max(n for _, n in ranges), row_groups=[rg], columns=list(columns)
    )
    buffer: list[pa.RecordBatch] = []
    buffer_start = buffered = 0  # row offset of the buffer's first row, rows in the buffer
    for start, num_rows in ranges:
        while buffer_start + buffered < start + num_rows:
            batch = next(batches, None)
            if batch is None:
                break
            buffer.append(batch)
            buffered += batch.num_rows
        if not buffer:
            raise RuntimeError(f"row group {rg} has no rows at {start}")
        table = pa.Table.from_batches(buffer)
        yield table.slice(start - buffer_start, num_rows)
        rest = table.slice(start - buffer_start + num_rows)
        buffer, buffer_start, buffered = rest.to_batches(), start + num_rows, rest.num_rows


def iter_units(
    units: Sequence[Unit], columns: Sequence[str], skip: Callable[[Unit], bool]
) -> Iterator[tuple[Unit, list[dict[str, Any]]]]:
    """Rows of every unit not skipped, read row group by row group.

    Each unit gets the rows at its own `start`, also when earlier units of the row group are not
    listed (a resume passes only the pending ones). Fully skipped row groups are not read.
    """
    by_group: dict[tuple[str, int], list[Unit]] = {}
    for u in units:
        by_group.setdefault((u.file, u.row_group), []).append(u)
    for (f, rg), group in by_group.items():
        todo = sorted((u for u in group if not skip(u)), key=lambda u: u.start)
        if not todo:
            continue
        ranges = [(u.start, u.num_rows) for u in todo]
        for unit, table in zip(todo, _row_group_ranges(pq.ParquetFile(f), rg, ranges, columns)):
            if table.num_rows != unit.num_rows:
                raise RuntimeError(
                    f"{unit.key}: read {table.num_rows} rows, planned {unit.num_rows}"
                )
            yield unit, table.to_pylist()


# --- checkpoint ----------------------------------------------------------------------------


class Checkpoint:
    """Finished and failed units in a JSON file, rewritten atomically after every change."""

    def __init__(self, path: Path, dataset: str, unit_rows: int) -> None:
        self.path = Path(path)
        self.data: dict[str, Any] = {
            "dataset": dataset,
            "unit_rows": unit_rows,
            "done": {},
            "failed": {},
        }
        if self.path.exists():
            stored = json.loads(self.path.read_text())
            if stored.get("dataset") != dataset:
                raise ValueError(
                    f"{path} is for dataset {stored.get('dataset')!r}, not {dataset!r}"
                )
            if stored.get("unit_rows") != unit_rows:
                raise ValueError(
                    f"{path} was written with unit_rows={stored.get('unit_rows')}, "
                    f"not {unit_rows}; unit keys would not match"
                )
            self.data.update(stored)

    @property
    def done(self) -> dict[str, Any]:
        return self.data["done"]

    @property
    def failed(self) -> dict[str, str]:
        return self.data["failed"]

    def is_done(self, key: str) -> bool:
        return key in self.done

    def mark_done(self, key: str, stats: Mapping[str, Any]) -> None:
        self.done[key] = dict(stats)
        self.failed.pop(key, None)
        self._save()

    def mark_failed(self, key: str, reason: str) -> None:
        self.failed[key] = reason
        self._save()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.data, indent=1, default=str))
        os.replace(tmp, self.path)


# --- writers -------------------------------------------------------------------------------


class Embedder(Protocol):
    def passages(self, texts: Sequence[str]) -> list[list[float]]: ...


class UnitSink(Protocol):
    failed: list[str]

    def add_judgment(
        self, props: dict[str, Any], uuid: str, vector: list[float] | None
    ) -> None: ...

    def add_chunk(
        self, props: dict[str, Any], uuid: str, parent_uuid: str, vector: list[float]
    ) -> None: ...


class UnitWriter(Protocol):
    def batch(self) -> Any:
        """Context manager yielding a `UnitSink`; on exit it is flushed and `sink.failed` set."""


@dataclass
class _WeaviateSink:
    batch: Any
    failed: list[str] = field(default_factory=list)

    def add_judgment(self, props, uuid, vector):
        self.batch.add_object(
            collection=JUDGMENTS_COLLECTION,
            properties=props,
            uuid=uuid,
            vector={JudgmentVector.THESIS: vector} if vector else None,
        )

    def add_chunk(self, props, uuid, parent_uuid, vector):
        self.batch.add_object(
            collection=JUDGMENT_CHUNKS_COLLECTION,
            properties=props,
            uuid=uuid,
            references={PARENT_JUDGMENT_REF: parent_uuid},
            vector={VectorName.DEFAULT: vector},
        )


class WeaviateWriter:
    """Fixed-size Weaviate batches opened per unit, so failures are attributed to their unit."""

    def __init__(self, client: Any, batch_size: int = 500, concurrent_requests: int = 4) -> None:
        self.client = client
        self.batch_size = batch_size
        self.concurrent_requests = concurrent_requests

    @contextmanager
    def batch(self) -> Iterator[_WeaviateSink]:
        with self.client.batch.fixed_size(
            batch_size=self.batch_size, concurrent_requests=self.concurrent_requests
        ) as batch:
            sink = _WeaviateSink(batch)
            yield sink
        sink.failed = [f"object: {f.message}" for f in self.client.batch.failed_objects] + [
            f"reference: {f.message}" for f in self.client.batch.failed_references
        ]


# --- chunking pool -------------------------------------------------------------------------

_SPLITTER: Any = None


def init_chunk_worker() -> None:
    """Pool initializer: one tokenizer + splitter per process."""
    global _SPLITTER
    _SPLITTER = make_splitter()


def split_text(text: str) -> list[str]:
    """Pool task: chunks of one judgment text."""
    return split_judgment(_SPLITTER, text)


# --- pipeline ------------------------------------------------------------------------------


@dataclass
class RunStats:
    units_done: int = 0
    units_skipped: int = 0
    units_failed: int = 0
    judgments: int = 0
    chunks: int = 0
    theses: int = 0
    skipped_empty: int = 0
    seconds: float = 0.0
    aborted: bool = False


@dataclass
class _Prepared:
    unit: Unit
    judgments: list[dict[str, Any]]
    chunks: Any  # handle with .get() -> list[list[str]]
    skipped_empty: int


def _embed_with_retry(
    embedder: Embedder, texts: Sequence[str], attempts: int, wait: float
) -> list[list[float]]:
    for attempt in range(1, attempts + 1):
        try:
            return embedder.passages(texts)
        except Exception as exc:  # TEI overload / restarts are transient
            if attempt == attempts:
                raise
            logger.warning("embedding attempt {}/{} failed: {}", attempt, attempts, exc)
            time.sleep(wait * 2 ** (attempt - 1))
    raise AssertionError("unreachable")


def _batches(items: list[Any], size: int) -> Iterator[list[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _raise_on_failures(unit: Unit, failed: list[str], what: str) -> None:
    if failed:
        for msg in failed[:3]:
            logger.error("{} {}: {}", unit.key, what, msg[:500])
        raise RuntimeError(f"{len(failed)} failed {what} objects/references")


def _embedded_in_order(
    pool: ThreadPoolExecutor,
    batches: Iterable[list[Any]],
    embed: Callable[[list[str]], list[list[float]]],
    window: int,
) -> Iterator[tuple[list[Any], list[list[float]]]]:
    """(batch, vectors) in submission order with at most `window` requests in flight."""
    in_flight: deque[tuple[list[Any], Future]] = deque()
    for batch in batches:
        in_flight.append((batch, pool.submit(embed, [item[2] for item in batch])))
        if len(in_flight) >= window:
            done_batch, fut = in_flight.popleft()
            yield done_batch, fut.result()
    while in_flight:
        done_batch, fut = in_flight.popleft()
        yield done_batch, fut.result()


def run_ingest(
    work: Iterable[tuple[Unit, list[Row]]],
    spec: DatasetSpec,
    *,
    split_async: Callable[[list[str]], Any],
    embedder: Embedder,
    writer: UnitWriter,
    checkpoint: Checkpoint,
    embed_batch: int = 128,
    embed_concurrency: int = 4,
    embed_attempts: int = 5,
    retry_wait: float = 5.0,
    limit_units: int | None = None,
    units_pending: int | None = None,
    max_consecutive_failures: int = 3,
) -> RunStats:
    """Ingest `work` unit by unit; chunking of the next unit overlaps embedding of this one.

    `split_async(texts)` returns a handle whose `.get()` gives one chunk list per text (e.g.
    `Pool.map_async`). Units already in the checkpoint are skipped. A unit with an exception or
    failed objects is recorded as failed and not checkpointed; after `max_consecutive_failures`
    such units in a row the run stops (the infrastructure is probably down).
    """
    stats = RunStats()
    now = datetime.now(timezone.utc)
    t_run = time.perf_counter()
    consecutive_failures = 0

    def embed(texts: list[str]) -> list[list[float]]:
        return _embed_with_retry(embedder, texts, embed_attempts, retry_wait)

    def prepare(unit: Unit, rows: list[Row]) -> _Prepared:
        judgments, texts, skipped = [], [], 0
        for row in rows:
            text = spec.chunk_text(row)
            if not row.get("judgment_id") or not text.strip():
                skipped += 1
                continue
            judgments.append(spec.judgment(row, now))
            texts.append(text)
        return _Prepared(unit, judgments, split_async(texts), skipped)

    def process(prep: _Prepared, pool: ThreadPoolExecutor) -> dict[str, Any]:
        t0 = time.perf_counter()
        chunk_lists = prep.chunks.get()
        t_chunked = time.perf_counter()
        theses = [(i, -1, j["thesis"]) for i, j in enumerate(prep.judgments) if j.get("thesis")]
        thesis_vectors: dict[int, list[float]] = {}
        for batch, vectors in _embedded_in_order(
            pool, _batches(theses, embed_batch), embed, embed_concurrency
        ):
            thesis_vectors.update((i, vec) for (i, _, _), vec in zip(batch, vectors))
        with writer.batch() as sink:
            for i, j in enumerate(prep.judgments):
                sink.add_judgment(j, judgment_uuid(j["judgment_id"]), thesis_vectors.get(i))
        _raise_on_failures(prep.unit, sink.failed, "judgments")

        items = [(i, idx, text) for i, cs in enumerate(chunk_lists) for idx, text in enumerate(cs)]
        n_chunks = 0
        with writer.batch() as sink:
            for batch, vectors in _embedded_in_order(
                pool, _batches(items, embed_batch), embed, embed_concurrency
            ):
                for (i, idx, text), vec in zip(batch, vectors):
                    j = prep.judgments[i]
                    sink.add_chunk(
                        chunk_properties(j, text, idx),
                        chunk_uuid(j["judgment_id"], idx),
                        judgment_uuid(j["judgment_id"]),
                        vec,
                    )
                    n_chunks += 1
        _raise_on_failures(prep.unit, sink.failed, "chunks")
        seconds = time.perf_counter() - t0
        return {
            "judgments": len(prep.judgments),
            "chunks": n_chunks,
            "theses": len(thesis_vectors),
            "skipped_empty": prep.skipped_empty,
            "chunk_wait_s": round(t_chunked - t0, 1),
            "seconds": round(seconds, 1),
            "passages_per_s": round((n_chunks + len(thesis_vectors)) / max(seconds, 1e-9), 1),
            "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }

    def finish(prep: _Prepared, pool: ThreadPoolExecutor) -> None:
        nonlocal consecutive_failures
        key = prep.unit.key
        try:
            unit_stats = process(prep, pool)
        except Exception as exc:
            stats.units_failed += 1
            consecutive_failures += 1
            checkpoint.mark_failed(key, f"{type(exc).__name__}: {exc}")
            logger.error("unit {} failed, not checkpointed: {}", key, exc)
            if consecutive_failures >= max_consecutive_failures:
                stats.aborted = True
            return
        consecutive_failures = 0
        checkpoint.mark_done(key, unit_stats)
        stats.units_done += 1
        stats.judgments += unit_stats["judgments"]
        stats.chunks += unit_stats["chunks"]
        stats.theses += unit_stats["theses"]
        stats.skipped_empty += unit_stats["skipped_empty"]
        elapsed = time.perf_counter() - t_run
        attempted = stats.units_done + stats.units_failed
        remaining = None
        if units_pending is not None:
            remaining = units_pending - attempted
            if limit_units is not None:
                remaining = min(remaining, limit_units - attempted)
        eta_h = remaining * elapsed / attempted / 3600 if remaining is not None else None
        logger.info(
            "unit {} done: {} judgments, {} chunks, {} theses, {} empty | {}s, {} passages/s, "
            "chunk wait {}s | run: {} units, {} chunks, {:.0f} chunks/s{}",
            key,
            unit_stats["judgments"],
            unit_stats["chunks"],
            unit_stats["theses"],
            unit_stats["skipped_empty"],
            unit_stats["seconds"],
            unit_stats["passages_per_s"],
            unit_stats["chunk_wait_s"],
            stats.units_done,
            stats.chunks,
            stats.chunks / max(elapsed, 1e-9),
            f", {remaining} units left, ETA {eta_h:.1f} h" if eta_h is not None else "",
        )

    pending: _Prepared | None = None
    with ThreadPoolExecutor(max_workers=embed_concurrency, thread_name_prefix="tei") as pool:
        for unit, rows in work:
            if checkpoint.is_done(unit.key):
                stats.units_skipped += 1
                continue
            if (
                limit_units is not None
                and stats.units_done + stats.units_failed + (pending is not None) >= limit_units
            ):
                break
            prepared = prepare(unit, rows)  # chunking starts in the background
            if pending is not None:
                finish(pending, pool)
                if stats.aborted:
                    pending = None
                    break
            pending = prepared
        if pending is not None and not stats.aborted:
            finish(pending, pool)
    stats.seconds = time.perf_counter() - t_run
    return stats
