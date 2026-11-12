"""Ingest a sample of JuDDGES/pl-court-raw into the v2 collections and time every stage.

Picks `--n` judgments that have a thesis (50-1500 chars) and a full text, plus any ids listed
in `--extra-ids` (one judgment_id per line), so the sample can include judgments that are
also in juddges-app for side-by-side comparison. Object ids are deterministic, so re-running
overwrites instead of duplicating.

    python scripts/weaviate_v2/ingest_sample.py --n 1000 --extra-ids chf_ids.txt
"""

import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import typer
from huggingface_hub import snapshot_download
from loguru import logger
from rich.console import Console
from rich.table import Table

from juddges.data.weaviate_v2.ingest import (
    TeiEmbedder,
    chunk_properties,
    judgment_properties,
    make_splitter,
    split_judgment,
)
from juddges.data.weaviate_v2.schema import (
    JUDGMENT_CHUNKS_COLLECTION,
    JUDGMENTS_COLLECTION,
    PARENT_JUDGMENT_REF,
    JudgmentVector,
    chunk_uuid,
    connect_local,
    create_collections,
    judgment_uuid,
)
from juddges.settings import VectorName

console = Console()
DATASET = "JuDDGES/pl-court-raw"
COLUMNS = [
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
]


def parquet_files() -> list[str]:
    snapshot = snapshot_download(DATASET, repo_type="dataset", allow_patterns=["data/*.parquet"])
    return sorted(str(p) for p in Path(snapshot).glob("data/*.parquet"))


def pick_ids(files: list[str], n: int, seed: int) -> list[str]:
    """`n` random judgments with a 50-1500 char thesis and a full text."""
    candidates: list[str] = []
    for f in files:
        t = pq.read_table(f, columns=["judgment_id", "thesis", "full_text"])
        thesis_len = pc.fill_null(pc.utf8_length(t["thesis"]), 0)
        text_len = pc.fill_null(pc.utf8_length(t["full_text"]), 0)
        mask = pc.and_(
            pc.and_(pc.greater_equal(thesis_len, 50), pc.less_equal(thesis_len, 1500)),
            pc.greater(text_len, 0),
        )
        candidates.extend(t.filter(mask)["judgment_id"].to_pylist())
    logger.info("{} judgments with a thesis and a full text", len(candidates))
    return random.Random(seed).sample(sorted(candidates), n)


def load_rows(files: list[str], ids: set[str]) -> list[dict]:
    rows: list[dict] = []
    for f in files:
        t = pq.read_table(f, columns=COLUMNS)
        rows.extend(t.filter(pc.is_in(t["judgment_id"], value_set=pa.array(list(ids)))).to_pylist())
    return rows


def main(
    n: int = typer.Option(1000, help="random judgments with a thesis"),
    extra_ids: Path | None = typer.Option(None, help="file with extra judgment_ids, one per line"),
    seed: int = typer.Option(42),
    tei_url: str = typer.Option("http://127.0.0.1:8111"),
    host: str = typer.Option("localhost"),
    port: int = typer.Option(8110),
    grpc_port: int = typer.Option(50061),
    embed_batch: int = typer.Option(64),
    stats_out: Path | None = typer.Option(None, help="write timing stats as JSON"),
) -> None:
    t_start = time.perf_counter()
    files = parquet_files()
    ids = set(pick_ids(files, n, seed))
    if extra_ids:
        ids |= {line.strip() for line in extra_ids.read_text().splitlines() if line.strip()}
    rows = load_rows(files, ids)
    t_loaded = time.perf_counter()
    logger.info("loaded {} of {} requested judgments", len(rows), len(ids))

    splitter = make_splitter()
    now = datetime.now(timezone.utc)
    judgments = [judgment_properties(r, now) for r in rows]
    chunks = [
        (j, idx, text)
        for j, r in zip(judgments, rows)
        for idx, text in enumerate(split_judgment(splitter, r["full_text"]))
    ]
    t_chunked = time.perf_counter()

    embedder = TeiEmbedder(tei_url, batch_size=embed_batch)
    chunk_vectors = embedder.passages([text for _, _, text in chunks])
    with_thesis = [j for j in judgments if j.get("thesis")]
    thesis_vectors = dict(
        zip(
            [j["judgment_id"] for j in with_thesis],
            embedder.passages([j["thesis"] for j in with_thesis]),
        )
    )
    embedder.close()
    t_embedded = time.perf_counter()

    client = connect_local(host, port, grpc_port, api_key=os.environ["WEAVIATE_V2_API_KEY"])
    try:
        create_collections(client)
        with client.collections.get(JUDGMENTS_COLLECTION).batch.fixed_size(
            200, concurrent_requests=4
        ) as b:
            for j in judgments:
                vec = thesis_vectors.get(j["judgment_id"])
                b.add_object(
                    properties=j,
                    uuid=judgment_uuid(j["judgment_id"]),
                    vector={JudgmentVector.THESIS: vec} if vec else None,
                )
        failed_j = len(client.collections.get(JUDGMENTS_COLLECTION).batch.failed_objects)
        t_judgments = time.perf_counter()
        with client.collections.get(JUDGMENT_CHUNKS_COLLECTION).batch.fixed_size(
            500, concurrent_requests=4
        ) as b:
            for (j, idx, text), vec in zip(chunks, chunk_vectors):
                b.add_object(
                    properties=chunk_properties(j, text, idx),
                    uuid=chunk_uuid(j["judgment_id"], idx),
                    references={PARENT_JUDGMENT_REF: judgment_uuid(j["judgment_id"])},
                    vector={VectorName.DEFAULT: vec},
                )
        chunk_batch = client.collections.get(JUDGMENT_CHUNKS_COLLECTION).batch
        failed_c = len(chunk_batch.failed_objects)
        for f in chunk_batch.failed_objects[:3]:
            logger.error("chunk import failed: {}", f.message)
        t_imported = time.perf_counter()
        totals = {
            name: client.collections.get(name).aggregate.over_all(total_count=True).total_count
            for name in (JUDGMENTS_COLLECTION, JUDGMENT_CHUNKS_COLLECTION)
        }
    finally:
        client.close()

    stats = {
        "judgments": len(judgments),
        "chunks": len(chunks),
        "chunks_per_judgment": round(len(chunks) / max(len(judgments), 1), 1),
        "with_thesis": len(with_thesis),
        "load_s": round(t_loaded - t_start, 1),
        "chunk_s": round(t_chunked - t_loaded, 1),
        "embed_s": round(t_embedded - t_chunked, 1),
        "embed_passages_per_s": round(
            (len(chunks) + len(with_thesis)) / (t_embedded - t_chunked), 1
        ),
        "import_judgments_s": round(t_judgments - t_embedded, 1),
        "import_chunks_s": round(t_imported - t_judgments, 1),
        "import_chunks_per_s": round(len(chunks) / (t_imported - t_judgments), 1),
        "failed_judgments": failed_j,
        "failed_chunks": failed_c,
        "collection_totals": totals,
    }
    table = Table(title="v2 sample ingest")
    table.add_column("metric")
    table.add_column("value", justify="right")
    for k, v in stats.items():
        table.add_row(k, str(v))
    console.print(table)
    if stats_out:
        stats_out.write_text(json.dumps(stats, indent=2))


if __name__ == "__main__":
    typer.run(main)
