"""Stream a whole dataset (pl-court-raw or pl-nsa) into the v2 collections, resumably.

Reads the parquet files from the HF cache in units of `--unit-rows` rows of one row group,
chunks in a process pool, embeds with TEI (several requests in flight) and imports with one
Weaviate batch per unit. Finished units go to a JSON checkpoint; re-running the same command
skips them, and units that failed (exception or failed objects) are retried. Pipeline details:
`juddges/data/weaviate_v2/stream.py`.

    python scripts/weaviate_v2/ingest_full.py --dataset pl-court-raw --limit-units 2   # smoke
    python scripts/weaviate_v2/ingest_full.py --dataset pl-court-raw \\
        --log-file .context/weaviate_v2/ingest_pl-court-raw.log                        # full
"""

import multiprocessing as mp
import os
import sys
from enum import Enum
from pathlib import Path

import typer
from loguru import logger
from rich.console import Console
from rich.table import Table

from juddges.data.weaviate_v2.ingest import TeiEmbedder
from juddges.data.weaviate_v2.schema import (
    JUDGMENT_CHUNKS_COLLECTION,
    JUDGMENTS_COLLECTION,
    connect_local,
    create_collections,
)
from juddges.data.weaviate_v2.stream import (
    DATASETS,
    Checkpoint,
    WeaviateWriter,
    init_chunk_worker,
    iter_units,
    parquet_files,
    plan_units,
    run_ingest,
    split_text,
)

console = Console()
Dataset = Enum("Dataset", {name: name for name in DATASETS}, type=str)


def main(
    dataset: Dataset = typer.Option(..., help="dataset to ingest"),
    unit_rows: int = typer.Option(1000, help="rows per unit (checkpoint granularity)"),
    limit_units: int | None = typer.Option(None, help="stop after N units (smoke runs)"),
    checkpoint: Path | None = typer.Option(
        None, help="checkpoint JSON [default: .context/weaviate_v2/ingest_<dataset>.json]"
    ),
    log_file: Path | None = typer.Option(None, help="also log to this file"),
    workers: int = typer.Option(6, help="chunking processes"),
    embed_batch: int = typer.Option(128, help="passages per TEI request"),
    embed_concurrency: int = typer.Option(4, help="TEI requests in flight"),
    weaviate_batch: int = typer.Option(500, help="objects per Weaviate batch request"),
    weaviate_concurrency: int = typer.Option(4, help="Weaviate batch requests in flight"),
    tei_url: str = typer.Option("http://127.0.0.1:8111"),
    host: str = typer.Option("localhost"),
    port: int = typer.Option(8110),
    grpc_port: int = typer.Option(50061),
    download: bool = typer.Option(False, help="download missing parquet files from the Hub"),
) -> None:
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.add(log_file, enqueue=True)
    name = dataset.value
    spec = DATASETS[name]
    ckpt = Checkpoint(
        checkpoint or Path(f".context/weaviate_v2/ingest_{name}.json"), name, unit_rows
    )
    units = plan_units(parquet_files(spec.repo_id, local_files_only=not download), unit_rows)
    pending = [u for u in units if not ckpt.is_done(u.key)]
    logger.info(
        "{}: {} units ({} rows), {} done, {} pending, {} previously failed; checkpoint {}",
        name,
        len(units),
        sum(u.num_rows for u in units),
        len(units) - len(pending),
        len(pending),
        len(ckpt.failed),
        ckpt.path,
    )

    embedder = TeiEmbedder(tei_url, batch_size=embed_batch)
    client = connect_local(host, port, grpc_port, api_key=os.environ["WEAVIATE_V2_API_KEY"])
    pool = mp.get_context("spawn").Pool(workers, initializer=init_chunk_worker)
    try:
        create_collections(client)
        stats = run_ingest(
            iter_units(pending, spec.columns, skip=lambda u: ckpt.is_done(u.key)),
            spec,
            split_async=lambda texts: pool.map_async(split_text, texts, chunksize=8),
            embedder=embedder,
            writer=WeaviateWriter(client, weaviate_batch, weaviate_concurrency),
            checkpoint=ckpt,
            embed_batch=embed_batch,
            embed_concurrency=embed_concurrency,
            limit_units=limit_units,
            units_pending=len(pending),
        )
        totals = {
            c: client.collections.get(c).aggregate.over_all(total_count=True).total_count
            for c in (JUDGMENTS_COLLECTION, JUDGMENT_CHUNKS_COLLECTION)
        }
    finally:
        pool.terminate()
        client.close()
        embedder.close()

    table = Table(title=f"v2 ingest {name}")
    table.add_column("metric")
    table.add_column("value", justify="right")
    rows = {
        "units done (this run)": stats.units_done,
        "units failed (this run)": stats.units_failed,
        "units done (checkpoint)": len(ckpt.done),
        "units total": len(units),
        "judgments": stats.judgments,
        "chunks": stats.chunks,
        "theses": stats.theses,
        "skipped empty text": stats.skipped_empty,
        "seconds": round(stats.seconds, 1),
        "chunks/s": round(stats.chunks / max(stats.seconds, 1e-9), 1),
        **{f"{c} total": n for c, n in totals.items()},
    }
    for k, v in rows.items():
        table.add_row(k, str(v))
    console.print(table)
    if stats.aborted:
        logger.error("aborted after repeated unit failures; fix the cause and re-run to resume")
    if stats.units_failed or stats.aborted:
        sys.exit(1)


if __name__ == "__main__":
    typer.run(main)
