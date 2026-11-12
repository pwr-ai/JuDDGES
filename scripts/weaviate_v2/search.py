"""Query the v2 collections, or run the thesis -> judgment eval on what is ingested.

    python scripts/weaviate_v2/search.py query "nieważność umowy kredytu frankowego"
    python scripts/weaviate_v2/search.py eval --n 300

`eval` uses each ingested judgment's thesis as the query and its own judgment as the only
relevant result (the task of docs/explanation/embedding-and-quantization.md), searching
`JudgmentChunks` only: the thesis vector on `Judgments` would leak the answer.
"""

import json
import os
import random
import statistics
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from juddges.data.weaviate_v2.ingest import TeiEmbedder
from juddges.data.weaviate_v2.schema import (
    HYBRID_ALPHA,
    JUDGMENTS_COLLECTION,
    connect_local,
    judgment_uuid,
)
from juddges.data.weaviate_v2.search import search_judgments

app = typer.Typer(add_completion=False)
console = Console()


def _client(host: str, port: int, grpc_port: int):
    return connect_local(host, port, grpc_port, api_key=os.environ["WEAVIATE_V2_API_KEY"])


@app.command()
def query(
    text: str,
    k: int = typer.Option(10),
    alpha: float = typer.Option(HYBRID_ALPHA, help="0 = BM25 only, 1 = vector only"),
    tei_url: str = typer.Option("http://127.0.0.1:8111"),
    host: str = typer.Option("localhost"),
    port: int = typer.Option(8110),
    grpc_port: int = typer.Option(50061),
) -> None:
    embedder = TeiEmbedder(tei_url)
    client = _client(host, port, grpc_port)
    try:
        t0 = time.perf_counter()
        vector = embedder.query(text)
        t1 = time.perf_counter()
        hits = search_judgments(client, text, vector, k=k, alpha=alpha)
        t2 = time.perf_counter()
        judgments = client.collections.get(JUDGMENTS_COLLECTION)
        table = Table(title=f"{text!r} (alpha={alpha})")
        for col in ("#", "score", "signature", "court", "date", "best chunk"):
            table.add_column(col)
        for rank, hit in enumerate(hits, 1):
            obj = judgments.query.fetch_object_by_id(
                judgment_uuid(hit.judgment_id),
                return_properties=["document_number", "court_name", "date_issued"],
            )
            p = obj.properties if obj else {}
            table.add_row(
                str(rank),
                f"{hit.score:.3f}",
                str(p.get("document_number", "")),
                str(p.get("court_name", "")),
                str(p.get("date_issued", ""))[:10],
                hit.best_chunk.chunk_text[:120].replace("\n", " "),
            )
        console.print(table)
        console.print(f"embed {1000 * (t1 - t0):.0f} ms, search {1000 * (t2 - t1):.0f} ms")
    finally:
        client.close()
        embedder.close()


@app.command("eval")
def evaluate(
    n: int = typer.Option(300, help="queries (judgments with a thesis)"),
    k: int = typer.Option(10),
    alphas: str = typer.Option("0,0.25,0.5,1", help="comma-separated hybrid alphas"),
    seed: int = typer.Option(42),
    tei_url: str = typer.Option("http://127.0.0.1:8111"),
    host: str = typer.Option("localhost"),
    port: int = typer.Option(8110),
    grpc_port: int = typer.Option(50061),
    out: Path | None = typer.Option(None, help="write results as JSON"),
) -> None:
    embedder = TeiEmbedder(tei_url)
    client = _client(host, port, grpc_port)
    try:
        judgments = client.collections.get(JUDGMENTS_COLLECTION)
        pairs = [
            (o.properties["judgment_id"], o.properties["thesis"])
            for o in judgments.iterator(return_properties=["judgment_id", "thesis"])
            if o.properties.get("thesis")
        ]
        pairs = random.Random(seed).sample(sorted(pairs), min(n, len(pairs)))
        vectors = [embedder.query(thesis) for _, thesis in pairs]
        results = []
        for alpha in (float(a) for a in alphas.split(",")):
            ranks, latencies = [], []
            for (jid, thesis), vec in zip(pairs, vectors):
                t0 = time.perf_counter()
                hits = search_judgments(client, thesis, vec, k=100, alpha=alpha)
                latencies.append(1000 * (time.perf_counter() - t0))
                ids = [h.judgment_id for h in hits]
                ranks.append(ids.index(jid) + 1 if jid in ids else None)
            found = [r for r in ranks if r is not None]
            results.append(
                {
                    "alpha": alpha,
                    "queries": len(ranks),
                    "recall@1": round(sum(r == 1 for r in found) / len(ranks), 3),
                    f"recall@{k}": round(sum(r <= k for r in found) / len(ranks), 3),
                    "recall@100": round(len(found) / len(ranks), 3),
                    f"mrr@{k}": round(sum(1 / r for r in found if r <= k) / len(ranks), 3),
                    "p50_ms": round(statistics.median(latencies), 1),
                    "p95_ms": round(sorted(latencies)[int(0.95 * len(latencies)) - 1], 1),
                }
            )
        total = judgments.aggregate.over_all(total_count=True).total_count
        table = Table(title=f"thesis -> judgment, {len(pairs)} queries over {total} judgments")
        for col in results[0]:
            table.add_column(col, justify="right")
        for row in results:
            table.add_row(*(str(v) for v in row.values()))
        console.print(table)
        if out:
            out.write_text(json.dumps(results, indent=2))
    finally:
        client.close()
        embedder.close()


if __name__ == "__main__":
    app()
