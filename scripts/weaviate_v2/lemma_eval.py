"""Inflection-robustness eval of v2 search: plain vs lemmatized BM25 inside hybrid.

    python scripts/weaviate_v2/lemma_eval.py labels --out eval_set.json
    python scripts/weaviate_v2/lemma_eval.py run --eval-set eval_set.json --out results/full.json \
        --collection JudgmentChunks --configs "plain@0,plain@0.25,expand@0.25"

A config is `<variant>@<alpha>`:
- `plain`: BM25 on `chunk_text` with the raw query (what production does).
- `expand`: BM25 on `chunk_text` with the query expanded to every inflected form (Morfeusz2
  analyse + generate); needs no reindex.
- `simplemma`, `morfeusz_all`: BM25 on `chunk_text_lemma_<backend>`, query lemmatized alike.
- `both_<backend>`: BM25 on `chunk_text` and `chunk_text_lemma_<backend>^2`.
The dense side always gets the raw query's vector.

Eval set (`eval_set.json`, built by `labels` from `lemma_concepts.json`): concept -> {forms,
relevant judgment ids}. A judgment is relevant when one of its `keywords` (portal tags) equals a
concept tag as a whole element, case-insensitively.
Metrics per config: nDCG@10 and capped Recall@100 (|rel in top100| / min(|rel|, 100)) over all
forms, overlap@10/@50 of the judgments returned for different forms of one concept, and
latency. `thesis` runs the thesis -> judgment task for the same configs.
"""

import itertools
import json
import math
import os
import random
import re
import statistics
import time
from collections.abc import Callable, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import typer
from loguru import logger
from rich.console import Console
from rich.table import Table

from juddges.data.documents_weaviate_db import POLISH_STOPWORDS
from juddges.data.weaviate_v2.ingest import TeiEmbedder
from juddges.data.weaviate_v2.lemma import Lemmatizer, tokenize
from juddges.data.weaviate_v2.schema import JUDGMENTS_COLLECTION, connect_local
from juddges.settings import VectorName
from weaviate.classes.query import Filter, HybridFusion, MetadataQuery

app = typer.Typer(add_completion=False)
console = Console()

EVAL_CHUNKS = 400  # chunks per query for ranking; MaxP gives up to that many judgments
PROD_CHUNKS = 100  # chunks per query in production (latency pass)
TOP = 100
CONCEPTS = Path(__file__).with_name("lemma_concepts.json")
FETCH_LIMIT = 10_000  # QUERY_MAXIMUM_RESULTS of the v2 stack
# Weaviate rejects a filter token that is a stopword (EN preset + Polish additions)
_FILTER_STOPWORDS = set(POLISH_STOPWORDS) | {"a", "and", "in", "of", "on", "or", "the", "to"}


def lemma_property(backend: str) -> str:
    return f"chunk_text_lemma_{backend}"


@lru_cache(maxsize=1)
def _generator():
    import morfeusz2

    return morfeusz2.Morfeusz(generate=True, analyse=True, expand_tags=False)


@lru_cache(maxsize=100_000)
def _forms(token: str) -> tuple[str, ...]:
    morf = _generator()
    lemmas = {lemma for _s, _e, (_f, lemma, *_rest) in morf.analyse(token)}
    forms = {token}
    for lemma in lemmas:
        try:
            forms.update(f.lower() for f, *_rest in morf.generate(lemma))
        except RuntimeError:  # lemma unknown to the generator (e.g. "ign")
            continue
    return tuple(sorted(forms))


def expand_query(text: str, lemmatizer: Lemmatizer) -> str:
    tokens = [t for t in tokenize(text) if lemmatizer.lemmas([t])]
    return " ".join(itertools.chain.from_iterable(_forms(t) for t in tokens))


class Variant:
    """How a config turns a query into BM25 text and which properties it searches."""

    def __init__(self, name: str) -> None:
        self.name = name
        if name == "plain":
            self.props, self.transform = ["chunk_text"], lambda q: q
        elif name == "expand":
            lem = Lemmatizer("morfeusz_all")
            self.props, self.transform = ["chunk_text"], lambda q: expand_query(q, lem)
        elif name.startswith("both_"):
            backend = name.removeprefix("both_")
            lem = Lemmatizer(backend)
            self.props = ["chunk_text", f"{lemma_property(backend)}^2"]
            self.transform = lambda q: f"{q} {lem(q)}"
        else:
            lem = Lemmatizer(name)
            self.props, self.transform = [lemma_property(name)], lem


def parse_configs(spec: str) -> list[tuple[str, float]]:
    out = []
    for item in spec.split(","):
        name, alpha = item.strip().split("@")
        out.append((name, float(alpha)))
    return out


def search(
    collection: Any,
    bm25_query: str,
    vector: Sequence[float],
    alpha: float,
    props: list[str],
    chunk_limit: int,
    filters: Any = None,
) -> list[str]:
    """Judgment ids ranked by MaxP over the top `chunk_limit` hybrid chunks."""
    response = collection.query.hybrid(
        query=bm25_query or " ",
        vector=list(vector),
        alpha=alpha,
        query_properties=props,
        target_vector=VectorName.DEFAULT,
        fusion_type=HybridFusion.RELATIVE_SCORE,
        limit=chunk_limit,
        filters=filters,
        return_properties=["judgment_id"],
        return_metadata=MetadataQuery(score=True),
    )
    best: dict[str, float] = {}
    for o in response.objects:
        jid, score = o.properties["judgment_id"], float(o.metadata.score or 0.0)
        if score > best.get(jid, -1.0):
            best[jid] = score
    return sorted(best, key=best.__getitem__, reverse=True)


def ndcg10(ranked: Sequence[str], relevant: set[str]) -> float:
    dcg = sum(1 / math.log2(i + 2) for i, j in enumerate(ranked[:10]) if j in relevant)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(10, len(relevant))))
    return dcg / ideal if ideal else 0.0


def recall100(ranked: Sequence[str], relevant: set[str]) -> float:
    return len(relevant.intersection(ranked[:TOP])) / min(len(relevant), TOP)


def overlap(a: Sequence[str], b: Sequence[str], k: int) -> float:
    return len(set(a[:k]) & set(b[:k])) / k


def percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def summarize(
    eval_set: dict[str, Any], runs: dict[str, list[str]], relevant: Callable[[str], set[str]]
) -> dict[str, float]:
    nd, rc, o10, o50 = [], [], [], []
    for cid, concept in eval_set.items():
        rel = relevant(cid)
        ranked = [runs[form] for form in concept["forms"]]
        nd += [ndcg10(r, rel) for r in ranked]
        rc += [recall100(r, rel) for r in ranked]
        for a, b in itertools.combinations(ranked, 2):
            o10.append(overlap(a, b, 10))
            o50.append(overlap(a, b, 50))
    return {
        "ndcg@10": round(statistics.mean(nd), 4),
        "recall@100": round(statistics.mean(rc), 4),
        "overlap@10": round(statistics.mean(o10), 4),
        "overlap@50": round(statistics.mean(o50), 4),
    }


def _table(title: str, rows: list[dict[str, Any]]) -> None:
    table = Table(title=title)
    for col in rows[0]:
        table.add_column(col, justify="right")
    for row in rows:
        table.add_row(*(str(v) for v in row.values()))
    console.print(table)


def _client():
    return connect_local(api_key=os.environ["WEAVIATE_V2_API_KEY"])


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


@app.command()
def labels(
    out: Path = typer.Option(...),
    concepts_path: Path = typer.Option(CONCEPTS, "--concepts"),
) -> None:
    """Relevance labels from `Judgments.keywords`.

    `keywords` is word-tokenized, so the filter only fetches candidates holding all tag tokens;
    the whole-element match is checked here. A fetch that hits the result cap is an error.
    """
    concepts = json.loads(concepts_path.read_text())
    client = _client()
    eval_set = {}
    try:
        judgments = client.collections.get(JUDGMENTS_COLLECTION)
        for cid, concept in concepts.items():
            relevant: set[str] = set()
            for tag in concept["tags"]:
                tokens = [t for t in tokenize(tag) if t not in _FILTER_STOPWORDS]
                response = judgments.query.fetch_objects(
                    filters=Filter.by_property("keywords").contains_all(tokens),
                    limit=FETCH_LIMIT,
                    return_properties=["judgment_id", "keywords"],
                )
                if len(response.objects) >= FETCH_LIMIT:
                    raise RuntimeError(f"{cid}: tag {tag!r} hits the {FETCH_LIMIT} result cap")
                relevant.update(
                    o.properties["judgment_id"]
                    for o in response.objects
                    if any(_norm(k) == _norm(tag) for k in o.properties.get("keywords") or [])
                )
            eval_set[cid] = {**concept, "n_relevant": len(relevant), "relevant": sorted(relevant)}
            logger.info("{}: {} relevant", cid, len(relevant))
    finally:
        client.close()
    out.write_text(json.dumps(eval_set, ensure_ascii=False))


@app.command()
def run(
    eval_set_path: Path = typer.Option(..., "--eval-set"),
    out: Path = typer.Option(...),
    collection_name: str = typer.Option("JudgmentChunks", "--collection"),
    configs: str = typer.Option("plain@0,plain@0.25,plain@0.5,plain@0.7,plain@1"),
    restrict_to: Path | None = typer.Option(
        None, help="JSON list of judgment ids in the collection; labels are intersected"
    ),
    tei_url: str = typer.Option("http://127.0.0.1:8111"),
    latency: bool = typer.Option(True, help="extra pass with the production chunk limit"),
) -> None:
    """Concept queries: nDCG@10, Recall@100, overlap between forms, latency."""
    eval_set = json.loads(eval_set_path.read_text())
    subset = set(json.loads(restrict_to.read_text())) if restrict_to else None

    def relevant(cid: str) -> set[str]:
        rel = set(eval_set[cid]["relevant"])
        return rel & subset if subset is not None else rel

    forms = sorted({f for c in eval_set.values() for f in c["forms"]})
    embedder = TeiEmbedder(tei_url)
    vectors = {f: embedder.query(f) for f in forms}
    embedder.close()
    client = _client()
    results: dict[str, Any] = {"runs": {}, "metrics": []}
    try:
        collection = client.collections.get(collection_name)
        for name, alpha in parse_configs(configs):
            variant = Variant(name)
            runs, lat, lat_prod = {}, [], []
            for form in forms:
                text = variant.transform(form)
                t0 = time.perf_counter()
                runs[form] = search(
                    collection, text, vectors[form], alpha, variant.props, EVAL_CHUNKS
                )[:TOP]
                lat.append(1000 * (time.perf_counter() - t0))
                if latency:
                    t0 = time.perf_counter()
                    search(collection, text, vectors[form], alpha, variant.props, PROD_CHUNKS)
                    lat_prod.append(1000 * (time.perf_counter() - t0))
            key = f"{name}@{alpha}"
            row = {"config": key, **summarize(eval_set, runs, relevant)}
            if lat_prod:
                row["p50_ms"] = round(statistics.median(lat_prod), 1)
                row["p95_ms"] = round(percentile(lat_prod, 0.95), 1)
            row["p50_ms_400"] = round(statistics.median(lat), 1)
            results["runs"][key] = runs
            results["metrics"].append(row)
            logger.info("{}", row)
            out.write_text(json.dumps(results, ensure_ascii=False))
    finally:
        client.close()
    _table(
        f"{len(eval_set)} concepts, {len(forms)} queries on {collection_name}", results["metrics"]
    )


@app.command()
def theses(
    out: Path = typer.Option(...),
    n: int = typer.Option(200),
    collect: int = typer.Option(3000, help="theses read (uuid order = random) before sampling"),
    seed: int = typer.Option(42),
    sample_path: Path | None = typer.Option(None, help="reuse/save the sampled (id, thesis)"),
    collection_name: str = typer.Option("JudgmentChunks", "--collection"),
    configs: str = typer.Option("plain@0,plain@0.25,plain@0.5,plain@0.7,plain@1"),
    tei_url: str = typer.Option("http://127.0.0.1:8111"),
) -> None:
    """Thesis -> judgment: each thesis must retrieve its own judgment (chunks only)."""
    client = _client()
    try:
        if sample_path and sample_path.exists():
            pairs = json.loads(sample_path.read_text())
        else:
            pairs = []
            judgments = client.collections.get(JUDGMENTS_COLLECTION)
            for o in judgments.iterator(return_properties=["judgment_id", "thesis"]):
                if o.properties.get("thesis"):
                    pairs.append((o.properties["judgment_id"], o.properties["thesis"]))
                    if len(pairs) >= collect:
                        break
            pairs = random.Random(seed).sample(sorted(pairs), min(n, len(pairs)))
            if sample_path:
                sample_path.write_text(json.dumps(pairs, ensure_ascii=False))
        embedder = TeiEmbedder(tei_url)
        vectors = [embedder.query(t) for _, t in pairs]
        embedder.close()
        collection = client.collections.get(collection_name)
        rows = []
        for name, alpha in parse_configs(configs):
            variant = Variant(name)
            ranks, lat = [], []
            for (jid, thesis), vec in zip(pairs, vectors):
                t0 = time.perf_counter()
                ids = search(
                    collection, variant.transform(thesis), vec, alpha, variant.props, EVAL_CHUNKS
                )
                lat.append(1000 * (time.perf_counter() - t0))
                ranks.append(ids.index(jid) + 1 if jid in ids else None)
            found = [r for r in ranks if r is not None]
            row = {
                "config": f"{name}@{alpha}",
                "queries": len(ranks),
                "recall@1": round(sum(r == 1 for r in found) / len(ranks), 3),
                "recall@10": round(sum(r <= 10 for r in found) / len(ranks), 3),
                "recall@100": round(sum(r <= TOP for r in found) / len(ranks), 3),
                "mrr@10": round(sum(1 / r for r in found if r <= 10) / len(ranks), 3),
                "p50_ms": round(statistics.median(lat), 1),
                "p95_ms": round(percentile(lat, 0.95), 1),
            }
            rows.append(row)
            logger.info("{}", row)
            out.write_text(json.dumps(rows, indent=1))
    finally:
        client.close()
    _table(f"thesis -> judgment on {collection_name}", rows)


if __name__ == "__main__":
    app()
