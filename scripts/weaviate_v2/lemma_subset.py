"""Build an `Exp*` chunk collection for the lemmatization experiment (never touches prod data).

    python scripts/weaviate_v2/lemma_subset.py plan --eval-set eval_set.json \
        --runs full_concepts.json --out subset_ids.json
    python scripts/weaviate_v2/lemma_subset.py copy --ids subset_ids.json --collection ExpChunksLemma

`plan` picks judgments: up to `--per-concept` labelled ones per concept, every judgment any
full-corpus run returned in its top 100 (the distractors production search actually meets;
`expand` runs pool the forms a lemma field would match), the thesis-sample judgments and a
random background read in uuid order from a random start.

`copy` creates the collection (same vector index and BM25 config as `JudgmentChunks`) and
copies the chunks of those judgments with their stored vectors (no re-embedding), adding one
`chunk_text_lemma_<backend>` property per lemmatizer backend.
"""

import json
import os
import random
import time
import uuid as uuidlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import typer
from loguru import logger

import weaviate.classes.config as wvcc
from juddges.data.weaviate_v2.lemma import BACKENDS, Lemmatizer
from juddges.data.weaviate_v2.schema import (
    JUDGMENT_CHUNKS_COLLECTION,
    JUDGMENTS_COLLECTION,
    SchemaConfig,
    _inverted_index,
    _vector,
    connect_local,
)
from juddges.settings import VectorName
from weaviate.classes.query import Filter

app = typer.Typer(add_completion=False)

FETCH_LIMIT = 10_000
_LEMMATIZERS: dict[str, Lemmatizer] = {}


def _lemmatize(texts: list[str]) -> list[dict[str, str]]:
    if not _LEMMATIZERS:
        _LEMMATIZERS.update({b: Lemmatizer(b) for b in BACKENDS})
    return [{f"chunk_text_lemma_{b}": lem(t) for b, lem in _LEMMATIZERS.items()} for t in texts]


def _client():
    return connect_local(api_key=os.environ["WEAVIATE_V2_API_KEY"])


@app.command()
def plan(
    eval_set: Path = typer.Option(...),
    runs: list[Path] = typer.Option(..., help="results JSON(s) of lemma_eval.py run"),
    thesis_sample: Path | None = typer.Option(None),
    out: Path = typer.Option(...),
    per_concept: int = typer.Option(300),
    background: int = typer.Option(30_000),
    seed: int = typer.Option(13),
) -> None:
    rng = random.Random(seed)
    concepts = json.loads(eval_set.read_text())
    labelled = set()
    for c in concepts.values():
        rel = sorted(c["relevant"])
        labelled.update(rng.sample(rel, min(per_concept, len(rel))))
    pooled = set()
    for path in runs:
        for ranked in json.loads(path.read_text())["runs"].values():
            for ids in ranked.values():
                pooled.update(ids)
    theses = {jid for jid, _ in json.loads(thesis_sample.read_text())} if thesis_sample else set()
    client = _client()
    bg: set[str] = set()
    try:
        judgments = client.collections.get(JUDGMENTS_COLLECTION)
        start = uuidlib.UUID(int=rng.getrandbits(128))
        for o in judgments.iterator(return_properties=["judgment_id"], after=start):
            bg.add(o.properties["judgment_id"])
            if len(bg) >= background:
                break
    finally:
        client.close()
    ids = sorted(labelled | pooled | theses | bg)
    stats = {
        "labelled": len(labelled),
        "pooled": len(pooled),
        "theses": len(theses),
        "background": len(bg),
        "total": len(ids),
    }
    logger.info("subset: {}", stats)
    out.write_text(json.dumps(ids))
    out.with_suffix(".stats.json").write_text(json.dumps(stats, indent=1))


def create_exp_collection(client, name: str, shards: int) -> None:
    if not name.startswith("Exp"):
        raise ValueError("experiment collections must be named Exp*")
    if client.collections.exists(name):
        logger.info("collection {} exists, appending", name)
        return
    text = lambda n: wvcc.Property(  # noqa: E731
        name=n,
        data_type=wvcc.DataType.TEXT,
        index_filterable=False,
        index_searchable=True,
        tokenization=wvcc.Tokenization.WORD,
    )
    config = SchemaConfig()
    client.collections.create(
        name=name,
        description="Lemmatization experiment: subset of JudgmentChunks with lemma fields",
        properties=[
            text("chunk_text"),
            *(text(f"chunk_text_lemma_{b}") for b in BACKENDS),
            wvcc.Property(
                name="judgment_id",
                data_type=wvcc.DataType.TEXT,
                index_filterable=True,
                index_searchable=False,
                tokenization=wvcc.Tokenization.FIELD,
            ),
            wvcc.Property(name="chunk_idx", data_type=wvcc.DataType.INT),
        ],
        vector_config=[_vector(VectorName.DEFAULT, config)],
        inverted_index_config=_inverted_index(config),
        sharding_config=wvcc.Configure.sharding(desired_count=shards),
    )


@app.command()
def copy(
    ids: Path = typer.Option(...),
    collection: str = typer.Option("ExpChunksLemma"),
    shards: int = typer.Option(2),
    judgments_per_fetch: int = typer.Option(200),
    workers: int = typer.Option(16),
) -> None:
    judgment_ids: list[str] = json.loads(ids.read_text())
    client = _client()
    try:
        create_exp_collection(client, collection, shards)
        source = client.collections.get(JUDGMENT_CHUNKS_COLLECTION)
        target = client.collections.get(collection)
        copied, t0 = 0, time.perf_counter()
        with (
            ProcessPoolExecutor(workers) as pool,
            target.batch.fixed_size(batch_size=500, concurrent_requests=4) as batch,
        ):
            groups = [
                judgment_ids[i : i + judgments_per_fetch]
                for i in range(0, len(judgment_ids), judgments_per_fetch)
            ]
            pending = []
            while groups:
                group = groups.pop()
                response = source.query.fetch_objects(
                    filters=Filter.by_property("judgment_id").contains_any(group),
                    limit=FETCH_LIMIT,
                    include_vector=True,
                    return_properties=["chunk_text", "judgment_id", "chunk_idx"],
                )
                if len(response.objects) >= FETCH_LIMIT:  # truncated: split and refetch
                    half = len(group) // 2
                    groups += [group[:half], group[half:]]
                    continue
                objs = response.objects
                pending.append(
                    (objs, pool.submit(_lemmatize, [o.properties["chunk_text"] for o in objs]))
                )
                while pending and (len(pending) > workers or not groups):
                    objs, fut = pending.pop(0)
                    for o, lemmas in zip(objs, fut.result()):
                        batch.add_object(
                            properties={**o.properties, **lemmas},
                            uuid=o.uuid,
                            vector={VectorName.DEFAULT: o.vector[VectorName.DEFAULT]},
                        )
                    copied += len(objs)
                logger.info(
                    "{} chunks copied, {:.0f}/s, {} groups left",
                    copied,
                    copied / (time.perf_counter() - t0),
                    len(groups),
                )
        failed = target.batch.failed_objects
        if failed:
            raise RuntimeError(f"{len(failed)} objects failed, first: {failed[0].message}")
        total = target.aggregate.over_all(total_count=True).total_count
        logger.info("done: {} chunks copied, collection has {}", copied, total)
    finally:
        client.close()


if __name__ == "__main__":
    app()
