"""T1: build the eval set (queries, judgments, chunks) from JuDDGES/pl-court-raw.

Run: python -m scripts.embedding_eval.build_dataset   (configure with EVAL_* env vars)
"""

import glob
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq
from loguru import logger
from rich.console import Console
from rich.table import Table

from scripts.embedding_eval.common import DEFAULT_OUT_DIR, HF_COURT_RAW_GLOB, env_override, out_dir

console = Console()


@dataclass
class DatasetConfig:
    parquet_glob: str = HF_COURT_RAW_GLOB
    out_dir: str = DEFAULT_OUT_DIR
    n_queries: int = 1000
    n_distractors: int = 19000
    thesis_min_chars: int = 50
    thesis_max_chars: int = 1500
    seed: int = 42
    tokenizer: str = "BAAI/bge-m3"


# name -> (chunk_size tokens, overlap tokens)
CHUNKINGS: dict[str, tuple[int, int]] = {"c512": (480, 64), "c2048": (2048, 256)}


def normalise(text: str) -> str:
    """Whitespace- and case-normalise text."""
    return " ".join(text.lower().split())


def tag_kind(thesis: str, full_text: str) -> str:
    """`verbatim` when the normalised thesis is a substring of the normalised full text."""
    return "verbatim" if normalise(thesis) in normalise(full_text) else "abstractive"


def sample_ids(meta: pd.DataFrame, cfg: DatasetConfig) -> tuple[list[str], list[str]]:
    """Pick query ids and distractor ids from a metadata frame.

    `meta` needs columns judgment_id, thesis_len, text_len. Returns (query_ids, distractor_ids),
    disjoint, both in a deterministic (seeded) order.
    """
    rng = np.random.default_rng(cfg.seed)
    has_text = meta["text_len"] > 0
    cand = meta[has_text & meta["thesis_len"].between(cfg.thesis_min_chars, cfg.thesis_max_chars)][
        "judgment_id"
    ].to_numpy()
    if len(cand) < cfg.n_queries:
        raise ValueError(f"only {len(cand)} query candidates, need {cfg.n_queries}")
    query_ids = rng.choice(cand, size=cfg.n_queries, replace=False)

    rest = meta[has_text & ~meta["judgment_id"].isin(set(query_ids))]["judgment_id"].to_numpy()
    if len(rest) < cfg.n_distractors:
        raise ValueError(f"only {len(rest)} distractor candidates, need {cfg.n_distractors}")
    distractor_ids = rng.choice(rest, size=cfg.n_distractors, replace=False)
    return list(map(str, query_ids)), list(map(str, distractor_ids))


def chunk_rows(judgment_id: str, pieces: list[str], n_tokens: list[int]) -> list[dict]:
    """Chunk bookkeeping: stable chunk_uid and chunk_idx per judgment, empty pieces dropped."""
    rows = []
    for text, n in zip(pieces, n_tokens):
        if not text.strip():
            continue
        idx = len(rows)
        rows.append(
            {
                "chunk_uid": f"{judgment_id}#{idx}",
                "judgment_id": judgment_id,
                "chunk_idx": idx,
                "text": text,
                "n_tokens": n,
            }
        )
    return rows


def read_metadata(files: list[str]) -> pd.DataFrame:
    """Cheap first pass: ids, thesis length and full_text length only."""
    parts = []
    for f in files:
        t = pq.read_table(f, columns=["judgment_id", "thesis", "full_text"])
        parts.append(
            pd.DataFrame(
                {
                    "judgment_id": t["judgment_id"].to_pandas().astype(str),
                    "thesis_len": pc.fill_null(pc.utf8_length(t["thesis"]), 0).to_numpy(),
                    "text_len": pc.fill_null(pc.utf8_length(t["full_text"]), 0).to_numpy(),
                }
            )
        )
        logger.info("metadata {}: {} rows", Path(f).name, t.num_rows)
    meta = pd.concat(parts, ignore_index=True)
    dup = meta["judgment_id"].duplicated().sum()
    if dup:
        logger.warning("dropping {} duplicated judgment_id rows", dup)
        meta = meta.drop_duplicates("judgment_id")
    return meta


def load_selected(files: list[str], ids: set[str]) -> pd.DataFrame:
    parts = []
    cols = ["judgment_id", "thesis", "full_text", "court_type", "judgment_type", "judgment_date"]
    for f in files:
        df = pq.read_table(f, columns=cols).to_pandas()
        df["judgment_id"] = df["judgment_id"].astype(str)
        parts.append(df[df["judgment_id"].isin(ids)])
    return pd.concat(parts, ignore_index=True).drop_duplicates("judgment_id")


def build_chunks(judgments: pd.DataFrame, tokenizer, size: int, overlap: int) -> pd.DataFrame:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer, chunk_size=size, chunk_overlap=overlap
    )
    rows: list[dict] = []
    for i, (jid, text) in enumerate(zip(judgments["judgment_id"], judgments["full_text"])):
        pieces = splitter.split_text(text)
        lens = [len(x) for x in tokenizer(pieces, add_special_tokens=False)["input_ids"]]
        rows.extend(chunk_rows(jid, pieces, lens))
        if (i + 1) % 2000 == 0:
            logger.info("chunked {}/{} judgments ({} chunks)", i + 1, len(judgments), len(rows))
    return pd.DataFrame(rows)


def run(cfg: DatasetConfig | None = None) -> None:
    cfg = env_override(cfg or DatasetConfig(), "EVAL_")
    out = out_dir(cfg.out_dir)
    files = sorted(glob.glob(cfg.parquet_glob))
    if not files:
        raise FileNotFoundError(cfg.parquet_glob)
    logger.info("{} parquet files, config={}", len(files), cfg)

    meta = read_metadata(files)
    logger.info(
        "{} judgments, {} with thesis, {} with text",
        len(meta),
        int((meta["thesis_len"] > 0).sum()),
        int((meta["text_len"] > 0).sum()),
    )
    query_ids, distractor_ids = sample_ids(meta, cfg)
    selected = load_selected(files, set(query_ids) | set(distractor_ids))

    # keep a deterministic order: queries first, then distractors
    order = {j: i for i, j in enumerate(query_ids + distractor_ids)}
    judgments = (
        selected.assign(_o=selected["judgment_id"].map(order)).sort_values("_o").drop(columns="_o")
    )
    judgments = judgments.reset_index(drop=True)
    judgments["is_query"] = judgments["judgment_id"].isin(set(query_ids))

    q = judgments[judgments["is_query"]].reset_index(drop=True)
    queries = pd.DataFrame(
        {
            "qid": [f"q{i:04d}" for i in range(len(q))],
            "judgment_id": q["judgment_id"],
            "text": q["thesis"],
            "kind": [tag_kind(t, f) for t, f in zip(q["thesis"], q["full_text"])],
        }
    )
    missing = set(queries["judgment_id"]) - set(judgments["judgment_id"])
    assert not missing, f"query judgments missing from corpus: {len(missing)}"
    queries.to_parquet(out / "queries.parquet", index=False)
    judgments.to_parquet(out / "judgments.parquet", index=False)

    table = Table(title="queries per kind")
    table.add_column("kind")
    table.add_column("n", justify="right")
    for kind, n in queries["kind"].value_counts().items():
        table.add_row(str(kind), str(n))
    console.print(table)
    console.print(
        f"judgments: {len(judgments)} (queries {len(queries)}, distractors {len(judgments) - len(queries)})"
    )

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.tokenizer)
    stats = Table(title="chunking stats (bge-m3 tokens)")
    for c in [
        "chunking",
        "chunks",
        "judgments",
        "chunks/judg",
        "min",
        "mean",
        "p50",
        "p95",
        "p99",
        "max",
    ]:
        stats.add_column(c, justify="right")
    for name, (size, overlap) in CHUNKINGS.items():
        chunks = build_chunks(judgments, tok, size, overlap)
        chunks.to_parquet(out / f"chunks_{name}.parquet", index=False)
        n = chunks["n_tokens"]
        stats.add_row(
            name,
            str(len(chunks)),
            str(chunks["judgment_id"].nunique()),
            f"{len(chunks) / chunks['judgment_id'].nunique():.2f}",
            str(n.min()),
            f"{n.mean():.1f}",
            str(int(n.quantile(0.5))),
            str(int(n.quantile(0.95))),
            str(int(n.quantile(0.99))),
            str(n.max()),
        )
        logger.info("wrote chunks_{}.parquet ({} chunks)", name, len(chunks))
    console.print(stats)


if __name__ == "__main__":
    run()
