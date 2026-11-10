# Embedding model and vector quantization for Polish judgments

This page explains which embedding model, chunk size, dimensionality, Weaviate
quantization and search mode to use for the v2 Weaviate stack, based on a
measured evaluation on our own judgments (October 2026, branch
`feat/embedding-eval`, code in `scripts/embedding_eval/`).

## Evaluation setup

- **Task.** Thesis → judgment retrieval. The query is the `thesis` of a judgment
  from `JuDDGES/pl-court-raw`, and the only relevant document is that judgment.
- **Scoring.** Chunks are scored by cosine similarity, and a judgment takes the
  score of its best chunk (MaxP).
- **Metrics.** nDCG@10, MRR@10 and Recall@1/10/100. The 95% CIs come from a paired
  bootstrap (1000 resamples, seed 42).
- **Query types.** Queries are split into `verbatim` and `abstractive`. A query is
  `verbatim` when the thesis appears literally in the text; that is about 6% of
  queries.
- **Iterations.** The corpus grows in six nested iterations, 0 through 5. Each
  iteration adds 1000 queries and 20k judgments, from 20k judgments / 1000
  queries up to 120k judgments / 6000 queries.
  - A config is pruned when the upper bound of the 95% CI of its nDCG@10 gap to
    the best config falls below -0.03.
  - `bgem3_c512` (BGE-M3) and the legacy `qwen06_d128_c2048` are always kept as reference
    configs.
- **Chunking.** Chunks are split with the BGE-M3 tokenizer:
  - `c512` is 480 tokens with 64 overlap.
  - `c2048` is 2048 tokens with 256 overlap, the same as the legacy prod config.
- **Weaviate.** The Weaviate runs use 1.40.2 with HNSW (ef_construction 128,
  max_connections 32, dynamic ef). Each run compares fp32, RQ8, RQ4 and RQ1
  vectors, and also BM25 and hybrid search with relativeScoreFusion.

Raw results are kept as CSVs under `data/experiments/embedding_eval/results/`.
`scripts/embedding_eval/export_tables.py` rebuilds the paper tables and figures
from them.

## Results

### Dense models (exact search, nDCG@10)

The "cumulative" columns use all queries up to that iteration, evaluated against
the growing corpus.

| Config | 20k judgments | 60k | 120k | Δ vs best @120k [95% CI] | Recall@100 @120k |
|---|---|---|---|---|---|
| **sdadas/mmlw-retrieval-e5-large, c512** | 0.469 | 0.385 | **0.341** | — | 0.605 |
| Qwen3-Embedding-4B, c512, 2560d | 0.462 | 0.382 | 0.337 | -0.004 [-0.012, +0.003] | 0.623 |
| Qwen3-Embedding-4B, c512, MRL 1024d | 0.455 | 0.373 | 0.326 | -0.015 [-0.023, -0.007] | 0.615 |
| Qwen3-Embedding-4B, c512, MRL 512d | 0.436 | 0.356 | 0.314 | -0.027 [-0.035, -0.019] | 0.599 |
| BAAI/bge-m3, c512 | 0.449 | 0.360 | 0.313 | -0.028 [-0.035, -0.022] | 0.579 |
| BAAI/bge-m3, c512, truncated 512d | 0.432 | 0.346 | 0.304 | pruned at 120k | 0.566 |
| sdadas/mmlw-retrieval-roberta-large-v2, c512 | 0.416 | pruned at 20k | | | |
| Qwen3-Embedding-0.6B, c512 | 0.388 | pruned at 20k | | | |
| BAAI/bge-m3, c2048 | 0.285 | pruned at 20k | | | |
| Qwen3-Embedding-0.6B, c2048, 128d (legacy prod) | 0.216 | 0.152 | 0.121 | -0.220 | 0.346 |

What these numbers show:

- **The top two models are tied.** mmlw-e5-large and Qwen3-4B are statistically
  tied at every corpus size.
- **Qwen3-4B costs far more to serve.** It embeds about 9x slower: 14 chunks/s on
  an RTX 3090 with TEI, against 125 chunks/s for mmlw-e5. Its vectors also have
  2.5x more dimensions.
- **BGE-M3 is consistently behind.** It trails the best model by about 0.028.
- **512-token chunks beat 2048-token chunks.** The gap is about 0.16 nDCG@10 for
  every model. The legacy prod config is the worst one tested.
- **BGE-M3 does not truncate well.** It is not Matryoshka-trained, and cutting it to
  512 dims costs 0.01-0.02.
- **Qwen3-4B truncates well.** With MRL it loses only 0.01 at 1024 dims.
- **Quality falls as the corpus grows, but the fall slows.** Take the 1000
  iteration-0 queries and evaluate them against the growing corpus:
  - mmlw-e5 scores 0.469, 0.413, 0.389, 0.370, 0.355 and 0.343 at 20k through
    120k judgments.
  - The step-to-step drops shrink: -0.056, -0.024, -0.019, -0.015 and -0.012.
  - The model ranking does not change.
  - Absolute numbers on the full 3M corpus will be lower again.

### Quantization (Weaviate 1.40.2, 120k judgments, 2.2M chunks)

| Vector index | bge-m3 | mmlw-e5 | Qwen3-4B | ANN recall@100 (mmlw-e5) |
|---|---|---|---|---|
| exact (brute force) | 0.313 | 0.341 | 0.337 | 1.000 |
| HNSW fp32 | 0.307 | 0.338 | 0.335 | 0.977 |
| HNSW RQ8 | 0.308 | 0.338 | 0.335 | 0.977 |
| **HNSW RQ4** | 0.308 | **0.339** | 0.335 | **0.986** |
| HNSW RQ1 | 0.307 | 0.333 | 0.333 | 0.870 |

- **RQ8 and RQ4 cost nothing.** Both are within 0.003 of fp32 HNSW in every
  iteration. RQ4 has the higher ANN recall, which is likely an effect of its
  rescoring.
- **Most of the loss against exact search comes from HNSW.** HNSW itself costs
  about 0.003-0.006, and quantization adds almost nothing on top.
- **RQ1 is unsafe for 1024-dim models.** On mmlw-e5 it drops ANN recall to 0.87. It
  holds up only on the 2560-dim Qwen3-4B.
- **Dense queries are fast.** p95 latency for a single dense query is 15-25 ms at
  2.2M chunks.

### BM25 and hybrid search (120k judgments, nDCG@10)

| Search | bge-m3 | mmlw-e5 | Qwen3-4B |
|---|---|---|---|
| dense (RQ8) | 0.308 | 0.338 | 0.335 |
| BM25 | 0.496 | 0.496 | 0.496 |
| hybrid α=0.25 | 0.494 | 0.496 | 0.501 |
| hybrid α=0.5 | 0.463 | 0.472 | 0.484 |
| hybrid α=0.75 | 0.354 | 0.383 | 0.387 |

- **BM25 beats dense retrieval by about +0.16.** It wins on both query types, with
  verbatim queries benefiting most.
- **Hybrid adds only a little.** Low-α hybrid barely moves nDCG over BM25 alone, but
  it raises Recall@100 from 0.697 to 0.706-0.712.
- **Higher α is worse.** Quality falls steadily as α increases.
- **This task favours lexical matching.** Theses are written from the judgment text,
  so these numbers likely overstate the advantage of BM25 for free-form user
  questions.
- **BM25 and hybrid are slow.** Their p95 latency is 300-450 ms at 2.2M chunks.

### Storage for the full corpus (3M judgments)

The figures below are extrapolated to 3M judgments. At about 18 chunks per judgment
with c512, that is about 54M vectors. The figures cover vector data only. The HNSW
graph comes on top, and the uncompressed vectors stay on disk for rescoring.

| Config | fp32 | RQ8 | RQ4 | GPU-hours to embed |
|---|---|---|---|---|
| mmlw-e5 / bge-m3 / Qwen3-4B@1024, c512 | 223 GB | 57 GB | **29 GB** | 138 / 120 / 1057 |
| Qwen3-4B, 2560d, c512 | 558 GB | 140 GB | 71 GB | 1057 |
| legacy Qwen3-0.6B@128, c2048 | 7 GB | 2 GB | 1 GB | 401 |

## Decision

| Choice | Value | Why |
|---|---|---|
| Model | `sdadas/mmlw-retrieval-e5-large` (1024d) | Best or tied-best quality at every corpus size, BGE-M3 speed, Apache-2.0. Prompts: `query:` / `passage:` followed by a space. |
| Chunking | 480 tokens, 64 overlap | +0.16 nDCG@10 over 2048-token chunks, and it fits the model's 512-token context. |
| Dimensions | full 1024 | The model has no MRL. RQ4 saves more space than truncation would, at no quality cost. |
| Quantization | RQ4 | No measurable loss against RQ8, at half the RAM: about 29 GB for 54M vectors. |
| Search | hybrid, α ≈ 0.25 | Matches BM25 on nDCG@10, gives the best Recall@100, and keeps semantic matching for questions that are not thesis-like. |

- **Fallback model.** Qwen3-Embedding-4B at 1024 dims (MRL) reaches nearly the
  same quality. It is the option if a longer context or instruction-following
  becomes important, at about 8x the embedding cost.
- **BGE-M3.** It remains acceptable, at -0.028 nDCG@10, but it is no longer the
  recommended default.

### Known limitations

- **Query set.** Theses only, with one relevant judgment per query. User questions
  will be shorter and less lexical.
- **Corpus scale.** The evaluated corpus reaches only 120k judgments, against 3M in
  production. The trends are stable, but absolute scores will be lower.
- **HNSW is not deterministic.** Repeated Weaviate runs differ by up to about 0.003
  nDCG@10.
