# Embedding model and vector quantization for Polish judgments

This page explains why the v2 Weaviate stack uses BGE-M3 with 1024 dimensions and
RQ 8-bit quantization, and what should be measured before changing either. It
reflects the state as of October 2026.

> **Status: preliminary.** The choices below are based on published benchmarks. They
> will be updated with measured results from the pre-ingest evaluation on our own
> judgments (branch `feat/embedding-eval`).

## Model choice

The table shows Polish retrieval quality as the average NDCG@10 over the 41 Polish
tasks of PIRB (Dadas et al., arXiv 2609.12913, September 2026):

| Model | PIRB NDCG@10 | Params | Dims | Max context | MRL | License |
|---|---|---|---|---|---|---|
| Qwen3-Embedding-8B | 60.93 | 8B | 4096 | 32K | ? | Apache-2.0 |
| sdadas/mmlw-retrieval-roberta-large-v2 | 60.71 | 0.4B | 1024 | 512 | no | gemma |
| Qwen3-Embedding-4B | 59.19 | 4B | 2560 | 32K | yes | Apache-2.0 |
| sdadas/mmlw-retrieval-e5-large | 58.30 | 0.6B | 1024 | 512 | no | Apache-2.0 |
| jina-embeddings-v3 | 57.33 | ~0.57B | 1024 | 8192 | yes | CC-BY-NC-4.0 |
| multilingual-e5-large | 57.29 | 0.56B | 1024 | 512 | no | MIT |
| **BAAI/bge-m3** | **55.75** | 0.57B | 1024 | 8192 | no | MIT |
| EmbeddingGemma-300m | 55.22 | 0.3B | 768 | 2048 | yes | Gemma |
| Qwen3-Embedding-0.6B (legacy stack) | 50.71 | 0.6B | 1024 | 32K | yes | Apache-2.0 |

BGE-M3 is the default because it is the strongest model that meets all four of
these requirements:

- **8K context.** Chunks are 500-1000 tokens, and the mmlw and e5 models truncate at 512.
- **Permissive license.** MIT, unlike jina-embeddings-v3.
- **Native TEI support.** BGE-M3 is XLM-RoBERTa. The plain-RoBERTa mmlw models are
  not on the TEI support list.
- **Cheap to serve at 33M vectors.** Qwen3-4B and Qwen3-8B cost 7-14x more compute
  per vector.

BGE-M3 is +5 NDCG@10 over the Qwen3-Embedding-0.6B model the legacy stack used.

No public benchmark reports BGE-M3, Qwen3 or mmlw on _Polish court judgments_. The
PIRB legal subsets (PolEval-2022 legal QA, e-Prawnik, Specprawnik) rank models
differently from the overall average. Candidates should therefore be validated on
our own data, see [Validation before full ingest](#validation-before-full-ingest).

## Dimensionality reduction

BGE-M3 is **not** Matryoshka-trained. Truncating it is not a supported way to save
space: reported cosine similarity to the full vector is only ~0.71 at 512 dims. If
fewer dimensions are needed, switch to an MRL model such as Qwen3-Embedding-4B or
EmbeddingGemma. For reference, EmbeddingGemma loses 0.4 / 1.5 / 2.9 MTEB points at
512 / 256 / 128 dims.

## Quantization (Weaviate 1.40)

| Method | Vector size vs float32 | Recall (Weaviate docs) | Training |
|---|---|---|---|
| RQ 8-bit | 4x smaller | 98-99% | none |
| RQ 4-bit (new in 1.40) | ~8x smaller | 94-95%, 99.8%+ with rescoring | none |
| RQ 1-bit / BQ | ~32x smaller | moderate, needs large rescore (default 512) | none |
| PQ | ~24x smaller | tunable trade-off | 10k-100k objects/shard |
| SQ | 4x smaller | better than BQ | 100k objects/shard |

The table below estimates RAM for 33M vectors (3M documents + ~30M chunks) at 1024
dims. The figures cover vector data only. The HNSW graph adds roughly 17 GB in
every variant.

| Variant | Vectors in RAM |
|---|---|
| float32 | ~135 GB |
| **RQ 8-bit** | **~34 GB** |
| RQ 4-bit | ~17 GB |
| RQ 1-bit | ~4.5 GB |

**Decision: RQ 8-bit.** RQ 8-bit fits comfortably on the host (251 GB RAM) at
near-lossless recall, with no training step. RQ 4-bit is the next step if memory
becomes tight, but only after it is measured on our queries, because legal retrieval
is precision-sensitive. In every variant the uncompressed vectors stay on disk
(~135 GB) for rescoring.

## Validation before full ingest

Run these steps on a sample of roughly 1-2M chunks before the full ingest:

1. Build ~200 gold queries from judgments, for example thesis → judgment or legal
   question → reasons section.
2. Compare BGE-M3 1024d with `mmlw-retrieval-e5-large` (with chunks ≤512 tokens) and
   `Qwen3-Embedding-4B` at 512d via MRL. Report NDCG@10 and Recall@100, using hybrid
   search with the production `alpha`.
3. For the winning model, compare RQ8 with RQ4 at the same `rescoreLimit`.

Keep BGE-M3 + RQ8 unless an alternative wins by a margin that justifies its serving
cost.
