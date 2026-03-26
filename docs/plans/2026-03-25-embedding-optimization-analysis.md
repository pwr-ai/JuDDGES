# Embedding Optimization Analysis Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Determine optimal embedding model, dimensions, quantization, and chunking strategy for a legal document system at scale (3.2M docs, 37.8M chunks, growing to 100M+), and design a synthetic data pipeline for fine-tuning a domain-specific retrieval encoder.

**Architecture:** BM25-dominant hybrid search where vectors serve as fallback for edge-case queries. Minimize vector storage (target 50-100x reduction from current 260 GB) while maintaining acceptable hybrid search quality.

**Tech Stack:** Weaviate 1.30+, sentence-transformers, Matryoshka embeddings, Binary Quantization, Claude/Sonnet for synthetic data generation

---

## Part 1: Embedding Model Analysis

### 1.1 Current State

| Parameter | Current Value | Problem |
|---|---|---|
| Model | `sdadas/mmlw-roberta-large` | 1024-dim, no Matryoshka, no BQ training |
| Chunk size | 512 chars / 128 overlap | Too small = 37.8M chunks for 3.2M docs (~12 chunks/doc) |
| Named vectors | 3 (base/dev/fast) | Only base used, 3x overhead |
| Quantization | None (float32) | 4 KB per vector |
| HNSW | ef=128, max_conn=32 | Tuned for high precision, overkill for hybrid fallback |
| Total vector storage | ~260 GB | Dominates Weaviate footprint |

### 1.2 Model Candidates

Ranked by suitability for this use case (hybrid search fallback, Polish+English legal, self-hosted):

#### Tier 1: Best fit (small, multilingual, Matryoshka)

| Model | Params | Native dim | Matryoshka dims | Max tokens | Polish | License |
|---|---|---|---|---|---|---|
| **Qwen3-Embedding-0.6B** | 600M | 1024 | 32-1024 | 32K | 100+ langs | Apache 2.0 |
| **multilingual-e5-small** | 118M | 384 | via fine-tune to 128/64 | 512 | 100 langs | MIT |
| **EmbeddingGemma** | 300M | 768 | 128-768 | 2048 | 100+ langs | Gemma license |

#### Tier 2: High quality but heavier

| Model | Params | Native dim | Matryoshka dims | Max tokens | Polish | License |
|---|---|---|---|---|---|---|
| **Qwen3-Embedding-4B** | 4B | 2560 | 32-2560 | 32K | 100+ langs | Apache 2.0 |
| **BGE-M3** | 568M | 1024 | No native | 8192 | 100+ langs | MIT |
| **sdadas/stella-pl-retrieval** | ~350M | 1024 | No | 512 | Polish-optimized | ? |

#### Tier 3: Polish-specific (no Matryoshka)

| Model | Params | Native dim | Max tokens | Notes |
|---|---|---|---|---|
| **sdadas/mmlw-retrieval-roberta-large-v2** | 355M | 1024 | 512 | Best Polish retrieval, no Matryoshka |
| **Silver Retriever** | ~125M | 768 | 512 | Good Polish, no Matryoshka |

### 1.3 Recommendation

**Primary: `Qwen3-Embedding-0.6B`** truncated to **128 dimensions** + **Binary Quantization**

Rationale:
- 0.6B params = reasonable for self-hosted GPU inference (fits single GPU easily)
- Native Matryoshka to 32-1024 dims — truncate to 128 without quality loss
- 32K token context — can embed entire legal documents without chunking for LegalDocuments collection
- 100+ languages including Polish
- Apache 2.0 — no license restrictions
- MTEB multilingual #1 family — the 0.6B variant is sufficient for hybrid fallback

**Fallback: `multilingual-e5-small`** if Qwen3-0.6B is too heavy or Polish quality insufficient.

### 1.4 Dimension vs Quality Trade-off

Based on published research (Matryoshka papers, QAMA, Vespa blog):

| Dimensions | Storage/vector (BQ) | Recall@10 loss vs full dim | Use case |
|---|---|---|---|
| 1024 (full) | 128 bytes | 0% | Overkill for hybrid fallback |
| 512 | 64 bytes | ~1-2% | Still too large |
| 256 | 32 bytes | ~2-4% | Good balance |
| **128** | **16 bytes** | **~4-6%** | **Sweet spot for hybrid fallback** |
| 64 | 8 bytes | ~8-12% | Aggressive, may hurt edge cases |
| 32 | 4 bytes | ~15-25% | Too lossy |

**128 dims + BQ = 16 bytes per vector** vs current 4096 bytes = **256x reduction per vector**.

### 1.5 Chunking Strategy Revision

| Parameter | Current | Proposed | Rationale |
|---|---|---|---|
| Chunk size | 512 chars | 2048 chars | ~4x fewer chunks, BM25 handles precision |
| Chunk overlap | 128 chars | 256 chars | Proportional overlap |
| Max chunks/doc | 16 | 8 | Fewer, larger, denser chunks |
| Min chunk | 50 chars | 200 chars | Skip tiny fragments |

**Impact on chunk count**: 37.8M → ~9.5M chunks (4x reduction)

Combined with 256x vector reduction: **~1000x total storage reduction**.

### 1.6 Storage Projection

| Component | Current | After optimization |
|---|---|---|
| DocumentChunks vectors | 37.8M × 4 KB = 145 GB | 9.5M × 16 B = 0.14 GB |
| LegalDocuments vectors | 3.2M × 4 KB = 12 GB | 3.2M × 16 B = 0.05 GB |
| HNSW index | ~100 GB | ~2-4 GB (fewer nodes, smaller vectors) |
| **Total vectors + index** | **~260 GB** | **~3-5 GB** |
| **Reduction factor** | | **~60-85x** |

---

## Part 2: Synthetic Data for Fine-Tuning Retrieval Encoders

### 2.1 Why Fine-Tune?

Even with Qwen3-Embedding-0.6B being SOTA multilingual, it has no legal domain knowledge. Fine-tuning on domain-specific query-document pairs can:
- Improve Polish legal terminology understanding
- Boost retrieval for domain-specific queries (case numbers, legal concepts)
- Compensate for Matryoshka truncation quality loss at 128 dims

### 2.2 Synthetic Data Generation Approaches

Based on InPars, Promptagator, GPL, and E5-Mistral research:

#### Approach A: InPars-style (query generation from documents)

```
Input:  Legal document (or chunk)
LLM:    Claude Sonnet / Opus
Output: 3-5 synthetic queries that this document would answer
```

**Prompt template:**
```
Given this Polish legal document excerpt, generate 3 diverse search
queries (in Polish) that a lawyer or citizen might use to find this
document. Include:
1. A natural language question
2. A keyword-style query
3. A conceptual/legal-principle query

Document: {document_text}
```

**Scale needed**: 50K-200K query-document pairs (based on InPars-v2 and E5-Mistral findings)
**Cost estimate** (Claude Sonnet, ~1K tokens input + 200 tokens output per doc):
- 100K documents × ~1.2K tokens = ~120M tokens
- At $3/M input + $15/M output: ~$360 + ~$300 = **~$660**

#### Approach B: GPL-style (with cross-encoder pseudo-labels)

```
1. Generate queries from documents (like Approach A)
2. Retrieve candidates using BM25
3. Score with cross-encoder (e.g., ms-marco-MiniLM-L-6-v2 or legal-specific)
4. Use scores as soft labels for contrastive training
```

**Advantage**: Hard negatives improve embedding quality significantly.
**Extra cost**: Cross-encoder inference on ~1M pairs, but this is cheap (~hours on GPU).

#### Approach C: E5-Mistral style (task-diverse synthetic data)

```
1. Define 50-100 retrieval task types for legal domain:
   - "Find judgments about X"
   - "Find precedents for Y"
   - "Find tax interpretations regarding Z"
2. For each task, generate query-document pairs with Claude
3. Include negative examples
```

**Advantage**: Most diverse, best generalization.
**Cost**: Higher prompt engineering upfront, but similar LLM cost.

### 2.3 Recommended Pipeline

**Phase 1: Quick baseline (1-2 days, ~$200)**
1. Sample 30K documents from Weaviate (diverse: judgments, tax interpretations, both languages)
2. Use Claude Sonnet to generate 3 queries per document = **90K pairs**
3. Fine-tune Qwen3-Embedding-0.6B with Matryoshka loss at [128, 256, 512] dims
4. Evaluate on held-out set

**Phase 2: Hard negatives (1 day, ~$100)**
5. For each query, retrieve top-50 with BM25
6. Score with cross-encoder
7. Re-train with hard negatives (InfoNCE loss + Matryoshka)

**Phase 3: Scale up if needed (2-3 days, ~$500)**
8. Expand to 100K documents, 5 queries each = 500K pairs
9. Add task-type diversity (Approach C)
10. Final fine-tune with full pipeline

### 2.4 How Many Examples Are Actually Needed?

Based on literature:

| Dataset size | Expected quality | Source |
|---|---|---|
| 10K pairs | Basic domain adaptation, ~5% NDCG improvement | GPL paper |
| 50K pairs | Solid improvement, ~8-12% NDCG improvement | InPars-v2 |
| 100K-200K pairs | Diminishing returns start | Promptagator |
| 500K+ pairs | Marginal gains (<1% more) | E5-Mistral (used 1.8M but across 100 langs) |

**For single-domain (legal), single-language (Polish+English): 50K-100K pairs is the sweet spot.**

E5-Mistral showed that <1K training steps suffice for LLM-based encoders. For a 0.6B model, more data helps but 50K-100K is likely sufficient for domain adaptation.

### 2.5 Fine-Tuning Recipe

```python
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.training_args import SentenceTransformerTrainingArgs

model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

# Matryoshka + contrastive loss
loss = losses.MatryoshkaLoss(
    model=model,
    loss=losses.MultipleNegativesRankingLoss(model),
    matryoshka_dims=[128, 256, 512, 1024],
)

args = SentenceTransformerTrainingArgs(
    output_dir="models/legal-embedding-0.6b",
    num_train_epochs=3,
    per_device_train_batch_size=32,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    bf16=True,
)

# Train
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    loss=loss,
)
trainer.train()
```

### 2.6 Evaluation Plan

**Metrics**: NDCG@10, Recall@10, MRR@10 on hybrid search

**Test sets**:
1. **Manual gold set**: 200 hand-curated query-document pairs from legal experts
2. **Synthetic hold-out**: 5K pairs from Phase 1 not used in training
3. **A/B on live Weaviate**: Compare hybrid search quality old vs new

**Comparison matrix**:
| Config | Dims | Quant | Chunks | Storage | NDCG@10 |
|---|---|---|---|---|---|
| Baseline (current) | 1024 | f32 | 512 char | 260 GB | ? |
| Qwen3-0.6B raw | 128 | BQ | 2048 char | ~3 GB | ? |
| Qwen3-0.6B fine-tuned | 128 | BQ | 2048 char | ~3 GB | ? |
| Qwen3-0.6B fine-tuned | 256 | BQ | 2048 char | ~5 GB | ? |

---

## Part 3: Execution Roadmap

### Week 1: Model selection & baseline
- [ ] Deploy Qwen3-Embedding-0.6B locally, test inference speed
- [ ] Run PL-MTEB retrieval benchmark on raw model at dims [64, 128, 256, 512]
- [ ] Compare with current mmlw-roberta-large on same benchmark
- [ ] Decision: confirm model choice or pivot

### Week 2: Synthetic data generation
- [ ] Sample 30K diverse documents from Weaviate
- [ ] Design prompt templates (3 query types per document)
- [ ] Generate 90K query-document pairs with Claude Sonnet
- [ ] Quality check: manually review 200 random pairs
- [ ] Generate hard negatives via BM25 + cross-encoder

### Week 3: Fine-tuning & evaluation
- [ ] Fine-tune Qwen3-0.6B with Matryoshka loss
- [ ] Evaluate at dims [128, 256] on synthetic hold-out
- [ ] Build manual gold test set (200 pairs) with domain expert
- [ ] Compare configs in evaluation matrix
- [ ] Decision: confirm final dims + quantization

### Week 4: Migration
- [ ] Update docker-compose: new transformer service
- [ ] Update chunking config: 2048 char / 256 overlap / max 8 chunks
- [ ] Create new Weaviate collections with BQ + single vector
- [ ] Re-ingest all documents
- [ ] Validate search quality on production data
- [ ] Delete old collections

---

## References

- [MTEB Benchmark](https://github.com/embeddings-benchmark/mteb)
- [MMTEB: Massive Multilingual Text Embedding Benchmark](https://arxiv.org/abs/2502.13595)
- [Qwen3-Embedding](https://qwenlm.github.io/blog/qwen3-embedding/)
- [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
- [Best Embedding Models for RAG 2026](https://blog.premai.io/best-embedding-models-for-rag-2026-ranked-by-mteb-score-cost-and-self-hosting/)
- [Best Open-Source Embedding Models 2026](https://www.bentoml.com/blog/a-guide-to-open-source-embedding-models)
- [InPars+: Supercharging Synthetic Data Generation for IR](https://arxiv.org/html/2508.13930v1)
- [Improving Text Embeddings with LLMs (E5-Mistral)](https://arxiv.org/html/2401.00368v2)
- [InPars-v2](https://arxiv.org/abs/2301.01820)
- [Promptagator-Style Dense Retriever Training](https://arxiv.org/pdf/2510.02241)
- [Fine-Tuning Embeddings for RAG with Synthetic Data](https://medium.com/llamaindex-blog/fine-tuning-embeddings-for-rag-with-synthetic-data-e534409a3971)
- [Scaling Vector Search: Quantization + Matryoshka for 80% Cost Reduction](https://towardsdatascience.com/649627-2/)
- [Quantization Aware Matryoshka Adaptation (QAMA)](https://dl.acm.org/doi/10.1145/3746252.3761077)
- [Matryoshka + Binary vectors (Vespa)](https://blog.vespa.ai/combining-matryoshka-with-binary-quantization-using-embedder/)
- [Binary and Scalar Embedding Quantization (HuggingFace)](https://huggingface.co/blog/embedding-quantization)
- [Matryoshka Quantization](https://arxiv.org/abs/2502.06786)
- [Weaviate Binary Quantization](https://docs.weaviate.io/weaviate/configuration/compression/bq-compression)
- [Weaviate 32x Reduction with BQ](https://weaviate.io/blog/binary-quantization)
- [Late Chunking](https://arxiv.org/html/2409.04701v2)
- [Weaviate Late Chunking Blog](https://weaviate.io/blog/late-chunking)
- [PL-MTEB: Polish Massive Text Embedding Benchmark](https://arxiv.org/html/2405.10138v1)
- [Polish Information Retrieval Benchmark (PIRB)](https://huggingface.co/spaces/sdadas/pirb)
- [sdadas/mmlw-retrieval-roberta-large-v2](https://huggingface.co/sdadas/mmlw-retrieval-roberta-large-v2)
- [sdadas/stella-pl-retrieval](https://huggingface.co/sdadas/stella-pl-retrieval)
- [Silver Retriever](https://aclanthology.org/2024.lrec-main.1291/)
- [EmbeddingGemma](https://developers.googleblog.com/introducing-embeddinggemma/)
- [Voyage-3-large](https://blog.voyageai.com/2025/01/07/voyage-3-large/)
