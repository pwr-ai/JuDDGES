<!-- cspell:words kredytowa kredytowej umowna kary umownej umownych umowne umowny zawodowi zawód umowć -->

# Polish lemmatization for BM25 in the v2 search

This page explains whether the v2 Weaviate search should index lemmas next to `chunk_text`,
and what the measurements say (November 2026, full v2 corpus: 2.69M judgments, 30.7M chunks;
code in `scripts/weaviate_v2/lemma_eval.py`, `lemma_subset.py` and
`juddges/data/weaviate_v2/lemma.py`).

## The problem

`JudgmentChunks.chunk_text` is indexed with `word` tokenization and no stemming, because
Weaviate has no Polish analyzer. An inflected query shares no term with another form of the
same phrase: "umowa kredytowa" and "umowy kredytowej" are different BM25 queries. Search runs
hybrid at alpha 0.25 (75% BM25), so the BM25 side decides most of the ranking.

Measured on the full corpus, the top 10 judgments of two forms of the same concept overlap by
only 2.5% at alpha 0 and 0.25. The overlap is 75% with the dense vector alone.

## Evaluation setup

- **Concept queries.** There are 72 legal concepts, each written in 3 forms: nominative, genitive
  and one other case or number (216 short queries, e.g. "kara umowna" / "kary umownej" /
  "kar umownych").
- **Labels.** Labels are weak and come without LLM calls. A judgment is relevant when its
  `keywords` (portal tags) contain a whole tag equal, case-insensitively, to the concept's tag,
  for example `Kary umowne`.
  - Each concept has 209 to 3867 relevant judgments.
  - 54 concepts come from ordinary courts and 18 from administrative courts.
- **Metrics.**
  - nDCG@10 and capped Recall@100 (relevant in the top 100 / min(relevant, 100)).
  - overlap@10 and overlap@50: mean |A ∩ B| / k over pairs of forms of one concept.
  - Thesis → judgment on 200 sampled theses (R@1/10/100, MRR@10).
  - Latency per Weaviate query, with 100 chunks as in production.
- **Search.** Hybrid on chunks with relativeScore fusion, then MaxP to judgments. The dense side
  always embeds the raw query; only the BM25 text and fields change.
- **Variants.**
  - `plain`: `chunk_text`, the raw query (production).
  - `simplemma` / `morfeusz_all`: a lemma field, with the query lemmatized by the same function.
  - `both_*`: `chunk_text` plus the lemma field weighted ^2.
  - `expand`: `chunk_text` with the query expanded to every inflected form (Morfeusz2
    `generate`), which needs no reindex.
- **Subset.** The lemma variants need a reindex, so they run on an experiment collection
  `ExpChunksLemma`, built with vectors copied and not re-embedded. It holds 1.67M chunks of
  79.9k judgments:
  - up to 300 labelled judgments per concept;
  - every judgment that a full-corpus `plain` or `expand` run returned in its top 100, which
    covers the forms a lemma field matches;
  - the 200 thesis judgments;
  - 30k random judgments.
  Labels are intersected with the subset.

## Lemmatizers

Throughput is per process on real chunks, with an LRU token cache.

| Lemmatizer | tokens/s | Install | Short-query quality |
|---|---|---|---|
| simplemma 2.0.0 | ~1.0M | pure Python, 19 MB | one lemma per token, mostly right: "kar umownych" -> "kara umowny" |
| Morfeusz2 1.99.15 + SGJP, all lemmas | ~0.57M | wheel with the dictionary, 26 MB | no tagger, so indexes every reading: "kary" -> kary, kar, kara |
| Morfeusz2, one lemma by POS priority | ~0.56M | as above | rejected: "kary umownej" -> "kary", "zawodowi" -> "zawód" |
| spaCy `pl_core_news_sm` / `lg` | 17-19k | 20 MB / 550 MB model plus spaCy | 50x slower; on queries without context "umownej" -> "umowć" (sm) |

## Results

### Full corpus, production field (`chunk_text`)

| config | nDCG@10 | R@100 | overlap@10 | overlap@50 | thesis R@10 | thesis MRR@10 | p95 ms (short / thesis) |
|---|---|---|---|---|---|---|---|
| plain α=0 | 0.265 | 0.268 | 0.025 | 0.052 | 0.590 | 0.490 | 19 / 2349 |
| plain α=0.25 (prod) | 0.264 | 0.270 | 0.025 | 0.058 | 0.595 | 0.492 | 40 / 2574 |
| plain α=0.5 | 0.273 | 0.273 | 0.269 | 0.296 | 0.575 | 0.455 | 42 / 2712 |
| plain α=0.7 | 0.292 | 0.279 | 0.646 | 0.674 | 0.495 | 0.375 | 38 / 2585 |
| plain α=1 | 0.290 | 0.282 | 0.746 | 0.771 | 0.430 | 0.316 | 16 / 48 |
| expand α=0.25 | 0.229 | 0.265 | 0.937 | 0.940 | - | - | 63 / - |
| expand α=0.5 | 0.257 | 0.274 | 0.810 | 0.831 | - | - | 64 / - |

### Subset (`ExpChunksLemma`), all variants

| config | nDCG@10 | R@100 | overlap@10 | overlap@50 | thesis R@10 | thesis MRR@10 | thesis p95 ms |
|---|---|---|---|---|---|---|---|
| plain α=0.25 (prod) | 0.255 | 0.271 | 0.025 | 0.057 | 0.665 | 0.620 | 307 |
| plain α=0.5 | 0.278 | 0.278 | 0.208 | 0.243 | 0.680 | 0.623 | 306 |
| plain α=0.7 | 0.294 | 0.289 | 0.582 | 0.652 | 0.655 | 0.572 | 302 |
| plain α=1 | 0.293 | 0.285 | 0.762 | 0.787 | 0.610 | 0.519 | 21 |
| simplemma α=0 | 0.290 | 0.315 | 0.879 | 0.886 | 0.675 | 0.626 | 494 |
| **simplemma α=0.25** | **0.299** | **0.315** | **0.841** | **0.871** | 0.670 | 0.621 | 455 |
| simplemma α=0.5 | 0.300 | 0.301 | 0.746 | 0.816 | 0.675 | 0.614 | 404 |
| morfeusz_all α=0.25 | 0.293 | 0.317 | 0.869 | 0.888 | 0.675 | 0.616 | 609 |
| morfeusz_all α=0.5 | 0.309 | 0.301 | 0.781 | 0.835 | 0.675 | 0.603 | 640 |
| both_simplemma α=0.25 | 0.310 | 0.316 | 0.259 | 0.389 | 0.685 | 0.619 | 422 |
| both_morfeusz_all α=0.25 | 0.313 | 0.316 | 0.284 | 0.406 | - | - | - |
| expand α=0.5 | 0.274 | 0.278 | 0.825 | 0.848 | - | - | - |

On short queries, p50 is 7-10 ms and p95 is 12-16 ms for every BM25 field.

nDCG@10 per form is shown as nominative / genitive / other:

| config | nominative | genitive | other |
|---|---|---|---|
| plain α=0.25 | 0.242 | 0.312 | 0.210 |
| plain α=0.7 | 0.293 | 0.283 | 0.305 |
| simplemma α=0.25 | 0.297 | 0.298 | 0.303 |
| morfeusz_all α=0.5 | 0.305 | 0.306 | 0.315 |

The table below gives paired differences over the 72 concepts (bootstrap 95% CI).

| comparison | mean ΔnDCG@10 | 95% CI | concepts better / worse |
|---|---|---|---|
| simplemma α=0.25 vs plain α=0.25 | +0.044 | [+0.005, +0.087] | 35 / 33, median 0.000 |
| simplemma α=0.5 vs plain α=0.5 | +0.023 | [-0.008, +0.057] | 35 / 33 |
| simplemma α=0.25 vs plain α=0.7 | +0.006 | [-0.051, +0.063] | 31 / 36, median -0.002 |
| morfeusz_all α=0.5 vs plain α=0.7 | +0.015 | [-0.027, +0.059] | 33 / 34 |
| both_morfeusz_all α=0.25 vs plain α=0.7 | +0.019 | [-0.030, +0.069] | 38 / 31 |
| morfeusz_all α=0.5 vs simplemma α=0.5 | +0.008 | [-0.006, +0.023] | 32 / 28 |

## What the numbers say

- **A lemma field removes the inflection defect.** It raises overlap@10 between forms from
  0.025 to 0.84-0.90 at α ≤ 0.25, and nDCG no longer depends on the form. Recall@100 rises by
  about 0.045. This gain holds by construction: every form becomes the same BM25 text.
- **The ranking gain is small and uneven.** Against production (α=0.25), simplemma adds
  +0.044 nDCG@10, but the median concept does not move (35 better, 33 worse). The mean comes
  from concepts where the plain query form is rare in the text. Against the best plain α, the
  gain is within noise.
- **Dense search already covers most of the gap on short queries.** Plain at α=0.7-1 is as
  good as any lemma config on concept queries. It loses 0.10 thesis R@10 on the full corpus,
  though, so it is not a free alpha change.
- **Query expansion is worse than both.** Expanding the query to every form fixes overlap but
  costs nDCG: the rare inflected forms get high IDF and dominate.
- **Indexing both fields scores best, but brings the form sensitivity back.** It reaches the
  best nDCG (0.31), yet overlap falls back to 0.26-0.28.
- **Lemma BM25 is slower on long queries.** Thesis p95 rises from 307 to 455 ms on the subset,
  because lemma posting lists are longer. On the full corpus, thesis p95 is already about 2.5 s.

## Decision

The gate was written down before the subset was run: the best lemma config must beat the best
plain config by ≥ 0.03 nDCG@10, raise overlap@10 by ≥ 0.10, and lose no more than 0.02 thesis
R@10. It **failed** on the first clause: the best lemma config beats the best plain config by
only +0.019, and the CI includes 0.

| Choice | Value | Why |
|---|---|---|
| Lemma field in production | **not now** | The gate failed. The robustness gain is real, but the ranking gain against tuned plain search is within noise. A rollout needs a 30.7M-object re-put and a schema property that cannot be removed. |
| Alpha | **keep 0.25** | α=0.7 gains +0.028 concept nDCG on the full corpus but loses 0.10 thesis R@10. α=0.5 trades +0.01 for -0.02. |
| Lemmatizer, if adopted later | simplemma 2.0.0, α=0.25 | 2x faster than Morfeusz2, pure Python, one lemma per token (smaller index, lower thesis latency). The gap to `morfeusz_all` is not significant. |

### Known limitations

- **Weak labels.** Portal tags miss many relevant judgments. Absolute scores are low for every
  config, so the comparisons are relative.
- **Easy thesis distractors on the subset.** The subset was not pooled with thesis
  distractors, so plain α=0.25 reaches R@10 0.665 there against 0.595 on the full corpus.
  "Thesis-neutral" holds on that easier set only.
- **Subset ranking.** Lemma variants were only measured on the subset. Their full-corpus nDCG
  is extrapolated from the matched plain runs, which agree within 0.01 between the full corpus
  and the subset.

## Reproducing

All steps run in the v2 stack's uv container, with
`--with morfeusz2==1.99.15,simplemma==2.0.0` added to the usual `--with` list.

1. **Eval set.** Concepts and forms are in `scripts/weaviate_v2/lemma_concepts.json`. Build the
   labels with `scripts/weaviate_v2/lemma_eval.py labels --out eval_set.json`.
2. **Full-corpus baselines.** Run the concept and thesis evals:
   - `scripts/weaviate_v2/lemma_eval.py run --eval-set eval_set.json --out full.json --configs "plain@0,plain@0.25,..."`
   - `scripts/weaviate_v2/lemma_eval.py theses --out theses.json --sample-path thesis_sample.json`
3. **Experiment collection.** Plan the subset, then copy it, then run the same two commands with
   `--collection ExpChunksLemma`, plus `--restrict-to subset_ids.json` for `run`:
   - `scripts/weaviate_v2/lemma_subset.py plan ...`
   - `scripts/weaviate_v2/lemma_subset.py copy --ids subset_ids.json`
