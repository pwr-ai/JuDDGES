import numpy as np
import pandas as pd
import pytest

from scripts.embedding_eval.evaluate_exact import (
    bootstrap_ci,
    maxp_update,
    per_query_metrics,
    storage_bytes,
    target_ranks,
)
from scripts.embedding_eval.evaluate_weaviate import (
    LONG_COLS,
    NO_RANK,
    WeaviateEvalConfig,
    ann_recall,
    build_rows,
    collection_name,
    exact_search,
    latency_summary,
    merge_topk,
    parse_models,
    polish_stopwords,
    ram_gb,
    ranks_from_hits,
    top_configs,
    vector_summary,
)


def test_parse_models():
    assert parse_models(("bgem3:c512", "qwen06:c2048")) == [("bgem3", "c512"), ("qwen06", "c2048")]
    with pytest.raises(ValueError):
        parse_models(("bgem3",))


def test_top_configs_wide_skips_mrl_slices_and_baseline():
    df = pd.DataFrame(
        {
            "variant": [
                "qwen06_d128_c2048",
                "mmlw_e5_c512",
                "bgem3_c512",
                "qwen06_c512",
                "x_c2048",
            ],
            "ndcg10": [0.9, 0.8, 0.7, 0.6, 0.5],
        }
    )
    assert top_configs(df, 2, "bgem3_c512") == ["mmlw_e5_c512", "qwen06_c512"]


def test_top_configs_long():
    df = pd.DataFrame(
        {
            "variant": ["a_c512", "b_c512", "a_c512"],
            "metric": ["ndcg10", "ndcg10", "ndcg10"],
            "view": ["cumulative", "cumulative", "fixed"],
            "value": [0.5, 0.6, 0.9],
        }
    )
    assert top_configs(df, 1, "none") == ["b_c512"]


def test_merge_topk_keeps_best():
    top_s = np.array([[0.9, 0.1]], dtype=np.float32)
    top_i = np.array([[7, 8]])
    s, i = merge_topk(
        top_s, top_i, np.array([[0.5, 0.95]], dtype=np.float32), np.array([[1, 2]]), 2
    )
    assert set(i[0].tolist()) == {2, 7}
    assert sorted(s[0].tolist()) == pytest.approx([0.9, 0.95])


def test_exact_search_matches_dense_reference():
    rng = np.random.default_rng(0)
    chunks = rng.normal(size=(50, 8)).astype(np.float32)
    chunks /= np.linalg.norm(chunks, axis=1, keepdims=True)
    q = rng.normal(size=(3, 8)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    jidx = np.repeat(np.arange(10), 5)
    scores, top = exact_search(q, chunks, jidx, 10, 5, block=12)  # block != multiple of 50
    sims = chunks @ q.T
    ref = np.full((10, 3), -np.inf, dtype=np.float32)
    maxp_update(ref, sims, jidx)
    np.testing.assert_allclose(scores, ref, rtol=1e-5)
    for qi in range(3):
        assert set(top[qi].tolist()) == set(np.argsort(-sims[:, qi])[:5].tolist())


def test_ann_recall():
    exact = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    r = ann_recall([[1, 2, 9, 10], [8, 7, 6, 5]], exact, 4)
    assert r.tolist() == [0.5, 1.0]
    # ANN list longer than k is truncated; shorter list lowers recall
    assert ann_recall([[9, 1, 2]], exact[:1], 2).tolist() == [0.5]
    assert ann_recall([[1]], exact[:1], 4).tolist() == [0.25]


def test_ranks_from_hits_maxp_and_missing():
    hits = [
        [("a", 0, 0.5), ("b", 0, 0.9), ("a", 1, 0.95), ("c", 0, 0.1)],
        [("a", 0, 0.5)],
        [("x", 0, 0.3), ("y", 0, 0.3)],
    ]
    ranks = ranks_from_hits(hits, ["a", "zzz", "y"])
    # q0: MaxP a=0.95 > b=0.9 -> rank 1; q1: absent; q2: tie does not count against
    assert ranks.tolist() == [1, NO_RANK, 1]
    pq = per_query_metrics(ranks)
    assert pq["recall100"].tolist() == [1, 0, 1]
    assert pq["ndcg10"][1] == 0.0


def test_ranks_from_hits_agree_with_target_ranks():
    scores = np.array([[0.2, 0.9], [0.8, 0.1], [0.5, 0.5]])  # [judgment, query]
    hits = [[(f"j{j}", 0, float(scores[j, q])) for j in range(3)] for q in range(2)]
    exp = target_ranks(scores, np.array([2, 0]))
    assert ranks_from_hits(hits, ["j2", "j0"]).tolist() == exp.tolist()


def test_latency_summary_percentiles():
    out = latency_summary(np.arange(1, 101, dtype=float))
    assert out["p50_ms"] == pytest.approx(50.5)
    assert out["p95_ms"] == pytest.approx(95.05)
    assert out["p99_ms"] == pytest.approx(99.01)


def test_bootstrap_ci_brackets_mean():
    x = np.random.default_rng(1).random(200)
    lo, hi = bootstrap_ci(x, 200, 42)
    assert lo < x.mean() < hi


def test_build_rows_long_format():
    pq = per_query_metrics(np.array([1, 3, 20, NO_RANK]))
    exact_pq = per_query_metrics(np.array([1, 1, 5, 150]))
    kinds = np.array(["verbatim", "verbatim", "abstractive", "abstractive"])
    meta = {"iter": 0, "n_judgments": 20, "n_queries": 4}
    rows = build_rows(meta, "m_c512|rq8", "cumulative", pq, kinds, exact_pq, 50, 42)
    df = pd.DataFrame(rows)
    assert list(df.columns) == LONG_COLS or set(LONG_COLS) == set(df.columns)
    metrics = set(df["metric"])
    assert {
        "ndcg10",
        "verbatim_mrr10",
        "abstractive_recall100",
        "ndcg10_delta",
        "recall10_delta",
    } <= metrics
    r = df[df["metric"] == "recall1"].iloc[0]
    assert r["value"] == 0.25 and r["ci_lo"] <= r["value"] <= r["ci_hi"]
    d = df[df["metric"] == "recall10_delta"].iloc[0]
    assert d["value"] == pytest.approx(0.5 - 0.75)  # weaviate recall@10 0.5, exact 0.75


def test_build_rows_without_exact_has_no_deltas():
    pq = per_query_metrics(np.array([1, 2]))
    rows = build_rows(
        {"iter": 0, "n_judgments": 1, "n_queries": 2},
        "v",
        "fixed",
        pq,
        np.array(["verbatim", "abstractive"]),
        None,
        20,
        1,
    )
    assert not any(r["metric"].endswith("_delta") for r in rows)


def test_ram_gb_matches_storage_bytes():
    out = ram_gb(1024, 18.0, 3_000_000)
    n = 18.0 * 3_000_000
    assert out["fp32"] == pytest.approx(4096 * n / 1e9)
    assert out["rq8"] == pytest.approx(storage_bytes(1024)["rq8"] * n / 1e9)
    assert out["rq1"] < out["rq4"] < out["rq8"] < out["fp32"]


def test_collection_name_is_valid():
    assert collection_name("mmlw_e5", "c512", 3) == "EmbEval_mmlw_e5_c512_i3"


def test_polish_stopwords_parsed_from_prod_module():
    words = polish_stopwords()
    assert "oraz" in words or "ale" in words
    assert len(words) > 20


def test_vector_summary_reads_schema_fields():
    ic = {
        "distance": "cosine",
        "efConstruction": 128,
        "maxConnections": 32,
        "ef": -1,
        "dynamicEfMin": 100,
        "dynamicEfMax": 500,
        "dynamicEfFactor": 8,
        "rq": {"enabled": True, "bits": 4},
        "bq": {"enabled": False},
        "pq": {"enabled": False},
        "sq": {"enabled": False},
        "skipDefaultQuantization": False,
    }
    s = vector_summary(
        {"vectorConfig": {"rq4": {"vectorIndexType": "hnsw", "vectorIndexConfig": ic}}}
    )
    assert s["rq4"]["rq"]["bits"] == 4 and s["rq4"]["rq"]["enabled"] is True
    assert s["rq4"]["skip_default_quantization"] is False


def test_config_env_override(monkeypatch):
    from scripts.embedding_eval.common import env_override

    monkeypatch.setenv("EVAL_WEAVIATE_MODELS", "bgem3:c512, qwen06:c512")
    monkeypatch.setenv("EVAL_KEEP_COLLECTIONS", "1")
    monkeypatch.setenv("EVAL_ITER", "2")
    cfg = env_override(WeaviateEvalConfig(), "EVAL_")
    assert cfg.weaviate_models == ("bgem3:c512", "qwen06:c512")
    assert cfg.keep_collections is True and cfg.iter == 2
