import numpy as np
import pytest

from scripts.embedding_eval.evaluate_exact import (
    ExactConfig,
    maxp_update,
    mrl_slice,
    paired_bootstrap,
    per_query_metrics,
    score_variant,
    storage_bytes,
    summarise,
    target_ranks,
)


def test_per_query_metrics_known_ranks():
    pq = per_query_metrics(np.array([1, 2, 10, 11, 100, 101]))
    assert pq["recall1"].tolist() == [1, 0, 0, 0, 0, 0]
    assert pq["recall10"].tolist() == [1, 1, 1, 0, 0, 0]
    assert pq["recall100"].tolist() == [1, 1, 1, 1, 1, 0]
    assert pq["mrr10"][:4] == pytest.approx([1.0, 0.5, 0.1, 0.0])
    assert pq["ndcg10"][:4] == pytest.approx([1.0, 1 / np.log2(3), 1 / np.log2(11), 0.0])


def test_target_ranks_ties_count_against():
    scores = np.array([[0.9, 0.1], [0.5, 0.1], [0.1, 0.1]])  # [n_judg, n_q]
    assert target_ranks(scores, np.array([1, 0])).tolist() == [2, 1]
    assert target_ranks(scores, np.array([2, 2])).tolist() == [3, 1]


def test_maxp_update_takes_max_per_judgment_unsorted():
    sims = np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.3], [0.7, 0.1]])
    jidx = np.array([1, 0, 1, 0])
    scores = np.full((3, 2), -np.inf)
    maxp_update(scores, sims, jidx)
    assert scores[0].tolist() == [0.8, 0.2]
    assert scores[1].tolist() == [0.3, 0.9]
    assert np.isneginf(scores[2]).all()
    # second block merges with the running max
    maxp_update(scores, np.array([[0.95, 0.0]]), np.array([1]))
    assert scores[1].tolist() == [0.95, 0.9]


def test_score_variant_blocks_and_mrl_match_direct():
    rng = np.random.default_rng(0)
    chunks = rng.normal(size=(40, 8)).astype(np.float32)
    chunks /= np.linalg.norm(chunks, axis=1, keepdims=True)
    q = chunks[[3, 17]] + 0.01
    jidx = np.repeat(np.arange(10), 4)
    cfg = ExactConfig(block_size=7)
    for dim in (8, 4):
        got = score_variant(q, chunks, jidx, 10, dim, cfg)
        c, qq = mrl_slice(chunks, dim, 8), mrl_slice(q, dim, 8)
        want = (c @ qq.T).reshape(10, 4, 2).max(axis=1)
        assert got == pytest.approx(want, abs=1e-5)
    ranks = target_ranks(score_variant(q, chunks, jidx, 10, 8, cfg), np.array([0, 4]))
    assert ranks.tolist() == [1, 1]


def test_mrl_slice_renormalises():
    x = np.array([[3.0, 4.0, 12.0]])
    s = mrl_slice(x, 2, 3)
    assert np.linalg.norm(s) == pytest.approx(1.0)
    assert mrl_slice(x, 3, 3) is x


def test_summarise_splits_by_kind():
    pq = per_query_metrics(np.array([1, 11, 1, 1]))
    kinds = np.array(["verbatim", "verbatim", "abstractive", "abstractive"])
    s = summarise(pq, kinds)
    assert s["recall10"] == 0.75
    assert s["verbatim_recall10"] == 0.5
    assert s["abstractive_recall10"] == 1.0


def test_paired_bootstrap_detects_shift_and_is_deterministic():
    rng = np.random.default_rng(1)
    b = rng.uniform(size=300)
    a = b + 0.2
    d, lo, hi = paired_bootstrap(a, b, 200, 42)
    assert d == pytest.approx(0.2) and lo > 0 and hi < 0.21
    assert paired_bootstrap(a, b, 200, 42) == (d, lo, hi)
    _, lo0, hi0 = paired_bootstrap(b, b, 200, 42)
    assert lo0 == hi0 == 0.0


def test_storage_bytes():
    assert storage_bytes(128) == {"fp32": 512, "rq8": 144, "rq4": 80, "rq1": 24}
