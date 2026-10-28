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


def test_score_variant_multi_part_equals_concatenated():
    from scripts.embedding_eval.evaluate_exact import ExactConfig as C

    rng = np.random.default_rng(1)
    chunks = rng.normal(size=(30, 6)).astype(np.float32)
    chunks /= np.linalg.norm(chunks, axis=1, keepdims=True)
    q = rng.normal(size=(3, 6)).astype(np.float32)
    j = np.repeat(np.arange(10), 3)
    cfg = C(block_size=4)
    whole = score_variant(q, chunks, j, 10, 6, cfg)
    parts = score_variant(q, [chunks[:12], chunks[12:]], [j[:12], j[12:]], 10, 6, cfg)
    assert parts == pytest.approx(whole)


def test_embedding_unit_and_merge_throughput_and_ci():
    from scripts.embedding_eval.evaluate_exact import (
        bootstrap_ci,
        embedding_unit,
        merge_throughput,
    )

    assert embedding_unit("qwen06_d128_c2048") == "qwen06_c2048"
    assert embedding_unit("mmlw_e5_c512") == "mmlw_e5_c512"
    assert embedding_unit("bgem3_d512_c512") == "bgem3_c512"
    m = [
        {"throughput": {"c512": {"n": 100, "seconds": 10.0}}},
        {"throughput": {"c512": {"n": 300, "seconds": 10.0}}},
    ]
    assert merge_throughput(m, "c512")["chunks_per_s"] == pytest.approx(20.0)
    assert merge_throughput([{}], "c512")["chunks_per_s"] is None
    lo, hi = bootstrap_ci(np.linspace(0, 1, 200), 200, 0)
    assert lo < 0.5 < hi


def test_decide_pruning_rule():
    from scripts.embedding_eval.evaluate_exact import decide_pruning

    rng = np.random.default_rng(0)
    best = rng.uniform(0.4, 0.6, 500)
    per_q = {
        "bgem3_c512": best,
        "bgem3_d512_c512": best - 0.01,  # within threshold -> kept
        "qwen06_c512": best - 0.2,  # clearly worse -> pruned
        "qwen06_d128_c2048": best - 0.3,  # reference -> kept despite being worse
        "old_c512": best - 0.5,  # already pruned earlier -> ignored
    }
    cfg = ExactConfig(n_boot=200)
    dec = decide_pruning(per_q, ["old_c512"], cfg)
    assert dec["best"] == "bgem3_c512"
    act = {d["variant"]: d["action"] for d in dec["decisions"]}
    assert act == {
        "bgem3_c512": "best",
        "bgem3_d512_c512": "kept",
        "qwen06_c512": "pruned",
        "qwen06_d128_c2048": "kept_reference",
    }
    assert dec["pruned_now"] == ["qwen06_c512"]
    assert dec["pruned_cumulative"] == ["old_c512", "qwen06_c512"]
    assert dec["embed_needed"] == ["bgem3_c512", "qwen06_c2048"]
    d0 = next(d for d in dec["decisions"] if d["variant"] == "qwen06_c512")
    assert d0["ci_hi"] < cfg.prune_threshold
