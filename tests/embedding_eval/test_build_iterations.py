import pandas as pd

from scripts.embedding_eval.build_dataset import (
    DatasetConfig,
    assemble,
    chunk_rows,
    plan_iterations,
    sample_ids,
)
from scripts.embedding_eval.iterations import data_dir, emb_dir, results_dir

CFG = DatasetConfig(n_queries=5, n_distractors=15, seed=42)


def _meta(n=400):
    return pd.DataFrame(
        {
            "judgment_id": [f"j{i}" for i in range(n)],
            "thesis_len": [100 if i % 2 == 0 else 0 for i in range(n)],
            "text_len": [500 if i % 7 else 0 for i in range(n)],
        }
    )


def test_iteration_zero_equals_original_sampling():
    meta = _meta()
    assert plan_iterations(meta, CFG, 0)[0] == sample_ids(meta, CFG)


def test_nested_no_duplicates_and_sizes():
    plan = plan_iterations(_meta(), CFG, 5)
    ids = [j for q, d in plan for j in q + d]
    assert len(ids) == len(set(ids)) == 6 * 20
    for q, d in plan:
        assert len(q) == 5 and len(d) == 15
    # iteration k's corpus is a strict superset of k-1: prefix of the replay is unchanged
    for k in range(1, 6):
        assert plan_iterations(_meta(), CFG, k)[:k] == plan[:k]
    # queries are valid candidates, distractors have text
    meta = _meta().set_index("judgment_id")
    for q, d in plan:
        assert (meta.loc[q, "thesis_len"] == 100).all()
        assert (meta.loc[q + d, "text_len"] > 0).all()


def test_deterministic_for_same_seed_and_differs_across_iterations():
    a, b = plan_iterations(_meta(), CFG, 3), plan_iterations(_meta(), CFG, 3)
    assert a == b
    assert a[1] != a[2]
    other = plan_iterations(_meta(), DatasetConfig(n_queries=5, n_distractors=15, seed=7), 3)
    assert other != a


def test_every_query_judgment_in_corpus_and_global_qids():
    meta = _meta()
    plan = plan_iterations(meta, CFG, 2)
    seen = set()
    for k, (q, d) in enumerate(plan):
        sel = pd.DataFrame(
            {
                "judgment_id": q + d,
                "thesis": ["t"] * 20,
                "full_text": ["t body"] * 20,
            }
        )
        judgments, queries = assemble(sel.sample(frac=1, random_state=k), q, d, k, 5)
        assert set(queries["judgment_id"]) <= set(judgments["judgment_id"])
        assert queries["judgment_id"].tolist() == q  # deterministic order, queries first
        assert judgments["judgment_id"].tolist() == q + d
        assert queries["qid"].tolist() == [f"q{k * 5 + i:04d}" for i in range(5)]
        assert not seen & set(queries["qid"])
        seen |= set(queries["qid"])


def test_chunk_uid_independent_of_iteration():
    rows = chunk_rows("abc", ["x", " ", "y"], [1, 1, 1])
    assert [r["chunk_uid"] for r in rows] == ["abc#0", "abc#1"]


def test_layout_resolver_keeps_iteration_zero_paths(tmp_path):
    assert data_dir(tmp_path, 0) == tmp_path
    assert data_dir(tmp_path, 3) == tmp_path / "iter_3"
    assert emb_dir(tmp_path, "bgem3", 0) == tmp_path / "embeddings" / "bgem3"
    assert emb_dir(tmp_path, "bgem3", 2) == tmp_path / "embeddings" / "bgem3" / "iter_2"
    assert results_dir(tmp_path, 0) == tmp_path / "results" / "iter_0"
