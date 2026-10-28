"""Pure-logic tests for scripts/embedding_eval (no GPU, no network)."""

import numpy as np
import pandas as pd
import pytest

from scripts.embedding_eval.build_dataset import (
    DatasetConfig,
    chunk_rows,
    normalise,
    sample_ids,
    tag_kind,
)
from scripts.embedding_eval.common import env_override
from scripts.embedding_eval.embed import (
    REGISTRY,
    EmbedConfig,
    l2_normalise,
    length_order,
    restore_order,
    sanity_cosine,
)


def test_tag_kind_verbatim_ignores_case_and_whitespace():
    assert tag_kind("Sąd  uznał\nże", "x. sąd uznał że y") == "verbatim"
    assert tag_kind("zupełnie inne", "sąd uznał że") == "abstractive"
    assert normalise("  A\tb\n C ") == "a b c"


def _meta(n=100):
    return pd.DataFrame(
        {
            "judgment_id": [f"j{i}" for i in range(n)],
            "thesis_len": [100 if i % 2 == 0 else 0 for i in range(n)],
            "text_len": [10 if i % 10 else 0 for i in range(n)],
        }
    )


def test_sample_ids_disjoint_deterministic_and_filtered():
    cfg = DatasetConfig(n_queries=10, n_distractors=20)
    meta = _meta()
    q, d = sample_ids(meta, cfg)
    assert (q, d) == sample_ids(meta, cfg)
    assert len(q) == 10 and len(d) == 20 and not set(q) & set(d)
    ok = set(meta[meta.text_len > 0].judgment_id)
    assert set(q) <= ok and set(d) <= ok
    assert all(int(x[1:]) % 2 == 0 for x in q)  # queries must have a thesis


def test_sample_ids_raises_when_too_few():
    with pytest.raises(ValueError):
        sample_ids(_meta(), DatasetConfig(n_queries=1000, n_distractors=1))


def test_chunk_rows_bookkeeping_drops_empty_and_reindexes():
    rows = chunk_rows("j1", ["a", "  ", "b"], [1, 0, 2])
    assert [r["chunk_idx"] for r in rows] == [0, 1]
    assert [r["chunk_uid"] for r in rows] == ["j1#0", "j1#1"]
    assert [r["n_tokens"] for r in rows] == [1, 2]


def test_length_order_roundtrip():
    texts = ["aa", "a", "aaaa", "aaa"]
    order = length_order(texts)
    assert [texts[i] for i in order] == ["aaaa", "aaa", "aa", "a"]
    vecs = np.array([[len(texts[i])] for i in order], dtype=np.float32)
    assert restore_order(vecs, order).ravel().tolist() == [2, 1, 4, 3]


def test_normalise_and_sanity_cosine():
    c = l2_normalise(np.array([[3, 4], [1, 0], [0, 1]], dtype=np.float32))
    assert np.allclose(np.linalg.norm(c, axis=1), 1)
    q = np.array([[1, 0]], dtype=np.float32)
    s = sanity_cosine(q, c, ["a"], ["a", "b", "b"], 5)
    assert s["own"] == pytest.approx(0.6) and s["random_other"] == pytest.approx(1.0)


def test_registry_and_env_override(monkeypatch):
    assert list(REGISTRY)[-1] == "qwen4b"
    assert REGISTRY["mmlw_e5"].query_prompt == "query: "
    monkeypatch.setenv("EVAL_MODELS", "bgem3, qwen06")
    monkeypatch.setenv("EVAL_BLOCK_SIZE", "7")
    cfg = env_override(EmbedConfig(), "EVAL_")
    assert cfg.models == ("bgem3", "qwen06") and cfg.block_size == 7
