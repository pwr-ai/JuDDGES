import json
import shutil

import pandas as pd
import pytest

from scripts.embedding_eval.export_tables import (
    LONG_COLS,
    ExportConfig,
    run,
    tex_escape,
    to_long,
)


def _wide(it: int, view: str, variants: dict[str, float]) -> pd.DataFrame:
    rows = []
    for v, nd in variants.items():
        rows.append(
            {
                "iter": it,
                "view": view,
                "n_judgments": 20000 * (it + 1),
                "n_queries": 1000 * (it + 1),
                "variant": v,
                "model": v.split("_")[0],
                "chunking": v.split("_")[-1],
                "dim": 1024,
                "mrr10": nd - 0.03,
                "ndcg10": nd,
                "recall1": 0.3,
                "recall10": 0.5,
                "recall100": 0.7,
                "verbatim_ndcg10": nd + 0.1,
                "abstractive_ndcg10": nd - 0.01,
                "vectors_per_judgment": 18.0,
                "corpus_vectors": 5.4e7,
                "gb_fp32_corpus": 221.0,
                "gb_rq8_corpus": 56.0,
                "gb_rq4_corpus": 28.0,
                "gb_rq1_corpus": 7.0,
                "gpu_hours_corpus": 118.0,
                "ndcg10_lo": nd - 0.02,
                "ndcg10_hi": nd + 0.02,
                "ndcg10_delta": nd - 0.45,
                "ndcg10_ci_lo": nd - 0.47,
                "ndcg10_ci_hi": nd - 0.43,
                "ci_excludes_0": True,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def base(tmp_path):
    for it, variants in ((0, {"bgem3_c512": 0.45, "qwen06_c512": 0.39}), (1, {"bgem3_c512": 0.43})):
        d = tmp_path / "results" / f"iter_{it}"
        d.mkdir(parents=True)
        pd.concat([_wide(it, v, variants) for v in ("cumulative", "fixed")]).to_csv(
            d / "results_exact.csv", index=False
        )
        dec = {
            "iter": it,
            "best": "bgem3_c512",
            "threshold": -0.03,
            "decisions": [
                {"variant": "bgem3_c512", "ndcg10": 0.45, "action": "best"},
                *(
                    [
                        {
                            "variant": "qwen06_c512",
                            "ndcg10": 0.39,
                            "delta_vs_best": -0.06,
                            "ci_lo": -0.08,
                            "ci_hi": -0.04,
                            "action": "pruned",
                        }
                    ]
                    if it == 0
                    else []
                ),
            ],
        }
        (d / "decisions.json").write_text(json.dumps(dec))
    return tmp_path


def test_to_long_columns_and_ci():
    long = to_long(_wide(0, "cumulative", {"bgem3_c512": 0.45}))
    assert list(long.columns) == LONG_COLS
    nd = long[long["metric"] == "ndcg10"].iloc[0]
    assert (nd["value"], nd["ci_lo"], nd["ci_hi"]) == pytest.approx((0.45, 0.43, 0.47))
    assert "ci_excludes_0" not in set(long["metric"])
    assert long[long["metric"] == "mrr10"]["ci_lo"].isna().all()


def test_export_writes_csvs_and_tables(base):
    written = run(ExportConfig(data_dir=str(base)))
    res = base / "results"
    allx = pd.read_csv(res / "all_exact.csv")
    assert set(allx["iter"]) == {0, 1} and set(allx["view"]) == {"cumulative", "fixed"}
    assert not (res / "all_weaviate.csv").exists()
    for name in (
        "main_results",
        "scaling_cumulative",
        "scaling_fixed",
        "storage_cost",
        "pruning_log",
    ):
        text = (res / "tables" / f"{name}.tex").read_text()
        assert "\\toprule" in text and "\\bottomrule" in text
    assert "quantization" not in written
    scaling = (res / "tables" / "scaling_cumulative.tex").read_text()
    assert "0.450 $\\pm$ 0.020" in scaling and "0.430 $\\pm$ 0.020" in scaling
    assert "qwen06\\_c512 & 0.390 $\\pm$ 0.020 & --" in scaling  # pruned at iter 1
    assert "pruned" in (res / "tables" / "pruning_log.tex").read_text()
    main = (res / "tables" / "main_results.tex").read_text()
    assert "iteration 1" in main and "qwen06" not in main  # latest iteration only


def test_export_from_csvs_alone_and_weaviate(base):
    run(ExportConfig(data_dir=str(base)))
    res = base / "results"
    for d in res.glob("iter_*"):
        shutil.rmtree(d)
    w = pd.DataFrame(
        [
            {"iter": 1, "view": "cumulative", "variant": v, "metric": m, "value": x}
            for v, x in (("hnsw_fp32", 0.43), ("rq8", 0.42))
            for m in ("ndcg10", "p95_ms")
        ]
    )
    w.to_csv(res / "all_weaviate.csv", index=False)
    written = run(ExportConfig(data_dir=str(base)))
    assert "quantization" in written
    assert "rq8" in written["quantization"].read_text()
    assert (res / "tables" / "pruning_log.tex").exists()


def test_figures_are_pdf(base):
    pytest.importorskip("matplotlib")
    written = run(ExportConfig(data_dir=str(base)))
    assert written["fig_cumulative"].read_bytes().startswith(b"%PDF")


def test_tex_escape():
    assert tex_escape("a_b%c") == "a\\_b\\%c"
