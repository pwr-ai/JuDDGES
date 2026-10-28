"""Export the iterative-scaling results as paper artefacts.

Aggregates `results/iter_*/results_exact.csv` (and `results_weaviate.csv` when present) into the
long-format `results/all_exact.csv` / `results/all_weaviate.csv`, the decision logs into
`results/all_decisions.csv`, and renders everything else from those CSVs alone:

    results/tables/*.tex      booktabs tables (main, scaling, storage/cost, pruning, quantization)
    results/figures/*.pdf     nDCG@10 vs n_judgments per config with CI bands, per view

Long format columns: iter, n_judgments, n_queries, view, variant, metric, value, ci_lo, ci_hi,
plus model, chunking, dim. When no per-iteration CSVs exist but the `all_*.csv` files do, the
tables are rebuilt from those.

Run: python -m scripts.embedding_eval.export_tables   (EVAL_DATA_DIR, EVAL_BASELINE override)
Figures need matplotlib; without it they are skipped with a warning.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from rich.console import Console

from scripts.embedding_eval.common import DEFAULT_OUT_DIR, env_override

console = Console()

LONG_COLS = [
    "iter",
    "n_judgments",
    "n_queries",
    "view",
    "variant",
    "metric",
    "value",
    "ci_lo",
    "ci_hi",
    "model",
    "chunking",
    "dim",
]
ID_COLS = {"iter", "view", "n_judgments", "n_queries", "variant", "model", "chunking", "dim"}
# wide column -> (metric, ci_lo column, ci_hi column) for metrics that carry a CI
CI_COLS = {"ndcg10": ("ndcg10_lo", "ndcg10_hi"), "ndcg10_delta": ("ndcg10_ci_lo", "ndcg10_ci_hi")}
SKIP_COLS = {"ndcg10_lo", "ndcg10_hi", "ndcg10_ci_lo", "ndcg10_ci_hi", "ci_excludes_0"}


@dataclass
class ExportConfig:
    data_dir: str = DEFAULT_OUT_DIR
    baseline: str = "bgem3_c512"
    view: str = "cumulative"  # view used by the main/storage tables


# ---------------------------------------------------------------- aggregation


def to_long(wide: pd.DataFrame) -> pd.DataFrame:
    """Melt a wide per-iteration results frame to the long format."""
    ids = [c for c in LONG_COLS[:5] + LONG_COLS[9:] if c in wide.columns]
    metrics = [
        c
        for c in wide.columns
        if c not in ID_COLS and c not in SKIP_COLS and wide[c].dtype.kind in "fiub"
    ]
    parts = []
    for m in metrics:
        part = wide[ids].copy()
        part["metric"] = m
        part["value"] = wide[m].astype(float)
        lo, hi = CI_COLS.get(m, (None, None))
        part["ci_lo"] = wide[lo] if lo in wide.columns else np.nan
        part["ci_hi"] = wide[hi] if hi in wide.columns else np.nan
        parts.append(part)
    out = pd.concat(parts, ignore_index=True)
    for c in LONG_COLS:
        if c not in out.columns:
            out[c] = np.nan
    return out[LONG_COLS].sort_values(["iter", "view", "variant", "metric"]).reset_index(drop=True)


def _iter_dirs(res: Path) -> list[Path]:
    return sorted(
        (d for d in res.glob("iter_*") if re.fullmatch(r"iter_\d+", d.name)),
        key=lambda d: int(d.name.split("_")[1]),
    )


def aggregate(base: Path) -> dict[str, pd.DataFrame]:
    """Write all_exact.csv / all_weaviate.csv / all_decisions.csv from the per-iteration files."""
    res = base / "results"
    dirs = _iter_dirs(res)
    out: dict[str, pd.DataFrame] = {}

    exact = [
        pd.read_csv(d / "results_exact.csv") for d in dirs if (d / "results_exact.csv").exists()
    ]
    if exact:
        out["exact"] = to_long(pd.concat(exact, ignore_index=True))
        out["exact"].to_csv(res / "all_exact.csv", index=False)

    weav = []
    for d in dirs:
        f = d / "results_weaviate.csv"
        if not f.exists():
            continue
        w = pd.read_csv(f)
        if "iter" not in w.columns:
            w["iter"] = int(d.name.split("_")[1])
        weav.append(w if "metric" in w.columns else to_long(w) if "variant" in w.columns else w)
    if weav:
        out["weaviate"] = pd.concat(weav, ignore_index=True)
        out["weaviate"].to_csv(res / "all_weaviate.csv", index=False)

    dec_rows = []
    for d in dirs:
        f = d / "decisions.json"
        if not f.exists():
            continue
        dec = json.loads(f.read_text())
        for r in dec["decisions"]:
            dec_rows.append({"iter": dec["iter"], "best": dec["best"], **r})
    if dec_rows:
        out["decisions"] = pd.DataFrame(dec_rows)
        out["decisions"].to_csv(res / "all_decisions.csv", index=False)
    return out


def load_csvs(base: Path) -> dict[str, pd.DataFrame]:
    out = {}
    for name in ("exact", "weaviate", "decisions"):
        f = base / "results" / f"all_{name}.csv"
        if f.exists():
            out[name] = pd.read_csv(f)
    return out


# ---------------------------------------------------------------- LaTeX helpers


def tex_escape(s: object) -> str:
    return re.sub(r"([_%&#$])", r"\\\1", str(s))


def latex_table(
    header: list[str], rows: list[list[str]], align: str, caption: str, label: str
) -> str:
    """booktabs table; cells are already LaTeX-safe strings (use `tex_escape` for raw text)."""
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        f"\\begin{{tabular}}{{{align}}}",
        "\\toprule",
        " & ".join(header) + " \\\\",
        "\\midrule",
        *[" & ".join(r) + " \\\\" for r in rows],
        "\\bottomrule",
        "\\end{tabular}",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\end{table}",
    ]
    return "\n".join(lines) + "\n"


def _num(x: float, fmt: str = "{:.3f}") -> str:
    return "--" if x is None or pd.isna(x) else fmt.format(x)


def _wide(df: pd.DataFrame, it: int, view: str) -> pd.DataFrame:
    """metric -> columns for one (iteration, view); index = variant."""
    d = df[(df["iter"] == it) & (df["view"] == view)]
    w = d.pivot(index="variant", columns="metric", values="value")
    ci = d[d["metric"] == "ndcg10"].set_index("variant")[["ci_lo", "ci_hi"]]
    dci = d[d["metric"] == "ndcg10_delta"].set_index("variant")[["ci_lo", "ci_hi"]]
    w[["ndcg_lo", "ndcg_hi"]] = ci.reindex(w.index).to_numpy()
    w[["delta_lo", "delta_hi"]] = dci.reindex(w.index).to_numpy()
    meta = d.drop_duplicates("variant").set_index("variant")[["dim", "n_judgments", "n_queries"]]
    return w.join(meta).sort_values("ndcg10", ascending=False)


def main_results_table(df: pd.DataFrame, cfg: ExportConfig) -> str:
    it = int(df["iter"].max())
    w = _wide(df, it, cfg.view)
    best = w["ndcg10"].max()
    rows = []
    for v, r in w.iterrows():
        nd = _num(r["ndcg10"])
        nd = f"\\textbf{{{nd}}}" if r["ndcg10"] == best else nd
        delta = (
            "--"
            if pd.isna(r.get("ndcg10_delta")) or v == cfg.baseline
            else f"{r['ndcg10_delta']:+.3f} [{r['delta_lo']:+.3f}, {r['delta_hi']:+.3f}]"
        )
        rows.append(
            [
                tex_escape(v),
                str(int(r["dim"])),
                f"{nd} [{_num(r['ndcg_lo'])}, {_num(r['ndcg_hi'])}]",
                _num(r["mrr10"]),
                _num(r["recall1"]),
                _num(r["recall10"]),
                _num(r["recall100"]),
                _num(r.get("verbatim_ndcg10")),
                _num(r.get("abstractive_ndcg10")),
                delta,
            ]
        )
    n_j, n_q = int(w["n_judgments"].iloc[0]), int(w["n_queries"].iloc[0])
    return latex_table(
        [
            "Config",
            "Dim",
            "nDCG@10 [95\\% CI]",
            "MRR@10",
            "R@1",
            "R@10",
            "R@100",
            "nDCG verb.",
            "nDCG abstr.",
            f"$\\Delta$ nDCG@10 vs {tex_escape(cfg.baseline)}",
        ],
        rows,
        "llcccccccc",
        f"Exact dense retrieval (MaxP), iteration {it}: {n_j:,} judgments, {n_q:,} queries "
        f"({cfg.view} view). Best nDCG@10 in bold.",
        "tab:embedding-main",
    )


def scaling_table(df: pd.DataFrame, view: str) -> str:
    its = sorted(df["iter"].unique())
    nd = df[(df["view"] == view) & (df["metric"] == "ndcg10")].copy()
    nd["cell"] = [
        f"{v:.3f} $\\pm$ {(hi - lo) / 2:.3f}"
        for v, lo, hi in zip(nd["value"], nd["ci_lo"], nd["ci_hi"])
    ]
    piv = nd.pivot(index="variant", columns="iter", values="cell")
    order = nd[nd["iter"] == nd["iter"].max()].sort_values("value", ascending=False)["variant"]
    order = list(order) + [v for v in piv.index if v not in set(order)]
    nj = nd.drop_duplicates("iter").set_index("iter")["n_judgments"]
    header = ["Config"] + [f"{int(nj.get(i, 0)) // 1000}k" for i in its]
    rows = [
        [tex_escape(v)]
        + [piv.loc[v, i] if i in piv.columns and pd.notna(piv.loc[v, i]) else "--" for i in its]
        for v in order
    ]
    return latex_table(
        header,
        rows,
        "l" + "c" * len(its),
        f"nDCG@10 $\\pm$ half-width of the 95\\% bootstrap CI over queries, by corpus size "
        f"({view} view). `--': pruned or not embedded at that iteration.",
        f"tab:embedding-scaling-{view}",
    )


def storage_table(df: pd.DataFrame, cfg: ExportConfig) -> str:
    d = df[df["view"] == cfg.view]
    last = d.groupby("variant")["iter"].max()
    rows = []
    for v in sorted(last.index):
        x = d[(d["variant"] == v) & (d["iter"] == last[v])].set_index("metric")["value"]
        if "corpus_vectors" not in x:
            continue
        rows.append(
            (
                x.get("gb_rq8_corpus", np.nan),
                [
                    tex_escape(v),
                    _num(x["vectors_per_judgment"], "{:.2f}"),
                    _num(x["corpus_vectors"] / 1e6, "{:.1f}"),
                    *[
                        _num(x.get(f"gb_{k}_corpus"), "{:.1f}")
                        for k in ("fp32", "rq8", "rq4", "rq1")
                    ],
                    _num(x.get("gpu_hours_corpus"), "{:.0f}"),
                ],
            )
        )
    rows = [r for _, r in sorted(rows, key=lambda t: (np.isnan(t[0]), t[0]))]
    return latex_table(
        ["Config", "Vec./judg.", "Vectors (M)", "fp32", "RQ8", "RQ4", "RQ1", "GPU-h"],
        rows,
        "lccccccc",
        "Storage (GB) and embedding cost for the full 3M-judgment corpus, extrapolated from the "
        "vectors per judgment of the latest evaluated iteration. RQ8 $=$ dim$+16$ B, RQ4 $=$ "
        "16$+$dim$/2$ B, RQ1 $=$ dim$/8+8$ B per vector. GPU-h from the measured throughput.",
        "tab:embedding-storage",
    )


def pruning_table(dec: pd.DataFrame) -> str:
    rows = []
    for _, r in dec.sort_values(["iter", "ndcg10"], ascending=[True, False]).iterrows():
        has = pd.notna(r.get("delta_vs_best"))
        rows.append(
            [
                str(int(r["iter"])),
                tex_escape(r["variant"]),
                _num(r["ndcg10"]),
                f"{r['delta_vs_best']:+.3f} [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]"
                if has
                else "--",
                tex_escape(r["action"]),
            ]
        )
    thr = (
        dec["threshold"].dropna().iloc[0]
        if "threshold" in dec and dec["threshold"].notna().any()
        else -0.03
    )
    return latex_table(
        ["Iter.", "Config", "nDCG@10", "$\\Delta$ vs best [95\\% CI]", "Decision"],
        rows,
        "clccl",
        f"Pre-registered pruning log: a non-reference config is pruned when the CI upper bound of "
        f"its nDCG@10 delta vs the current best is below ${thr}$.",
        "tab:embedding-pruning",
    )


QUANT_METRICS = [
    ("ndcg10", "nDCG@10", "{:.3f}"),
    ("recall100", "R@100", "{:.3f}"),
    ("ann_recall100", "ANN R@100", "{:.3f}"),
    ("p50_ms", "p50 (ms)", "{:.1f}"),
    ("p95_ms", "p95 (ms)", "{:.1f}"),
]


def quantization_table(w: pd.DataFrame) -> str | None:
    """Rows = variant at the latest iteration, columns = whichever of QUANT_METRICS exist."""
    if not {"variant", "metric", "value", "iter"} <= set(w.columns):
        logger.warning("all_weaviate.csv is not in long format, skipping the quantization table")
        return None
    w = w[w["iter"] == w["iter"].max()]
    if "view" in w.columns and (w["view"] == "cumulative").any():
        w = w[w["view"] == "cumulative"]
    cols = [m for m in QUANT_METRICS if (w["metric"] == m[0]).any()]
    if not cols:
        return None
    piv = w.pivot_table(index="variant", columns="metric", values="value", aggfunc="first")
    rows = [[tex_escape(v)] + [_num(piv.loc[v].get(m), f) for m, _, f in cols] for v in piv.index]
    return latex_table(
        ["Variant"] + [h for _, h, _ in cols],
        rows,
        "l" + "c" * len(cols),
        "Weaviate quantization variants at the latest iteration.",
        "tab:embedding-quantization",
    )


# ---------------------------------------------------------------- figures


def plot_scaling(df: pd.DataFrame, view: str, path: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed, skipping figures")
        return False
    nd = df[(df["view"] == view) & (df["metric"] == "ndcg10")]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for v, g in nd.groupby("variant"):
        g = g.sort_values("n_judgments")
        (line,) = ax.plot(g["n_judgments"], g["value"], marker="o", ms=3, lw=1.2, label=v)
        ax.fill_between(
            g["n_judgments"], g["ci_lo"], g["ci_hi"], alpha=0.15, color=line.get_color()
        )
    ax.set_xlabel("corpus size (judgments)")
    ax.set_ylabel("nDCG@10")
    ax.set_title(f"nDCG@10 vs corpus size ({view} view)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6, ncol=2, loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


# ---------------------------------------------------------------- entrypoint


def run(cfg: ExportConfig | None = None) -> dict[str, Path]:
    cfg = env_override(cfg or ExportConfig(), "EVAL_")
    base = Path(cfg.data_dir)
    res = base / "results"
    data = aggregate(base) if _iter_dirs(res) else {}
    if "exact" not in data:
        data = load_csvs(base)
    if "exact" not in data:
        raise FileNotFoundError(f"no results_exact.csv / all_exact.csv under {res}")

    tdir, fdir = res / "tables", res / "figures"
    tdir.mkdir(parents=True, exist_ok=True)
    fdir.mkdir(parents=True, exist_ok=True)
    exact = data["exact"]
    written: dict[str, Path] = {}

    def write(name: str, text: str | None) -> None:
        if text is None:
            return
        p = tdir / f"{name}.tex"
        p.write_text(text)
        written[name] = p

    write("main_results", main_results_table(exact, cfg))
    for view in sorted(exact["view"].unique()):
        write(f"scaling_{view}", scaling_table(exact, view))
        if plot_scaling(exact, view, fdir / f"ndcg10_vs_n_judgments_{view}.pdf"):
            written[f"fig_{view}"] = fdir / f"ndcg10_vs_n_judgments_{view}.pdf"
    write("storage_cost", storage_table(exact, cfg))
    if "decisions" in data:
        write("pruning_log", pruning_table(data["decisions"]))
    else:
        logger.warning("no decisions CSV, skipping the pruning table")
    if "weaviate" in data and len(data["weaviate"]):
        write("quantization", quantization_table(data["weaviate"]))
    for name, p in written.items():
        logger.info("wrote {} -> {}", name, p)
    console.print(f"[green]exported {len(written)} artefacts under {res}[/green]")
    return written


if __name__ == "__main__":
    run()
