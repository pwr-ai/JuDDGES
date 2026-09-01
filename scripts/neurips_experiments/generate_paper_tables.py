#!/usr/bin/env python
"""Generate LaTeX tables from experiment results for the NeurIPS paper.

Reads JSON outputs from exp1-exp7 and produces .tex table files
that can be \\input{} in the paper.

Usage:
    python scripts/neurips_experiments/generate_paper_tables.py \
        --results-dir results/neurips/ \
        --output-dir papers/neurips-2026-juddges-bench/tables/
"""

import json
from pathlib import Path
from typing import Any

import typer
from loguru import logger


def latex_escape(s: str) -> str:
    """Escape special LaTeX characters."""
    replacements = {
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
        "$": r"\$",
    }
    for old, new in replacements.items():
        s = s.replace(old, new)
    return s


def generate_annotation_validation_table(data: dict[str, Any]) -> str:
    """Generate Table: Semi-auto vs Manual annotation comparison."""
    summary = data["summary"]
    deltas = data["per_field_deltas"]

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Annotation validation: GPT-4.1 performance on semi-automatic (GPT-4 pre-annotated) vs.\ fully manual expert annotations on \texttt{pl-swiss-franc-loans}. Positive $\Delta$ indicates inflation from pre-annotation circularity.}",
        r"\label{tab:annotation_validation}",
        r"\begin{tabular}{lccr}",
        r"\toprule",
        r"\textbf{Metric} & \textbf{Semi-Auto} & \textbf{Manual} & \textbf{$\Delta$} \\",
        r"\midrule",
        f"Overall & {summary['semi_auto_overall']:.3f} & {summary['manual_overall']:.3f} & {summary['overall_delta']:+.3f} \\\\",
        f"\\# Documents & {summary['num_semi_auto_docs']} & {summary['num_manual_docs']} & -- \\\\",
    ]

    # Add top 5 most inflated fields
    if data.get("most_inflated_fields"):
        lines.append(r"\midrule")
        lines.append(r"\multicolumn{4}{l}{\textit{Most inflated fields:}} \\")
        for item in data["most_inflated_fields"][:5]:
            field = latex_escape(item["field"])
            lines.append(
                f"\\quad {field} & -- & -- & {item['delta']:+.3f} \\\\"
            )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def generate_field_difficulty_table(data: dict[str, Any]) -> str:
    """Generate Table: Field difficulty taxonomy."""
    ranking = data.get("difficulty_ranking", [])

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Field difficulty taxonomy: mean extraction score across all models, stratified by field type. Performance varies by %.1fx across types.}" % data["summary"]["performance_variance_ratio"],
        r"\label{tab:field_difficulty}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"\textbf{Field Type} & \textbf{\# Fields} & \textbf{Mean Score} & \textbf{Range} \\",
        r"\midrule",
    ]

    type_stats = data.get("type_statistics", {})
    for item in ranking:
        ftype = item["type"]
        stats = type_stats.get(ftype, {})
        n_fields = stats.get("num_fields", 0)
        overall = stats.get("overall", {})
        mean_score = overall.get("mean", 0)
        min_score = overall.get("min", 0)
        max_score = overall.get("max", 0)
        lines.append(
            f"  {latex_escape(ftype)} & {n_fields} & {mean_score:.3f} & [{min_score:.3f}, {max_score:.3f}] \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def generate_cross_model_table(data: dict[str, Any]) -> str:
    """Generate Table: Cross-model agreement."""
    field_ranking = data.get("field_ranking", [])

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Cross-model agreement between Gemini~2.5~Pro and GPT-4.1 on Schema~B (16 fields, %d documents). Agreement measured by field-type-appropriate metrics.}" % data.get("sample_size", 0),
        r"\label{tab:cross_model_agreement}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"\textbf{Field} & \textbf{Type} & \textbf{Agreement} & \textbf{Exact\%} \\",
        r"\midrule",
    ]

    for item in field_ranking:
        field = latex_escape(item["field"])
        ftype = item.get("field_type", "")
        agreement = item.get("mean_agreement", 0)
        exact = item.get("exact_match_pct", 0)
        lines.append(f"  {field} & {ftype} & {agreement:.3f} & {exact:.1f}\\% \\\\")

    lines.extend([
        r"\midrule",
        f"  \\textbf{{Overall}} & -- & \\textbf{{{data.get('overall_agreement', 0):.3f}}} & -- \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def generate_aggregation_table(data: dict[str, Any]) -> str:
    """Generate Table: Multi-document aggregation accuracy by scale."""
    scale_results = data.get("scale_results", {})

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Multi-document aggregation error (relative) at different collection sizes. Error decreases as collection size increases, with proportion queries being most robust.}",
        r"\label{tab:multidoc_aggregation}",
        r"\begin{tabular}{rcccc}",
        r"\toprule",
        r"\textbf{Size} & \textbf{Overall} & \textbf{Count} & \textbf{Proportion} & \textbf{Statistic} \\",
        r"\midrule",
    ]

    for size in sorted(scale_results.keys(), key=int):
        sr = scale_results[size]
        pt = sr.get("per_type", {})
        lines.append(
            f"  {size} & {sr['overall_mean_error']:.3f}"
            f" & {pt.get('count', {}).get('mean_error', 0):.3f}"
            f" & {pt.get('proportion', {}).get('mean_error', 0):.3f}"
            f" & {pt.get('statistic', {}).get('mean_error', 0):.3f} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def generate_error_taxonomy_table(data: dict[str, Any]) -> str:
    """Generate Table: Error taxonomy distribution."""
    dist = data.get("overall_distribution", {})

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Error taxonomy: distribution of extraction error types across %d sampled errors. Interpretation and omission errors dominate.}" % data.get("total_errors_analyzed", 0),
        r"\label{tab:error_taxonomy}",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"\textbf{Error Category} & \textbf{Count} & \textbf{\%} \\",
        r"\midrule",
    ]

    for cat, stats in dist.items():
        lines.append(
            f"  {latex_escape(cat)} & {stats['count']} & {stats['pct']:.1f}\\% \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def main(
    results_dir: Path = typer.Option(
        "results/neurips/", help="Directory with experiment JSON results"
    ),
    output_dir: Path = typer.Option(
        None, help="Output directory for .tex files (default: same as results)"
    ),
):
    """Generate LaTeX tables from experiment results."""
    if output_dir is None:
        output_dir = results_dir / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)

    generators = {
        "exp1_annotation_validation.json": (
            "tab_annotation_validation.tex",
            generate_annotation_validation_table,
        ),
        "exp4_cross_model_agreement.json": (
            "tab_cross_model_agreement.tex",
            generate_cross_model_table,
        ),
        "exp5_multidoc_aggregation.json": (
            "tab_multidoc_aggregation.tex",
            generate_aggregation_table,
        ),
        "exp6_field_difficulty.json": (
            "tab_field_difficulty.tex",
            generate_field_difficulty_table,
        ),
        "exp7_error_taxonomy.json": (
            "tab_error_taxonomy.tex",
            generate_error_taxonomy_table,
        ),
    }

    for json_name, (tex_name, generator) in generators.items():
        json_path = results_dir / json_name
        if json_path.exists():
            with open(json_path) as f:
                data = json.load(f)
            tex_content = generator(data)
            tex_path = output_dir / tex_name
            with open(tex_path, "w") as f:
                f.write(tex_content)
            logger.info(f"Generated {tex_path}")
        else:
            logger.warning(f"Not found: {json_path}")

    logger.info(f"Tables written to {output_dir}")


if __name__ == "__main__":
    typer.run(main)
