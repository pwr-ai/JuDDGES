import json
from pathlib import Path

import pandas as pd
import typer


def main(
    root_dir: Path = typer.Option(...),
) -> None:
    results = []
    for llm_dir in Path("data/experiments/predict/pl-court-instruct").iterdir():
        llm_results = []
        for f in llm_dir.glob("metrics_*.json"):
            model_name = llm_dir.name
            with f.open() as file:
                m_res = json.load(file)
                llm_results.append(
                    {
                        "llm": model_name,
                        "full_text_chrf": m_res["full_text_chrf"],
                        **m_res["field_chrf"],
                    }
                )
        if llm_results:
            metric_mean = pd.DataFrame(llm_results).groupby("llm").mean()
            metric_std = pd.DataFrame(llm_results).groupby("llm").std()
            stats = (
                metric_mean.map(lambda x: f"{x:0.3f}")
                + " (± "
                + metric_std.map(lambda x: f"{x:0.3f}")
                + ")"
            )

            results.append(stats)

    summary_file = root_dir / "metrics_summary.md"
    pd.concat(results).to_markdown(summary_file)


if __name__ == "__main__":
    typer.run(main)
