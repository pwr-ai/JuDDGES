from pathlib import Path
import json

import pandas as pd
import typer


def main(
    root_dir: Path = typer.Option(...),
) -> None:
    results = []
    for f in Path("data/experiments/predict/pl-court-instruct").glob("metrics_*.json"):
        model_name = f.stem.replace("metrics_", "")
        with f.open() as file:
            m_res = json.load(file)
            results.append(
                {
                    "llm": model_name,
                    "full_text_chrf": m_res["full_text_chrf"],
                    **m_res["field_chrf"],
                }
            )

    summary_file = root_dir / "metrics_summary.md"
    pd.DataFrame(results).sort_values("llm").to_markdown(summary_file, index=False, floatfmt=".3f")


if __name__ == "__main__":
    typer.run(main)
