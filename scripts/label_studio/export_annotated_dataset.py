# todo: rewrite to use label studio api

import json
from pathlib import Path

import pandas as pd
import typer

from label_studio_toolkit.schemas.swiss_frank import SwissFrancJudgmentAnnotation


def main(
    input_file: Path = typer.Option(...),
    output_path: Path = typer.Option("data/label_studio/exports/swiss_franc"),
):
    with open(input_file, "r") as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    output_data = []

    for i, entry in df.iterrows():
        values = {k: v for k, v in dict(entry).items() if pd.notna(v)}
        for k, v in values.items():
            if isinstance(v, dict) and "choices" in v:
                values[k] = v["choices"]
        datapoint = SwissFrancJudgmentAnnotation(**values)
        output_data.append(
            {
                "context": entry["text"],
                "output": datapoint.model_dump_json(),
            }
        )

    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path / "dataset.json", "w") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)

    with open(output_path / "schema.yaml", "w") as f:
        f.write(SwissFrancJudgmentAnnotation.get_schema_string())


if __name__ == "__main__":
    typer.run(main)
