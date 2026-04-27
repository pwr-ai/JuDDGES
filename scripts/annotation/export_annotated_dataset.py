# todo: rewrite to use label studio api

import json
from pathlib import Path

import numpy as np
import pandas as pd
import typer

from label_studio_toolkit.schemas.en_appealcourt import AppealCourtAnnotation
from label_studio_toolkit.schemas.swiss_frank import SwissFrancJudgmentAnnotation

SCHEMA_MAP = {
    "swiss_frank": SwissFrancJudgmentAnnotation,
    "en_appealcourt": AppealCourtAnnotation,
}


def main(
    input_file: Path = typer.Option(...),
    output_path: Path = typer.Option(...),
    schema: str = typer.Option(...),
):
    df = pd.read_parquet(input_file)
    schema = SCHEMA_MAP[schema]
    output_data = []

    for i, entry in df.iterrows():
        values = {k: v for k, v in dict(entry).items() if isinstance(v, np.ndarray) or pd.notna(v)}
        datapoint = schema(**values)
        output_data.append(
            {
                "context": entry["text"],
                "output": datapoint.model_dump_json(),
            }
        )

    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path / f"{input_file.stem}_dataset.json", "w") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)

    with open(output_path / "schema.yaml", "w") as f:
        f.write(schema.get_schema_string())


if __name__ == "__main__":
    typer.run(main)
