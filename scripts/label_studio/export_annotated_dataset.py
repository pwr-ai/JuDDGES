import json
from pathlib import Path

import pandas as pd
import typer
from pydantic import ValidationError

from label_studio_toolkit.schemas.en_appealcourt import AppealCourtAnnotation
from label_studio_toolkit.schemas.personal_rights import PersonalRightsAnnotation
from label_studio_toolkit.schemas.swiss_frank import SwissFrancJudgmentAnnotation

SCHEMA_MAP = {
    "personal_rights": PersonalRightsAnnotation,
    "swiss_frank": SwissFrancJudgmentAnnotation,
    "en_appealcourt": AppealCourtAnnotation,
}


def main(
    input_file: Path = typer.Option(...),
    output_path: Path = typer.Option("data/label_studio/exports"),
    use_preannotation: bool = typer.Option(False),
    schema: str = typer.Option(..., help=f"One of: {list(SCHEMA_MAP.keys())}"),
):
    schema_cls = SCHEMA_MAP[schema]

    with open(input_file, "r") as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    output_data = []

    for i, entry in df.iterrows():
        if use_preannotation:
            datapoint = get_preannotation(entry, schema_cls)
            text = entry["data"]["text"]
        else:
            values = {k: v for k, v in dict(entry).items() if pd.notna(v)}
            for k, v in values.items():
                if isinstance(v, dict) and "choices" in v:
                    values[k] = v["choices"]
            datapoint = schema_cls(**values)
            text = entry["text"]
        output_data.append(
            {
                "context": text,
                "output": datapoint.model_dump_json(),
            }
        )

    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path / "dataset.json", "w") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)

    with open(output_path / "schema.yaml", "w") as f:
        f.write(schema_cls.get_schema_string())


def get_preannotation(entry: dict, schema_cls) -> dict:
    [predictions] = entry["predictions"]
    values = {}
    for result in predictions["result"]:
        [value] = result["value"].values()
        values[result["from_name"]] = value
    try:
        datapoint = schema_cls(**values)
    except ValidationError as e:
        errors = e.errors()
        for error in errors:
            if error["type"] in ["enum", "string_type"]:
                [values[error["loc"][0]]] = values[error["loc"][0]]
        datapoint = schema_cls(**values)

    return datapoint


if __name__ == "__main__":
    typer.run(main)
