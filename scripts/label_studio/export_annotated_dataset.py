# todo: rewrite to use label studio api
# todo: this script uses 2 different export types: json and json_min

import json
from pathlib import Path

import pandas as pd
import typer
from pydantic import ValidationError

from label_studio_toolkit.schemas.en_appealcourt import BaseModel
from label_studio_toolkit.schemas.personal_rights import PersonalRightsAnnotation


def main(
    input_file: Path = typer.Option(...),
    output_path: Path = typer.Option("data/label_studio/exports/personal_rights"),
    use_preannotation: bool = typer.Option(False),
):
    with open(input_file, "r") as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    output_data = []

    for i, entry in df.iterrows():
        if use_preannotation:
            datapoint = get_preannotation(entry, PersonalRightsAnnotation)
            text = entry["data"]["text"]
        else:
            values = {k: v for k, v in dict(entry).items() if pd.notna(v)}
            for k, v in values.items():
                if isinstance(v, dict) and "choices" in v:
                    values[k] = v["choices"]
            datapoint = PersonalRightsAnnotation(**values)
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
        f.write(PersonalRightsAnnotation.get_schema_string())


def get_preannotation(entry: dict, schema: BaseModel) -> dict:
    [predictions] = entry["predictions"]
    values = {}
    for result in predictions["result"]:
        [value] = result["value"].values()
        values[result["from_name"]] = value
    try:
        datapoint = schema(**values)
    except ValidationError as e:
        errors = e.errors()
        for error in errors:
            if error["type"] in ["enum", "string_type"]:
                [values[error["loc"][0]]] = values[error["loc"][0]]
        datapoint = schema(**values)

    return datapoint


if __name__ == "__main__":
    typer.run(main)
