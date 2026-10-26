"""Exit 0 when a finished run can be skipped: its run_record.json matches the current settings."""

import json
from pathlib import Path

import typer

from juddges.utils.run_record import run_record_matches


def main(
    output_dir: Path = typer.Argument(...),
    seed: int = typer.Option(...),
    guided_json: bool = typer.Option(...),
    max_model_len: int = typer.Option(...),
    max_docs: int = typer.Option(0, help="0 means all documents (config value null)"),
) -> None:
    matches = run_record_matches(
        output_dir,
        seed,
        guided_json=guided_json,
        max_model_len=max_model_len,
        max_docs=max_docs or None,
    )
    typer.echo(json.dumps({"output_dir": str(output_dir), "matches": matches}))
    raise typer.Exit(code=0 if matches else 1)


if __name__ == "__main__":
    typer.run(main)
