"""Converts the instruction-format `sprawy_frankowe` splits to the `swiss_franc_loans` layout.

Source records carry `prompt`, `context` and `output`. The `output` is YAML produced by an
LLM extraction (silver labels), not expert-corrected gold. Only `context` and `output` are kept;
`annotated.json` (the expert-corrected documents) is not produced here.
"""

import json
from pathlib import Path

import typer


def main(
    source_dir: Path = typer.Option(Path("data/datasets/pl/sprawy_frankowe")),
    target_dir: Path = typer.Option(Path("data/datasets/pl/swiss_franc_loans")),
) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        records = json.loads((source_dir / f"{split}.jsonl").read_text(encoding="utf-8"))
        rows = [{"context": r["context"], "output": r["output"]} for r in records]
        (target_dir / f"{split}.json").write_text(
            json.dumps(rows, ensure_ascii=False), encoding="utf-8"
        )
        typer.echo(f"{split}: {len(rows)} records -> {target_dir / f'{split}.json'}")


if __name__ == "__main__":
    typer.run(main)
