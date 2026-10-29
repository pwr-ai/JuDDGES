"""Converts the instruction-format `sprawy_frankowe` splits to the `swiss_franc_loans` layout.

Source records carry `prompt`, `context` and `output`. The `output` is YAML produced by an
LLM extraction (silver labels), not expert-corrected gold. Only `context` and `output` are kept;
`annotated.json` (the expert-corrected documents) is not produced here. With a leakage
report, `test_clean.json` keeps the test documents the report did not flag.
"""

import json
from pathlib import Path

import typer


def main(
    source_dir: Path = typer.Option(Path("data/datasets/pl/sprawy_frankowe")),
    target_dir: Path = typer.Option(Path("data/datasets/pl/swiss_franc_loans")),
    leakage_report: Path | None = typer.Option(
        None, help="check_split_leakage.py report; flagged test documents go to no test_clean.json"
    ),
) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        records = json.loads((source_dir / f"{split}.jsonl").read_text(encoding="utf-8"))
        rows = [{"context": r["context"], "output": r["output"]} for r in records]
        (target_dir / f"{split}.json").write_text(
            json.dumps(rows, ensure_ascii=False), encoding="utf-8"
        )
        typer.echo(f"{split}: {len(rows)} records -> {target_dir / f'{split}.json'}")
        if split == "test" and leakage_report is not None:
            report = json.loads(leakage_report.read_text(encoding="utf-8"))
            flagged = {f["index"] for f in report["flagged"] if f["split"] == "test"}
            clean = [r for i, r in enumerate(rows) if i not in flagged]
            (target_dir / "test_clean.json").write_text(
                json.dumps(clean, ensure_ascii=False), encoding="utf-8"
            )
            typer.echo(f"test_clean: {len(clean)} records ({len(flagged)} flagged removed)")


if __name__ == "__main__":
    typer.run(main)
