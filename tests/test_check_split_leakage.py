import json

from typer.testing import CliRunner

from scripts.dataset import check_split_leakage as module

LONG_TEXT = " ".join(f"slowo{i}" for i in range(200))


def _docs() -> dict[str, list[str]]:
    near_duplicate = LONG_TEXT.replace("slowo100", "zmienione")
    return {
        "train": [
            "Sygn. akt I C 123/19 wyrok " + LONG_TEXT,
            "zupelnie inny dokument " + " ".join(f"inne{i}" for i in range(200)),
            "Sygn. akt II Ca 5/20 " + " ".join(f"trzeci{i}" for i in range(200)),
        ],
        "test": [
            "SYGN. AKT I C 123/19 WYROK " + LONG_TEXT.upper(),  # same text, different casing
            near_duplicate,  # near duplicate, no signature
            "Sygn. akt II Ca 5/20 " + " ".join(f"czwarty{i}" for i in range(200)),  # same case
            "calkiem nowy " + " ".join(f"nowe{i}" for i in range(200)),  # clean
        ],
    }


def test_check_leakage_flags_expected_documents():
    report = module.check_leakage(_docs(), "train", threshold=0.8)

    flagged = {item["index"]: item["train_matches"] for item in report["flagged"]}
    assert sorted(flagged) == [0, 1, 2]
    assert flagged[0]["identical_after_normalization"] == [0]
    assert flagged[1]["near_duplicate_jaccard"].keys() == {0}
    assert flagged[2]["same_signature"] == {"II Ca 5/20": [2]}
    assert report["num_flagged"] == 3


def test_cli_reports_sizes_sha256_and_exit_code(tmp_path):
    docs = _docs()
    for name, contexts in docs.items():
        (tmp_path / f"{name}.json").write_text(
            json.dumps([{"context": c, "output": "{}"} for c in contexts])
        )
    report_file = tmp_path / "report.json"

    app = module.typer.Typer(add_completion=False)
    app.command()(module.main)
    result = CliRunner().invoke(
        app,
        [
            "--dataset-dir",
            str(tmp_path),
            "--eval-splits",
            "test",
            "--report-file",
            str(report_file),
        ],
    )

    assert result.exit_code == 1  # leakage found
    report = json.loads(report_file.read_text())
    assert report["splits"]["train"]["size"] == 3
    assert report["splits"]["test"]["size"] == 4
    assert len(report["splits"]["train"]["sha256"]) == 64


def test_clean_splits_pass():
    docs = _docs()
    docs["test"] = docs["test"][-1:]
    assert module.check_leakage(docs, "train")["num_flagged"] == 0
