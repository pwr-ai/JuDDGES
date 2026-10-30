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
    signatures = {item["index"]: item["signature_matches"] for item in report["flagged"]}
    assert signatures[0] == {"I C 123/19": [0]}
    assert signatures[1] == {}
    assert flagged[2] == {}  # signature hit only, reported separately
    assert signatures[2] == {"II Ca 5/20": [2]}
    assert report["num_flagged"] == 3
    assert report["num_flagged_text"] == 2
    assert report["num_flagged_signature"] == 2


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


def test_containment_flags_short_document_copied_into_long_train_document():
    train_doc = " ".join(f"zdanie{i}" for i in range(400))
    short_copy = " ".join(f"zdanie{i}" for i in range(100, 160))
    docs = {"train": [train_doc], "test": [short_copy]}

    report = module.check_leakage(docs, "train", threshold=0.8, containment_threshold=0.5)

    matches = report["flagged"][0]["train_matches"]
    assert "near_duplicate_jaccard" not in matches  # Jaccard is low: the train document is longer
    assert matches["contained_in_train"] == {0: 1.0}
    assert module.check_leakage(docs, "train", containment_threshold=1.1)["num_flagged"] == 0


def test_signature_is_found_beyond_first_3000_chars():
    filler = " ".join(f"wypelniacz{i}" for i in range(600))  # > 3000 chars
    docs = {
        "train": ["pierwszy " + filler + " sygn. akt I C 9/21"],
        "test": ["drugi " + filler[::-1] + " sygn. akt I C 9/21"],
    }

    report = module.check_leakage(docs, "train")

    assert report["flagged"][0]["signature_matches"] == {"I C 9/21": [0]}
    assert report["num_flagged_text"] == 0


def test_signature_is_keyed_by_court_when_available():
    docs = {
        "train": ["Sygn. akt I C 1/20 " + " ".join(f"a{i}" for i in range(50))],
        "test": ["Sygn. akt I C 1/20 " + " ".join(f"b{i}" for i in range(50))],
    }

    same = module.check_leakage(
        docs, "train", courts={"train": ["SR Wroclaw"], "test": ["SR Wroclaw"]}
    )
    other = module.check_leakage(
        docs, "train", courts={"train": ["SR Wroclaw"], "test": ["SR Krakow"]}
    )

    assert same["num_flagged_signature"] == 1
    assert other["num_flagged"] == 0


def test_empty_texts_are_skipped():
    docs = {"train": ["", "  "], "test": ["", "...", "tekst " * 20]}

    report = module.check_leakage(docs, "train")

    assert report["num_flagged"] == 0
    assert report["num_empty_texts_skipped"] == 4


def test_help_states_limitation():
    app = module.typer.Typer(add_completion=False)
    app.command()(module.main)

    result = CliRunner().invoke(app, ["--help"])

    assert "appeal" in result.output and "does not detect" in result.output
