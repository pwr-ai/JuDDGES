"""Reports sizes and sha256 of dataset splits and flags documents shared between splits.

Flags (train vs. every evaluation split): exact duplicates, duplicates after text normalization,
near duplicates (Jaccard similarity of word shingles) and the same case signature (e.g. "I C 123/19").
"""

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import typer

SIGNATURE_PATTERN = re.compile(r"\b([IVX]+)\s+([A-Z][A-Za-z]{0,3})\s+(\d+)\s*/\s*(\d{2,4})\b")
SIGNATURE_SEARCH_CHARS = 3000
SHINGLE_SIZE = 5


def main(
    dataset_dir: Path = typer.Option(Path("data/datasets/pl/swiss_franc_loans")),
    train_split: str = typer.Option("train"),
    eval_splits: list[str] = typer.Option(["test", "annotated"]),
    context_field: str = typer.Option("context"),
    threshold: float = typer.Option(0.8, help="Jaccard similarity flagged as near duplicate"),
    report_file: Path | None = typer.Option(None, help="Optional path for the JSON report"),
) -> None:
    splits = {}
    files = {}
    for name in [train_split, *eval_splits]:
        files[name] = dataset_dir / f"{name}.json"
        with files[name].open() as f:
            splits[name] = [item[context_field] for item in json.load(f)]

    report = check_leakage(splits, train_split, threshold)
    report["splits"] = {
        name: {"size": len(splits[name]), "sha256": _file_sha256(files[name])} for name in splits
    }
    report_str = json.dumps(report, indent=2, ensure_ascii=False)
    typer.echo(report_str)
    if report_file is not None:
        report_file.write_text(report_str)
    if report["num_flagged"]:
        raise typer.Exit(code=1)


def check_leakage(
    splits: dict[str, list[str]], train_split: str, threshold: float = 0.8
) -> dict[str, Any]:
    train_docs = splits[train_split]
    train_norm = [_normalize(doc) for doc in train_docs]
    train_hashes = {}
    for idx, text in enumerate(train_norm):
        train_hashes.setdefault(_text_hash(text), []).append(idx)
    train_shingles = [_shingles(text) for text in train_norm]
    train_signatures: dict[str, list[int]] = {}
    for idx, doc in enumerate(train_docs):
        if (signature := _signature(doc)) is not None:
            train_signatures.setdefault(signature, []).append(idx)

    flagged = []
    for split_name, docs in splits.items():
        if split_name == train_split:
            continue
        for idx, doc in enumerate(docs):
            text = _normalize(doc)
            reasons: dict[str, Any] = {}
            if matches := train_hashes.get(_text_hash(text)):
                reasons["identical_after_normalization"] = matches
            shingles = _shingles(text)
            near = {
                train_idx: round(score, 3)
                for train_idx, train_sh in enumerate(train_shingles)
                if (score := _jaccard(shingles, train_sh)) >= threshold
            }
            if near:
                reasons["near_duplicate_jaccard"] = near
            if (signature := _signature(doc)) is not None and signature in train_signatures:
                reasons["same_signature"] = {signature: train_signatures[signature]}
            if reasons:
                flagged.append({"split": split_name, "index": idx, "train_matches": reasons})

    return {"jaccard_threshold": threshold, "num_flagged": len(flagged), "flagged": flagged}


def _normalize(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


def _text_hash(normalized_text: str) -> str:
    return hashlib.sha256(normalized_text.encode()).hexdigest()


def _shingles(normalized_text: str) -> set[int]:
    words = normalized_text.split()
    if len(words) <= SHINGLE_SIZE:
        return {hash(" ".join(words))}
    return {
        hash(" ".join(words[i : i + SHINGLE_SIZE])) for i in range(len(words) - SHINGLE_SIZE + 1)
    }


def _jaccard(a: set[int], b: set[int]) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def _signature(text: str) -> str | None:
    match = SIGNATURE_PATTERN.search(text[:SIGNATURE_SEARCH_CHARS])
    if match is None:
        return None
    roman, kind, number, year = match.groups()
    return f"{roman} {kind} {int(number)}/{year[-2:]}"


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    typer.run(main)
