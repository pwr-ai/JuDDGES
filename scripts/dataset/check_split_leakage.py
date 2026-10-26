"""Reports sizes and sha256 of dataset splits and flags documents shared between splits.

Text hits (train vs. every evaluation split): exact duplicates, duplicates after text
normalization, near duplicates (Jaccard similarity of word shingles) and containment (share of
an evaluation document's shingles found in one train document, a lower threshold than Jaccard
because a short document copied into a long one has a low Jaccard). Signature hits (the same
case signature, e.g. "I C 123/19", keyed by court when the court field exists) are reported
separately. Empty texts are skipped.
"""

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import typer

SIGNATURE_PATTERN = re.compile(r"\b([IVX]+)\s+([A-Z][A-Za-z]{0,3})\s+(\d+)\s*/\s*(\d{2,4})\b")
SHINGLE_SIZE = 5


def main(
    dataset_dir: Path = typer.Option(Path("data/datasets/pl/swiss_franc_loans")),
    train_split: str = typer.Option("train"),
    eval_splits: list[str] = typer.Option(["test", "annotated"]),
    context_field: str = typer.Option("context"),
    court_field: str = typer.Option("court", help="Signatures are keyed by court if present"),
    threshold: float = typer.Option(0.8, help="Jaccard similarity flagged as near duplicate"),
    containment_threshold: float = typer.Option(
        0.5, help="Share of an eval document's shingles found in one train document"
    ),
    report_file: Path | None = typer.Option(None, help="Optional path for the JSON report"),
) -> None:
    """Checks train/evaluation leakage by text and by case signature.

    Limitation: a first-instance judgment and its appeal have different signatures and different
    texts, so this check does not detect such pairs. If the data has fields for the parties or
    the number of the case, compare them as well (same parties, same case number).
    """
    splits = {}
    courts = {}
    files = {}
    for name in [train_split, *eval_splits]:
        files[name] = dataset_dir / f"{name}.json"
        with files[name].open() as f:
            items = json.load(f)
        splits[name] = [item[context_field] for item in items]
        courts[name] = [item.get(court_field) for item in items]

    report = check_leakage(splits, train_split, threshold, containment_threshold, courts)
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
    splits: dict[str, list[str]],
    train_split: str,
    threshold: float = 0.8,
    containment_threshold: float = 0.5,
    courts: dict[str, list[str | None]] | None = None,
) -> dict[str, Any]:
    def court_of(split_name: str, idx: int) -> str | None:
        return courts[split_name][idx] if courts else None

    train_docs = splits[train_split]
    train_norm = [_normalize(doc) for doc in train_docs]
    train_hashes: dict[str, list[int]] = {}
    for idx, text in enumerate(train_norm):
        if text:
            train_hashes.setdefault(_text_hash(text), []).append(idx)
    train_shingles = {idx: _shingles(text) for idx, text in enumerate(train_norm) if text}
    train_signatures: dict[tuple[str | None, str], list[int]] = {}
    for idx, doc in enumerate(train_docs):
        if train_norm[idx] and (signature := _signature(doc)) is not None:
            train_signatures.setdefault((court_of(train_split, idx), signature), []).append(idx)

    flagged = []
    num_empty = sum(not text for text in train_norm)
    for split_name, docs in splits.items():
        if split_name == train_split:
            continue
        for idx, doc in enumerate(docs):
            text = _normalize(doc)
            if not text:
                num_empty += 1
                continue
            text_matches: dict[str, Any] = {}
            if matches := train_hashes.get(_text_hash(text)):
                text_matches["identical_after_normalization"] = matches
            shingles = _shingles(text)
            near, contained = {}, {}
            for train_idx, train_sh in train_shingles.items():
                if (score := _jaccard(shingles, train_sh)) >= threshold:
                    near[train_idx] = round(score, 3)
                if (score := _containment(shingles, train_sh)) >= containment_threshold:
                    contained[train_idx] = round(score, 3)
            if near:
                text_matches["near_duplicate_jaccard"] = near
            if contained:
                text_matches["contained_in_train"] = contained
            signature_matches: dict[str, Any] = {}
            if (signature := _signature(doc)) is not None:
                key = (court_of(split_name, idx), signature)
                if key in train_signatures:
                    signature_matches[signature] = train_signatures[key]
            if text_matches or signature_matches:
                flagged.append(
                    {
                        "split": split_name,
                        "index": idx,
                        "train_matches": text_matches,
                        "signature_matches": signature_matches,
                    }
                )

    return {
        "jaccard_threshold": threshold,
        "containment_threshold": containment_threshold,
        "num_empty_texts_skipped": num_empty,
        "num_flagged": len(flagged),
        "num_flagged_text": sum(bool(item["train_matches"]) for item in flagged),
        "num_flagged_signature": sum(bool(item["signature_matches"]) for item in flagged),
        "flagged": flagged,
    }


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


def _containment(a: set[int], b: set[int]) -> float:
    """Share of the shingles of `a` that also occur in `b`: |A and B| / |A|."""
    return len(a & b) / len(a) if a else 0.0


def _signature(text: str) -> str | None:
    match = SIGNATURE_PATTERN.search(text)
    if match is None:
        return None
    roman, kind, number, year = match.groups()
    return f"{roman} {kind} {int(number)}/{year[-2:]}"


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    typer.run(main)
