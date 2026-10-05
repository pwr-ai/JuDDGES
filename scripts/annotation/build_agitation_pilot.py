"""Build a blind, case-separated annotation pilot from saved agitation search results."""

import argparse
import hashlib
import json
import random
from pathlib import Path

import pandas as pd


STRATA = ("q1", "q2_4", "q5_9", "q10_plus")


def stratum(query_count: int) -> str:
    if query_count == 1:
        return "q1"
    if query_count < 5:
        return "q2_4"
    if query_count < 10:
        return "q5_9"
    return "q10_plus"


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def build_pilot(merged: pd.DataFrame, count: int, seed: int) -> tuple[list, list]:
    if count <= 0 or count % len(STRATA):
        raise ValueError("count must be a positive multiple of four")
    if merged.uuid.duplicated().any():
        raise ValueError("source contains duplicate UUIDs")

    candidates = {band: [] for band in STRATA}
    for row in merged.itertuples(index=False):
        properties = row.properties
        full_text = properties.get("full_text")
        if not isinstance(full_text, str) or not full_text.strip():
            continue
        query_count = int(row.query_count)
        if query_count < 1:
            continue
        judgment_id = str(properties.get("judgment_id") or row.uuid)
        docket = str(properties.get("docket_number") or "").strip().casefold()
        text_hash = hashlib.sha256(full_text.encode("utf-8")).hexdigest()
        candidates[stratum(query_count)].append(
            (str(row.uuid), judgment_id, docket or judgment_id, full_text, text_hash, query_count)
        )

    rng = random.Random(seed)
    for band in STRATA:
        candidates[band].sort(key=lambda item: item[0])
        rng.shuffle(candidates[band])

    chosen = []
    used_cases = set()
    used_texts = set()
    per_band = count // len(STRATA)
    for band in STRATA:
        taken = 0
        for uuid, judgment_id, case_key, full_text, text_hash, query_count in candidates[band]:
            if case_key in used_cases or text_hash in used_texts:
                continue
            used_cases.add(case_key)
            used_texts.add(text_hash)
            chosen.append((uuid, judgment_id, case_key, full_text, text_hash, query_count, band))
            taken += 1
            if taken == per_band:
                break
        if taken != per_band:
            raise ValueError(f"insufficient distinct cases in {band}: {taken}/{per_band}")

    rng.shuffle(chosen)
    tasks = [
        {"data": {"judgment_id": item[1], "text": item[3]}} for item in chosen
    ]
    manifest_rows = [
        {
            "task_number": index,
            "uuid": item[0],
            "judgment_id": item[1],
            "case_key": item[2],
            "text_sha256": item[4],
            "query_count": item[5],
            "stratum": item[6],
        }
        for index, item in enumerate(chosen, start=1)
    ]
    return tasks, manifest_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merged", type=Path, required=True)
    parser.add_argument("--extractions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261005)
    args = parser.parse_args()

    if args.output.exists():
        raise SystemExit(f"output already exists: {args.output}")
    merged = pd.read_pickle(args.merged)
    tasks, rows = build_pilot(merged, args.count, args.seed)
    extractions = pd.read_pickle(args.extractions)
    extracted_by_uuid = extractions.set_index("uuid").extracted_informations.to_dict()
    weak_labels = [
        {
            "task_number": row["task_number"],
            "uuid": row["uuid"],
            "is_agitation_case": extracted_by_uuid[row["uuid"]].get("is_agitation_case"),
            "provenance": "existing extraction; unverified weak label",
        }
        for row in rows
        if row["uuid"] in extracted_by_uuid
        and isinstance(extracted_by_uuid[row["uuid"]], dict)
    ]

    args.output.mkdir(parents=True)
    (args.output / "tasks.json").write_text(
        json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output / "sampling_manifest.json").write_text(
        json.dumps(
            {
                "source_sha256": digest(args.merged),
                "extractions_sha256": digest(args.extractions),
                "seed": args.seed,
                "count": args.count,
                "rows": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (args.output / "weak_labels.json").write_text(
        json.dumps(weak_labels, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Created {len(tasks)} blind tasks and {len(weak_labels)} separate weak labels in {args.output}")


if __name__ == "__main__":
    main()
