import pprint
import re
from itertools import product
from typing import Any

from datasets import load_dataset

from juddges.settings import DATA_PATH

COURT_FILTER = [r"Sąd Okręgowy"]

LEGAL_BASES_REGEX_PATTERNS = [
    r"(?:^|[^0-9])23\s*k\.?c\.?\b",  # matches "23 k.c.", "art.23 kc", "art. 23 k.c.", etc.
    r"(?:^|[^0-9])24\s*k\.?c\.?\b",  # matches "24 k.c.", "art.24 kc", "art. 24 k.c.", etc.
    r"(?:^|[^0-9])448\s*k\.?c\.?\b",  # matches "448 k.c.", "art.448 kc", "art. 448 k.c.", etc.
]

OUTPUT_PATH = DATA_PATH / "analysis" / "personal_rights" / "samples"


def main():
    ds = load_dataset("JuDDGES/pl-court-raw", split="train")
    ds = ds.filter(filter_data, num_proc=6)
    ds = ds.select(range(200))

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    for row in ds:
        row.pop("xml_content")
        text = pprint.pformat(row, width=120)
        with open(OUTPUT_PATH / f"{row['judgment_id']}.json", "w") as f:
            f.write(text)


def filter_data(entry: dict[str, Any]) -> bool:
    if entry["court_name"] is None:
        return False

    if entry["legal_bases"] is None:
        return False
    court_filter = any(
        re.search(pattern, entry["court_name"], re.IGNORECASE) for pattern in COURT_FILTER
    )

    legal_bases_filter = False
    for legal_base, pattern in product(entry["legal_bases"], LEGAL_BASES_REGEX_PATTERNS):
        if re.search(pattern, legal_base, re.IGNORECASE):
            legal_bases_filter = True
            break

    return court_filter and legal_bases_filter


if __name__ == "__main__":
    main()
