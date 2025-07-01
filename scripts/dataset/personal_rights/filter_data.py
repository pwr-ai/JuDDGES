import re
from itertools import product
from typing import Any

import pandas as pd
from datasets import load_dataset

from juddges.settings import DATA_PATH

COURT_FILTER = [r"Sąd Okręgowy"]

LEGAL_BASES_REGEX_PATTERNS = [
    r"(?:^|[^0-9])23\s*k\.?c\.?\b",  # matches "23 k.c.", "art.23 kc", "art. 23 k.c.", etc.
    r"(?:^|[^0-9])24\s*k\.?c\.?\b",  # matches "24 k.c.", "art.24 kc", "art. 24 k.c.", etc.
]

OUTPUT_PATH = DATA_PATH / "analysis" / "personal_rights" / "samples"

SKIP_FIELDS = [
    "excerpt",
    "thesis",
    "decision",
    "xml_content",
    "source",
    "judgment_id",
    "docket_number",
    "judgment_date",
    "publication_date",
    "last_update",
    "court_id",
    "department_id",
    "judgment_type",
    "presiding_judge",
    "judges",
    "publisher",
    "reviser",
    "recorder",
    "num_pages",
    "volume_number",
    "volume_type",
    "country",
]

FIELDS_FIRST = ["full_text"]


def main():
    ds = load_dataset("JuDDGES/pl-court-raw", split="train")
    ds = ds.filter(filter_data, num_proc=6)

    output = []
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    for row in ds:
        text = ""
        keys = FIELDS_FIRST + [k for k in row.keys() if k not in FIELDS_FIRST]
        for k in keys:
            v = row[k]
            if k in SKIP_FIELDS:
                continue
            text += "=" * 80 + "\n"
            text += f"{k}\n"
            text += "=" * 80 + "\n"
            text += f"{v}\n\n"
        output.append(text)

    output_df = pd.DataFrame({"text": output})
    output_df.to_csv(OUTPUT_PATH / "filtered_data.csv", index=False)


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
