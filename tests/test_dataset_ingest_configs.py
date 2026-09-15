"""Guard the AI-TAX eureka ingestion configs against field-name typos (#67).

``column_mapping`` is applied by ``StreamingIngester._apply_column_mapping``,
which skips any source field absent from the row rather than raising. A typo in
a source field name therefore produces no exception and no empty column — the
target property is simply never written — so it survives ingestion of the whole
corpus unnoticed. That is how ``docker_number`` left ``document_number`` unset
on every Polish tax interpretation.

These tests pin the mapping's source side to the fields the dataset actually
declares, so the next typo fails here instead of in the index.
"""

from pathlib import Path

import pytest
import yaml

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "datasets"

EUREKA_CONFIGS = [
    "AI-TAX_pl-eureka-raw.yaml",
    "AI-TAX_pl-eureka-raw-sample.yaml",
]

# Field names declared by AI-TAX/pl-eureka-raw, built by the eureka fetcher's
# NAMES_MAPPING / FEATURES (ai-tax repo, eureka_fetcher/src/hf_dataset.py).
# "docket_number" is the SYG column - the sygnatura - not "docker_number".
EUREKA_DATASET_FIELDS = frozenset(
    {
        "id",
        "category",
        "status",
        "title",
        "author",
        "publication_date",
        "docket_number",
        "keywords",
        "regulations",
        "html_content",
        "full_text",
        "introduction",
        "scope",
        "factual_state",
        "question",
        "position",
        "assessment",
        "justification",
        "additional_information",
    }
)


def _column_mapping(config_name: str) -> dict[str, str]:
    config = yaml.safe_load((CONFIG_DIR / config_name).read_text(encoding="utf-8"))
    return config["column_mapping"]


@pytest.mark.parametrize("config_name", EUREKA_CONFIGS)
def test_mapping_sources_exist_in_the_dataset(config_name: str):
    mapping = _column_mapping(config_name)
    unknown = sorted(set(mapping) - EUREKA_DATASET_FIELDS)
    assert not unknown, (
        f"{config_name} maps source fields the dataset does not declare: {unknown}. "
        "_apply_column_mapping skips these silently, so the target property is "
        "never written."
    )


@pytest.mark.parametrize("config_name", EUREKA_CONFIGS)
def test_sygnatura_reaches_document_number(config_name: str):
    mapping = _column_mapping(config_name)
    assert mapping.get("docket_number") == "document_number"
