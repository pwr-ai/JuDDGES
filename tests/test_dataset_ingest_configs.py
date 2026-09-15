"""Guard the AI-TAX eureka ingestion configs against silent mapping faults.

Two bug classes, both of which reached production:

#67 - a typo in a mapping's *source* field. ``_apply_column_mapping`` skips any
source field absent from the row rather than raising, so the target property is
never written and nothing complains.

#68 - a duplicated mapping key. YAML keeps the last occurrence silently, so one
mapping shadows another. In the eureka config ``html_content`` appeared twice,
which dropped ``raw_content`` and put raw HTML into ``full_text``, overwriting
the clean text the dataset derives with BeautifulSoup.

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


class _NoDuplicateKeys(yaml.SafeLoader):
    """SafeLoader that refuses a mapping with a repeated key."""


def _reject_duplicates(loader, node, deep=False):
    seen: set = set()
    duplicates = []
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in seen:
            duplicates.append(key)
        seen.add(key)
    if duplicates:
        raise ValueError(f"duplicate keys: {sorted(duplicates)}")
    return yaml.SafeLoader.construct_mapping(loader, node, deep)


_NoDuplicateKeys.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _reject_duplicates)


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


@pytest.mark.parametrize("config_path", sorted(CONFIG_DIR.glob("*.yaml")), ids=lambda p: p.name)
def test_no_duplicate_keys(config_path):
    """A repeated key is resolved silently by YAML, shadowing the earlier one."""
    try:
        yaml.load(config_path.read_text(encoding="utf-8"), Loader=_NoDuplicateKeys)
    except ValueError as exc:
        pytest.fail(f"{config_path.name}: {exc}")


@pytest.mark.parametrize("config_name", EUREKA_CONFIGS)
def test_clean_text_maps_to_full_text(config_name: str):
    # The dataset derives full_text from the HTML with BeautifulSoup
    # (ai-tax, eureka_fetcher/src/hf_dataset.py). full_text is a vectorized
    # property, so feeding it markup would embed the tags.
    mapping = _column_mapping(config_name)
    assert mapping.get("full_text") == "full_text"


@pytest.mark.parametrize("config_name", EUREKA_CONFIGS)
def test_markup_maps_to_raw_content(config_name: str):
    # Matches JuDDGES_pl-court-raw.yaml, which maps xml_content -> raw_content.
    mapping = _column_mapping(config_name)
    assert mapping.get("html_content") == "raw_content"
