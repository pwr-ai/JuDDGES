"""Unit tests for the Weaviate v2 information-extraction (IE) property generator. No network."""

from pathlib import Path

import pytest

import weaviate.classes.config as wvcc
from juddges.data.weaviate_v2.ie_properties import (
    DEFAULT_POLICY,
    FlatCopyPolicy,
    IeKind,
    build_ie_payload,
    ie_chunk_payload,
    ie_chunk_properties,
    ie_properties,
    load_ie_schema,
    parse_ie_schema,
    sanitize_name,
)
from juddges.settings import CONFIG_PATH

IE_SCHEMA_FILES = sorted((CONFIG_PATH / "ie_schema").glob("*.yaml"))
PROP_NAME_RE = r"^[_A-Za-z][_0-9A-Za-z]*$"


def _by_name(props: list[wvcc.Property]) -> dict[str, wvcc.Property]:
    return {p.name: p for p in props}


def test_ie_schema_files_found() -> None:
    assert len(IE_SCHEMA_FILES) >= 5


@pytest.mark.parametrize("path", IE_SCHEMA_FILES, ids=lambda p: p.stem)
def test_real_schema_generates_valid_properties(path: Path) -> None:
    import re

    schema = load_ie_schema(path)
    props = ie_properties(schema)
    names = [p.name for p in props]

    assert len(names) == len(set(names)), "property names must be unique"
    assert all(re.match(PROP_NAME_RE, n) for n in names)

    by_name = _by_name(props)
    obj = by_name[f"ie_{path.stem}"]
    assert obj.dataType == wvcc.DataType.OBJECT
    nested = {p.name for p in obj.nestedProperties}
    assert nested == {k.prop for k in schema.keys}
    for n in nested:
        assert re.match(PROP_NAME_RE, n)

    # meta properties
    assert by_name[f"ie_{path.stem}_meta__extracted_at"].dataType == wvcc.DataType.DATE
    assert by_name[f"ie_{path.stem}_meta__model"].dataType == wvcc.DataType.TEXT
    assert f"ie_{path.stem}_meta__schema_version" in by_name

    # every flat copy is filterable and has a Weaviate-valid name
    for k in schema.keys:
        flat = f"ie_{path.stem}__{k.prop}"
        if k.flat_copy:
            assert by_name[flat].indexFilterable is True
        else:
            assert flat not in by_name


@pytest.mark.parametrize("path", IE_SCHEMA_FILES, ids=lambda p: p.stem)
def test_nested_properties_are_not_indexed(path: Path) -> None:
    """Nested paths cannot be filtered in Weaviate 1.40, so indexing them is wasted space."""
    obj = ie_properties(load_ie_schema(path))[0]
    for p in obj.nestedProperties:
        assert p.indexFilterable is False
        assert p.indexSearchable is False


def test_personal_rights_types() -> None:
    schema = load_ie_schema(CONFIG_PATH / "ie_schema" / "personal_rights.yaml")
    props = _by_name(ie_properties(schema))
    flat = "ie_personal_rights__"

    # enum -> text, FIELD tokenization, filterable
    p = props[flat + "naruszenie_dobr_osobistych"]
    assert p.dataType == wvcc.DataType.TEXT
    assert p.tokenization == wvcc.Tokenization.FIELD
    assert p.indexSearchable is False

    # integer -> int with range filters
    p = props[flat + "skala_naruszenia"]
    assert p.dataType == wvcc.DataType.INT
    assert p.indexRangeFilters is True

    # list of enums -> text[] FIELD
    p = props[flat + "podstawa_prawna"]
    assert p.dataType == wvcc.DataType.TEXT_ARRAY
    assert p.tokenization == wvcc.Tokenization.FIELD

    # free text and list of free text are only in the object
    assert flat + "opis_naruszenia" not in props
    assert flat + "inne_podstawy_prawne" not in props
    nested = _by_name(props["ie_personal_rights"].nestedProperties)
    assert nested["opis_naruszenia"].dataType == wvcc.DataType.TEXT
    assert nested["inne_podstawy_prawne"].dataType == wvcc.DataType.TEXT_ARRAY


def test_number_type_gets_number_property() -> None:
    schema = load_ie_schema(CONFIG_PATH / "ie_schema" / "personal_rights_refined.yaml")
    props = _by_name(ie_properties(schema))
    assert props["ie_personal_rights_refined__skala_naruszenia"].dataType == wvcc.DataType.NUMBER


def test_date_like_strings_become_date_with_range_filters() -> None:
    schema = load_ie_schema(CONFIG_PATH / "ie_schema" / "swiss_franc_loans.yaml")
    props = _by_name(ie_properties(schema))
    for key in ("data_wyroku", "data_aneksu"):
        p = props[f"ie_swiss_franc_loans__{key}"]
        assert p.dataType == wvcc.DataType.DATE
        assert p.indexRangeFilters is True
    # "data_rozpoczecia_odsetek" is an enum despite its name
    assert props["ie_swiss_franc_loans__data_rozpoczecia_odsetek"].dataType == wvcc.DataType.TEXT


def test_list_of_dates_and_ints_in_object_only() -> None:
    schema = load_ie_schema(CONFIG_PATH / "ie_schema" / "en_appealcourt.yaml")
    props = _by_name(ie_properties(schema))
    nested = _by_name(props["ie_en_appealcourt"].nestedProperties)
    assert nested["convictPleaDate"].dataType == wvcc.DataType.DATE_ARRAY
    assert nested["remandCustodyTime"].dataType == wvcc.DataType.INT_ARRAY
    assert "ie_en_appealcourt__convictPleaDate" not in props


def test_chunk_properties_default_only_enums() -> None:
    schema = load_ie_schema(CONFIG_PATH / "ie_schema" / "personal_rights.yaml")
    chunk_props = ie_chunk_properties(schema)
    assert chunk_props
    flat_by_name = _by_name(ie_properties(schema))
    for p in chunk_props:
        assert p.name in flat_by_name
        assert p.dataType == wvcc.DataType.TEXT  # scalar enums only
        assert p.tokenization == wvcc.Tokenization.FIELD
    assert "ie_personal_rights__skala_naruszenia" not in {p.name for p in chunk_props}


def test_chunk_policy_can_include_more_kinds() -> None:
    policy = FlatCopyPolicy(chunk_kinds=frozenset({IeKind.ENUM, IeKind.INTEGER, IeKind.DATE}))
    schema = load_ie_schema(CONFIG_PATH / "ie_schema" / "personal_rights.yaml", policy)
    assert "ie_personal_rights__skala_naruszenia" in {p.name for p in ie_chunk_properties(schema)}
    assert DEFAULT_POLICY.chunk_kinds == frozenset({IeKind.ENUM})


def test_policy_include_exclude_and_kind_override() -> None:
    spec = {
        "a_enum": {"type": "enum", "choices": ["x", "y"], "description": "d"},
        "free": {"type": "string", "description": "free text"},
        "when": {"type": "string", "description": "no hint here"},
    }
    policy = FlatCopyPolicy(
        include=frozenset({"free"}),
        exclude=frozenset({"a_enum"}),
        kind_overrides={"when": IeKind.DATE},
    )
    schema = parse_ie_schema("t", spec, policy)
    props = _by_name(ie_properties(schema))
    assert "ie_t__a_enum" not in props
    assert props["ie_t__free"].dataType == wvcc.DataType.TEXT
    assert props["ie_t__when"].dataType == wvcc.DataType.DATE


def test_list_enum_copy_can_be_disabled() -> None:
    spec = {"tags": {"type": "list", "items": {"type": "enum", "choices": ["a"]}}}
    assert "ie_t__tags" in _by_name(ie_properties(parse_ie_schema("t", spec)))
    off = parse_ie_schema("t", spec, FlatCopyPolicy(list_enum=False))
    assert "ie_t__tags" not in _by_name(ie_properties(off))


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("simple", "simple"),
        ("Zażółć_gęślą", "Zazolc_gesla"),
        ("ŁÓDŹ", "LODZ"),
        ("with space-and.dots", "with_space_and_dots"),
        ("a__b", "a_b"),
        ("1st", "_1st"),
        ("  trim  ", "trim"),
    ],
)
def test_sanitize_name(raw: str, expected: str) -> None:
    assert sanitize_name(raw, lower_first=False) == expected


def test_sanitize_lowercases_first_letter_by_default() -> None:
    assert sanitize_name("ConvictPleaDate") == "convictPleaDate"


def test_sanitize_rejects_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        sanitize_name("!!!")


def test_collision_fails_loudly() -> None:
    spec = {
        "Typ sądu": {"type": "string", "description": "a"},
        "typ_sadu": {"type": "string", "description": "b"},
    }
    with pytest.raises(ValueError, match="collision"):
        parse_ie_schema("t", spec)


def test_unknown_type_fails_loudly() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        parse_ie_schema("t", {"k": {"type": "matrix"}})


def test_list_without_items_fails_loudly() -> None:
    with pytest.raises(ValueError, match="items"):
        parse_ie_schema("t", {"k": {"type": "list"}})


def test_schema_name_is_sanitized() -> None:
    schema = parse_ie_schema("My Schema-1", {"k": {"type": "string"}})
    assert ie_properties(schema)[0].name == "ie_My_Schema_1"


def test_enum_choices_ignore_null() -> None:
    spec = {"k": {"type": "enum", "choices": ["a", None]}}
    assert parse_ie_schema("t", spec).keys[0].choices == ("a",)


# --- payload builder ---------------------------------------------------------------------------


@pytest.fixture
def franc_schema():
    return load_ie_schema(CONFIG_PATH / "ie_schema" / "swiss_franc_loans.yaml")


def test_payload_coerces_and_drops(franc_schema) -> None:
    result = {
        "data_wyroku": "2021-03-05",
        "data_aneksu": "not a date",
        "typ_sadu": None,
        "nonexistent_key": "x",
        "kwota_kredytu": "",
    }
    payload = build_ie_payload(
        franc_schema,
        result,
        schema_version="1",
        model="gemini-x",
        extracted_at="2025-01-02T03:04:05+00:00",
    )
    obj = payload["ie_swiss_franc_loans"]
    assert obj["data_wyroku"] == "2021-03-05T00:00:00Z"
    assert "data_aneksu" not in obj  # unparseable -> dropped
    assert "typ_sadu" not in obj and "kwota_kredytu" not in obj
    assert "nonexistent_key" not in obj
    assert payload["ie_swiss_franc_loans__data_wyroku"] == "2021-03-05T00:00:00Z"
    assert "ie_swiss_franc_loans__data_aneksu" not in payload
    assert payload["ie_swiss_franc_loans_meta__model"] == "gemini-x"
    assert payload["ie_swiss_franc_loans_meta__schema_version"] == "1"
    assert payload["ie_swiss_franc_loans_meta__extracted_at"] == "2025-01-02T03:04:05Z"


def test_payload_enum_stays_text_and_flat_copy_matches_object() -> None:
    schema = load_ie_schema(CONFIG_PATH / "ie_schema" / "personal_rights.yaml")
    payload = build_ie_payload(
        schema,
        {
            "naruszenie_dobr_osobistych": "Tak",
            "skala_naruszenia": "4",
            "podstawa_prawna": ["23 KC", "24 KC", None],
            "opis_naruszenia": "cytat",
        },
    )
    obj = payload["ie_personal_rights"]
    assert obj["naruszenie_dobr_osobistych"] == "Tak"
    assert obj["skala_naruszenia"] == 4 and isinstance(obj["skala_naruszenia"], int)
    assert obj["podstawa_prawna"] == ["23 KC", "24 KC"]
    assert payload["ie_personal_rights__naruszenie_dobr_osobistych"] == "Tak"
    assert payload["ie_personal_rights__skala_naruszenia"] == 4
    assert payload["ie_personal_rights__podstawa_prawna"] == ["23 KC", "24 KC"]
    assert "ie_personal_rights__opis_naruszenia" not in payload
    assert not any("_meta__" in k for k in payload)  # no meta given


def test_payload_empty_result_gives_empty_payload() -> None:
    schema = load_ie_schema(CONFIG_PATH / "ie_schema" / "personal_rights.yaml")
    assert build_ie_payload(schema, {}) == {}
    assert build_ie_payload(schema, {"opis_naruszenia": "  "}) == {}


def test_payload_bool_is_not_an_int() -> None:
    schema = load_ie_schema(CONFIG_PATH / "ie_schema" / "personal_rights.yaml")
    assert build_ie_payload(schema, {"skala_naruszenia": True}) == {}


def test_payload_list_of_ints_and_dates() -> None:
    schema = load_ie_schema(CONFIG_PATH / "ie_schema" / "en_appealcourt.yaml")
    payload = build_ie_payload(
        schema,
        {"ConvictPleaDate": ["2003-01-22", "bad"], "RemandCustodyTime": ["4", 7.0, "x"]},
    )
    obj = payload["ie_en_appealcourt"]
    assert obj["convictPleaDate"] == ["2003-01-22T00:00:00Z"]
    assert obj["remandCustodyTime"] == [4, 7]


def test_payload_scalar_given_for_list_is_wrapped() -> None:
    schema = parse_ie_schema(
        "t", {"tags": {"type": "list", "items": {"type": "enum", "choices": ["a", "b"]}}}
    )
    assert build_ie_payload(schema, {"tags": "a"})["ie_t"] == {"tags": ["a"]}


def test_chunk_payload_is_subset_of_flat_enum_copies() -> None:
    schema = load_ie_schema(CONFIG_PATH / "ie_schema" / "personal_rights.yaml")
    payload = build_ie_payload(
        schema,
        {
            "naruszenie_dobr_osobistych": "Tak",
            "skala_naruszenia": 3,
            "rodzaj_naruszajacego": "Osoba prywatna",
        },
    )
    chunk_payload = ie_chunk_payload(schema, payload)
    assert chunk_payload == {
        "ie_personal_rights__naruszenie_dobr_osobistych": "Tak",
        "ie_personal_rights__rodzaj_naruszajacego": "Osoba prywatna",
    }
