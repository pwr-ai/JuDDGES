"""Unit tests for the Weaviate v2 information-extraction (IE) property generator. No network."""

from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

import weaviate.classes.config as wvcc
from juddges.data.weaviate_v2.ie_properties import (
    DEFAULT_POLICY,
    FlatCopyPolicy,
    IeKind,
    add_ie_schema,
    add_missing_properties,
    build_ie_payload,
    check_ie_collisions,
    existing_ie_schema_names,
    ie_chunk_payload,
    ie_chunk_properties,
    ie_properties,
    load_ie_schema,
    parse_ie_schema,
    property_drift,
    sanitize_name,
    to_rfc3339,
)
from juddges.settings import CONFIG_PATH

IE_DIR = CONFIG_PATH / "ie_schema"
IE_SCHEMA_FILES = sorted(IE_DIR.glob("*.yaml"))
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


# --- int64 handling ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("4", 4),
        (" -7 ", -7),
        ("4.0", 4),
        (7.0, 7),
        (2**63 - 1, 2**63 - 1),
        ("9223372036854775807", 2**63 - 1),  # float() would round this to 2**63
        ("1_000", None),
        ("1e20", None),
        ("1e3", 1000),
        (2**63, None),
        ("9223372036854775808", None),
        (-(2**63) - 1, None),
        (1e20, None),
        ("1e999999999", None),
        ("4.5", None),
        (float("nan"), None),
        (float("inf"), None),
        ("abc", None),
        (True, None),
    ],
)
def test_integer_coercion(raw, expected) -> None:
    schema = parse_ie_schema("t", {"n": {"type": "integer"}})
    payload = build_ie_payload(schema, {"n": raw})
    assert payload.get("ie_t__n") == expected
    if expected is None:
        assert payload == {}


def test_large_int_keeps_precision_in_lists() -> None:
    schema = parse_ie_schema("t", {"n": {"type": "list", "items": {"type": "int"}}})
    value = 2**62 + 1
    assert build_ie_payload(schema, {"n": [value, str(value), 2**64]})["ie_t"] == {
        "n": [value, value]
    }


def test_number_still_uses_float() -> None:
    schema = parse_ie_schema("t", {"n": {"type": "number"}})
    assert build_ie_payload(schema, {"n": "1e20"})["ie_t"] == {"n": 1e20}
    assert build_ie_payload(schema, {"n": "nan"}) == {}


# --- dates -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2021-03-05", "2021-03-05T00:00:00Z"),
        ("2021-03-05T10:00:00Z", "2021-03-05T10:00:00Z"),
        ("2021-03-05T10:00:00+02:00", "2021-03-05T08:00:00Z"),
        ("2021-03-05T23:30:00-05:00", "2021-03-06T04:30:00Z"),
        ("2021-03-05 10:00:00", "2021-03-05T10:00:00Z"),  # naive = UTC
        (datetime(2021, 3, 5, 10, 0, tzinfo=timezone.utc), "2021-03-05T10:00:00Z"),
        (date(2021, 3, 5), "2021-03-05T00:00:00Z"),
        ("2021-13-45", None),
        ("", None),
        (20210305, None),
        # astimezone() overflows for the earliest/latest representable instants
        ("0001-01-01T00:00:00+05:00", None),
        ("9999-12-31T23:59:59-05:00", None),
    ],
)
def test_to_rfc3339(raw, expected) -> None:
    assert to_rfc3339(raw) == expected


# --- chunk allowlist ---------------------------------------------------------------------------


def _chunk_names(schema) -> set[str]:
    return {p.name for p in ie_chunk_properties(schema)}


def test_chunk_keys_allowlist_replaces_kind_rule() -> None:
    spec = {
        "a": {"type": "enum", "choices": ["x"]},
        "b": {"type": "enum", "choices": ["y"]},
        "n": {"type": "integer"},
    }
    default = parse_ie_schema("t", spec)
    assert _chunk_names(default) == {"ie_t__a", "ie_t__b"}

    only_a = parse_ie_schema("t", spec, FlatCopyPolicy(chunk_keys={"t": frozenset({"a"})}))
    assert _chunk_names(only_a) == {"ie_t__a"}

    with_int = parse_ie_schema("t", spec, FlatCopyPolicy(chunk_keys={"t": frozenset({"a", "n"})}))
    assert _chunk_names(with_int) == {"ie_t__a", "ie_t__n"}

    none = parse_ie_schema("t", spec, FlatCopyPolicy(chunk_keys={"t": frozenset()}))
    assert _chunk_names(none) == set()

    other_schema = parse_ie_schema("u", spec, FlatCopyPolicy(chunk_keys={"t": frozenset()}))
    assert len(_chunk_names(other_schema)) == 2  # allowlist is per schema


def test_chunk_keys_rejects_unknown_and_non_scalar_keys() -> None:
    spec = {
        "tags": {"type": "list", "items": {"type": "enum", "choices": ["a"]}},
        "free": {"type": "string"},
    }
    with pytest.raises(ValueError, match="unknown keys"):
        parse_ie_schema("t", spec, FlatCopyPolicy(chunk_keys={"t": frozenset({"nope"})}))
    for key in ("tags", "free"):
        with pytest.raises(ValueError, match="not scalar keys with a flat copy"):
            parse_ie_schema("t", spec, FlatCopyPolicy(chunk_keys={"t": frozenset({key})}))


# --- cross-schema collisions -------------------------------------------------------------------


def test_meta_vs_key_collision_across_schemas() -> None:
    a = parse_ie_schema("a", {"x": {"type": "enum", "choices": ["1"]}})
    a_meta = parse_ie_schema("a_meta", {"model": {"type": "enum", "choices": ["1"]}})
    with pytest.raises(ValueError, match="both produce 'ie_a_meta__model'"):
        check_ie_collisions([a, a_meta])


def test_same_sanitized_stem_collides() -> None:
    one = parse_ie_schema("my schema", {"x": {"type": "string"}})
    two = parse_ie_schema("my-schema", {"x": {"type": "string"}})
    assert one.name == two.name == "my_schema"
    with pytest.raises(ValueError, match="given twice"):
        check_ie_collisions([one, two])


def test_collision_with_schema_already_on_collection() -> None:
    a_meta = parse_ie_schema("a_meta", {"model": {"type": "enum", "choices": ["1"]}})
    with pytest.raises(ValueError, match="collides with existing IE schema 'a'"):
        check_ie_collisions([a_meta], existing_names=["a"])
    check_ie_collisions([a_meta], existing_names=["a_meta", "unrelated"])  # re-adding is fine


def test_real_schemas_do_not_collide() -> None:
    schemas = [load_ie_schema(p) for p in IE_SCHEMA_FILES]
    check_ie_collisions(schemas)


# --- add_missing_properties with a fake collection ---------------------------------------------


class _FakeConfig:
    def __init__(self, props: list) -> None:
        self.props = props
        self.added: list[wvcc.Property] = []

    def get(self):
        return SimpleNamespace(properties=self.props)

    def add_property(self, prop: wvcc.Property) -> None:
        self.added.append(prop)


class _FakeCollection:
    name = "Fake"

    def __init__(self, props: list) -> None:
        self.config = _FakeConfig(props)


def _existing(prop: wvcc.Property, **overrides):
    """What `collection.config.get().properties` returns for a created property."""
    nested = [
        SimpleNamespace(name=n.name, data_type=n.dataType) for n in prop.nestedProperties or ()
    ]
    values = {
        "name": prop.name,
        "data_type": prop.dataType,
        "index_filterable": prop.indexFilterable,
        "index_searchable": prop.indexSearchable,
        "index_range_filters": prop.indexRangeFilters,
        "tokenization": prop.tokenization,
        "nested_properties": nested or None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.fixture
def warnings_log():
    from loguru import logger

    messages: list[str] = []
    sink = logger.add(lambda m: messages.append(m.record["message"]), level="WARNING")
    yield messages
    logger.remove(sink)


def test_type_conflict_raises() -> None:
    wanted = wvcc.Property(name="p", data_type=wvcc.DataType.INT, index_filterable=True)
    coll = _FakeCollection([_existing(wanted, data_type=wvcc.DataType.TEXT)])
    with pytest.raises(ValueError, match=r"Fake\.p exists as text, schema wants int"):
        add_missing_properties(coll, [wanted])
    assert coll.config.added == []


def test_add_missing_adds_only_new_and_is_idempotent(warnings_log) -> None:
    old = wvcc.Property(name="old", data_type=wvcc.DataType.INT, index_filterable=True)
    new = wvcc.Property(name="new", data_type=wvcc.DataType.TEXT, index_filterable=True)
    coll = _FakeCollection([_existing(old)])
    assert add_missing_properties(coll, [old, new]) == ["new"]
    assert [p.name for p in coll.config.added] == ["new"]
    assert warnings_log == []


def test_flag_and_tokenization_drift_warns(warnings_log) -> None:
    wanted = wvcc.Property(
        name="p",
        data_type=wvcc.DataType.TEXT,
        index_filterable=True,
        index_searchable=False,
        tokenization=wvcc.Tokenization.FIELD,
    )
    have = _existing(wanted, index_filterable=False, tokenization=wvcc.Tokenization.WORD)
    coll = _FakeCollection([have])
    assert add_missing_properties(coll, [wanted]) == []
    assert len(warnings_log) == 2
    assert any("index_filterable" in m for m in warnings_log)
    assert any("tokenization is word, wanted field" in m for m in warnings_log)


def test_nested_drift_names_keys(warnings_log) -> None:
    def nested(name, dt):
        return wvcc.Property(name=name, data_type=dt)

    wanted = wvcc.Property(
        name="ie_t",
        data_type=wvcc.DataType.OBJECT,
        nested_properties=[
            nested("kept", wvcc.DataType.TEXT),
            nested("retyped", wvcc.DataType.INT),
            nested("brand_new", wvcc.DataType.TEXT),
        ],
    )
    have = _existing(wanted)
    have.nested_properties = [
        SimpleNamespace(name="kept", data_type=wvcc.DataType.TEXT),
        SimpleNamespace(name="retyped", data_type=wvcc.DataType.TEXT),
    ]
    assert property_drift(have, wanted) == [
        "nested keys missing and impossible to add: ['brand_new']",
        "nested keys with a different type: ['retyped']",
    ]
    add_missing_properties(_FakeCollection([have]), [wanted])
    assert any("brand_new" in m for m in warnings_log)
    assert any("retyped" in m for m in warnings_log)


def test_add_ie_schema_with_fakes_and_collision_with_existing() -> None:
    path = IE_DIR / "personal_rights.yaml"
    judgments, chunks = _FakeCollection([]), _FakeCollection([])
    added = add_ie_schema(judgments, path, chunk_collection=chunks)
    assert "ie_personal_rights" in added
    assert {p.name for p in chunks.config.added} <= set(added)

    # IE objects already on a collection are recognised by their `ie_` object property
    existing_obj = SimpleNamespace(name="ie_personal", data_type=wvcc.DataType.OBJECT)
    assert existing_ie_schema_names(_FakeCollection([existing_obj])) == ["personal"]
