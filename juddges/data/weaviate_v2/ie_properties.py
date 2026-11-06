"""Weaviate properties generated from the information-extraction schemas in configs/ie_schema.

For every schema file `<name>.yaml` a judgment (and optionally chunk) collection gets:

- `ie_<name>`: one `object` property holding the full extraction result. It is stored only,
  because Weaviate 1.40 cannot filter by nested path (`invalid where filter`).
- `ie_<name>__<key>`: flat top-level copies of the keys worth filtering on (see `FlatCopyPolicy`).
- `ie_<name>_meta__schema_version|model|extracted_at`: provenance of the extraction.

All functions here are pure (no network) except `add_ie_schema`, which talks to a collection.
"""

import math
import re
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import yaml
from loguru import logger

import weaviate.classes.config as wvcc

IE_PREFIX = "ie_"
FLAT_SEP = "__"
META_INFIX = "_meta"
MAX_PROPERTY_NAME_LENGTH = 231  # Weaviate limit

# Characters NFKD does not decompose into base letter + combining mark
_TRANSLITERATION = str.maketrans({"ł": "l", "Ł": "L", "đ": "d", "Đ": "D", "ø": "o", "Ø": "O"})
_DATE_HINT = re.compile(r"YYYY-MM-DD|ISO\s*8601", re.IGNORECASE)


class IeKind(str, Enum):
    """Value kind of an IE key, independent of the YAML spelling."""

    ENUM = "enum"
    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    DATE = "date"
    BOOLEAN = "boolean"


_YAML_TYPES: dict[str, IeKind] = {
    "enum": IeKind.ENUM,
    "string": IeKind.STRING,
    "str": IeKind.STRING,
    "text": IeKind.STRING,
    "int": IeKind.INTEGER,
    "integer": IeKind.INTEGER,
    "number": IeKind.NUMBER,
    "float": IeKind.NUMBER,
    "date": IeKind.DATE,
    "datetime": IeKind.DATE,
    "bool": IeKind.BOOLEAN,
    "boolean": IeKind.BOOLEAN,
}

_SCALAR_TYPE: dict[IeKind, wvcc.DataType] = {
    IeKind.ENUM: wvcc.DataType.TEXT,
    IeKind.STRING: wvcc.DataType.TEXT,
    IeKind.INTEGER: wvcc.DataType.INT,
    IeKind.NUMBER: wvcc.DataType.NUMBER,
    IeKind.DATE: wvcc.DataType.DATE,
    IeKind.BOOLEAN: wvcc.DataType.BOOL,
}
_ARRAY_TYPE: dict[IeKind, wvcc.DataType] = {
    IeKind.ENUM: wvcc.DataType.TEXT_ARRAY,
    IeKind.STRING: wvcc.DataType.TEXT_ARRAY,
    IeKind.INTEGER: wvcc.DataType.INT_ARRAY,
    IeKind.NUMBER: wvcc.DataType.NUMBER_ARRAY,
    IeKind.DATE: wvcc.DataType.DATE_ARRAY,
    IeKind.BOOLEAN: wvcc.DataType.BOOL_ARRAY,
}


@dataclass(frozen=True)
class FlatCopyPolicy:
    """Which IE keys get a flat, filterable top-level copy (`ie_<name>__<key>`).

    Defaults: scalar enums, dates, integers/numbers and booleans, plus lists of enums.
    Free-text strings and lists of anything else live only in the `ie_<name>` object.

    Attributes:
        enum, date, number, boolean: copy scalar keys of that kind.
        list_enum: also copy lists of enums (as `text[]`, FIELD tokenization).
        include: key names (YAML key or sanitized name) copied regardless of the flags above.
        exclude: key names never copied. Wins over `include`.
        kind_overrides: force the kind of a key, e.g. {"data_wyroku": IeKind.DATE} when the
            description carries no "YYYY-MM-DD" hint.
        chunk_kinds: scalar kinds whose flat copies are also written to chunks (denormalized for
            pre-filtering ANN search). Default only enums: cheap, low cardinality.
    """

    enum: bool = True
    date: bool = True
    number: bool = True
    boolean: bool = True
    list_enum: bool = True
    include: frozenset[str] = frozenset()
    exclude: frozenset[str] = frozenset()
    kind_overrides: Mapping[str, IeKind] = field(default_factory=dict)
    chunk_kinds: frozenset[IeKind] = frozenset({IeKind.ENUM})


DEFAULT_POLICY = FlatCopyPolicy()


@dataclass(frozen=True)
class IeKey:
    """One key of an IE schema."""

    key: str  # name in the YAML / in extraction results
    prop: str  # sanitized Weaviate property name
    kind: IeKind
    is_list: bool
    description: str
    required: bool
    choices: tuple[str, ...]
    flat_copy: bool
    on_chunks: bool


@dataclass(frozen=True)
class IeSchema:
    """Parsed IE schema ready to be turned into Weaviate properties."""

    name: str  # sanitized schema name (YAML file stem)
    keys: tuple[IeKey, ...]

    @property
    def object_name(self) -> str:
        return f"{IE_PREFIX}{self.name}"

    def flat_name(self, key: IeKey) -> str:
        return f"{self.object_name}{FLAT_SEP}{key.prop}"

    def meta_name(self, field_name: str) -> str:
        return f"{self.object_name}{META_INFIX}{FLAT_SEP}{field_name}"


def sanitize_name(raw: str, *, lower_first: bool = True) -> str:
    """Turn an arbitrary string into a valid Weaviate property name.

    Transliterates Polish diacritics, replaces anything outside [A-Za-z0-9_] with `_`, collapses
    repeated underscores (`__` is reserved as separator), prefixes names starting with a digit.
    By default the first letter is lowercased, as Weaviate property names are lowerCamel.
    """
    ascii_name = unicodedata.normalize("NFKD", raw.translate(_TRANSLITERATION))
    ascii_name = "".join(c for c in ascii_name if not unicodedata.combining(c))
    name = re.sub(r"[^0-9A-Za-z_]+", "_", ascii_name)
    name = re.sub(r"_+", "_", name).strip("_")
    if not name:
        raise ValueError(f"cannot sanitize {raw!r}: result is empty")
    if name[0].isdigit():
        name = f"_{name}"
    if lower_first:
        name = name[0].lower() + name[1:]
    return name


def _kind_of(key: str, spec: Mapping[str, Any], policy: FlatCopyPolicy) -> tuple[IeKind, bool]:
    """Resolve (kind, is_list) of one key spec."""
    raw_type = str(spec.get("type", "")).lower()
    is_list = raw_type in ("list", "array")
    if is_list:
        items = spec.get("items")
        if not isinstance(items, Mapping) or "type" not in items:
            raise ValueError(f"key {key!r}: list needs `items.type`")
        raw_type = str(items["type"]).lower()
    if raw_type not in _YAML_TYPES:
        raise ValueError(f"key {key!r}: unsupported type {raw_type!r}")
    kind = _YAML_TYPES[raw_type]
    if kind == IeKind.STRING and not is_list and _DATE_HINT.search(spec.get("description") or ""):
        kind = IeKind.DATE
    override = policy.kind_overrides.get(key)
    return (override if override is not None else kind), is_list


def _wants_flat_copy(key: str, prop: str, kind: IeKind, is_list: bool, p: FlatCopyPolicy) -> bool:
    if key in p.exclude or prop in p.exclude:
        return False
    if key in p.include or prop in p.include:
        return True
    if is_list:
        return kind == IeKind.ENUM and p.list_enum
    return {
        IeKind.ENUM: p.enum,
        IeKind.DATE: p.date,
        IeKind.INTEGER: p.number,
        IeKind.NUMBER: p.number,
        IeKind.BOOLEAN: p.boolean,
        IeKind.STRING: False,
    }[kind]


def parse_ie_schema(
    name: str, spec: Mapping[str, Mapping[str, Any]], policy: FlatCopyPolicy = DEFAULT_POLICY
) -> IeSchema:
    """Build an `IeSchema` from the parsed YAML mapping. Fails loudly on bad types/collisions."""
    schema_name = sanitize_name(name, lower_first=False)
    keys: list[IeKey] = []
    seen: dict[str, str] = {}
    for key, key_spec in spec.items():
        prop = sanitize_name(str(key))
        if prop in seen:
            raise ValueError(
                f"schema {name!r}: name collision, keys {seen[prop]!r} and {key!r} -> {prop!r}"
            )
        seen[prop] = str(key)
        kind, is_list = _kind_of(str(key), key_spec, policy)
        flat = _wants_flat_copy(str(key), prop, kind, is_list, policy)
        keys.append(
            IeKey(
                key=str(key),
                prop=prop,
                kind=kind,
                is_list=is_list,
                description=str(key_spec.get("description") or ""),
                required=bool(key_spec.get("required", True)),
                choices=tuple(str(c) for c in key_spec.get("choices") or () if c is not None),
                flat_copy=flat,
                on_chunks=flat and not is_list and kind in policy.chunk_kinds,
            )
        )
    schema = IeSchema(name=schema_name, keys=tuple(keys))
    _check_name_lengths(schema)
    return schema


def _check_name_lengths(schema: IeSchema) -> None:
    for k in schema.keys:
        flat = schema.flat_name(k)
        if len(flat) > MAX_PROPERTY_NAME_LENGTH:
            raise ValueError(f"property name too long ({len(flat)}): {flat}")


def load_ie_schema(path: Path, policy: FlatCopyPolicy = DEFAULT_POLICY) -> IeSchema:
    """Load `configs/ie_schema/<name>.yaml` (the file stem becomes the schema name)."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        spec = yaml.safe_load(f)
    if not isinstance(spec, Mapping) or not spec:
        raise ValueError(f"{path}: expected a non-empty mapping of keys")
    return parse_ie_schema(path.stem, spec, policy)


# --- Weaviate properties -----------------------------------------------------------------------


def _nested_property(k: IeKey) -> wvcc.Property:
    # Not indexed: nested paths cannot be filtered/searched in Weaviate 1.40, so indexes are waste.
    data_type = _ARRAY_TYPE[k.kind] if k.is_list else _SCALAR_TYPE[k.kind]
    return wvcc.Property(
        name=k.prop,
        data_type=data_type,
        description=k.description or None,
        index_filterable=False,
        index_searchable=False,
    )


def _flat_property(schema: IeSchema, k: IeKey) -> wvcc.Property:
    data_type = _ARRAY_TYPE[k.kind] if k.is_list else _SCALAR_TYPE[k.kind]
    kwargs: dict[str, Any] = {
        "name": schema.flat_name(k),
        "data_type": data_type,
        "description": f"Flat copy of {schema.object_name}.{k.prop}",
        "index_filterable": True,
    }
    if k.kind == IeKind.ENUM:
        kwargs.update(tokenization=wvcc.Tokenization.FIELD, index_searchable=False)
    elif k.kind == IeKind.STRING:
        kwargs.update(tokenization=wvcc.Tokenization.WORD, index_searchable=False)
    elif not k.is_list and k.kind in (IeKind.INTEGER, IeKind.NUMBER, IeKind.DATE):
        kwargs.update(index_range_filters=True)
    return wvcc.Property(**kwargs)


def _meta_properties(schema: IeSchema) -> list[wvcc.Property]:
    return [
        wvcc.Property(
            name=schema.meta_name("schema_version"),
            data_type=wvcc.DataType.TEXT,
            description="Version of the IE schema used for the extraction",
            index_filterable=True,
            index_searchable=False,
            tokenization=wvcc.Tokenization.FIELD,
        ),
        wvcc.Property(
            name=schema.meta_name("model"),
            data_type=wvcc.DataType.TEXT,
            description="Model that produced the extraction",
            index_filterable=True,
            index_searchable=False,
            tokenization=wvcc.Tokenization.FIELD,
        ),
        wvcc.Property(
            name=schema.meta_name("extracted_at"),
            data_type=wvcc.DataType.DATE,
            description="When the extraction was produced",
            index_filterable=True,
            index_range_filters=True,
        ),
    ]


def ie_properties(schema: IeSchema) -> list[wvcc.Property]:
    """Judgment-level properties: the `ie_<name>` object, meta fields and flat copies."""
    obj = wvcc.Property(
        name=schema.object_name,
        data_type=wvcc.DataType.OBJECT,
        description=f"Full IE result for schema {schema.name} (stored, not filterable by path)",
        nested_properties=[_nested_property(k) for k in schema.keys],
    )
    flats = [_flat_property(schema, k) for k in schema.keys if k.flat_copy]
    props = [obj, *_meta_properties(schema), *flats]
    names = [p.name for p in props]
    if len(names) != len(set(names)):
        dups = sorted({n for n in names if names.count(n) > 1})
        raise ValueError(f"schema {schema.name!r}: property name collision {dups}")
    return props


def ie_chunk_properties(schema: IeSchema) -> list[wvcc.Property]:
    """Flat copies that are denormalized onto chunks (see `FlatCopyPolicy.chunk_kinds`)."""
    return [_flat_property(schema, k) for k in schema.keys if k.on_chunks]


# --- payloads ----------------------------------------------------------------------------------


def to_rfc3339(value: Any) -> Optional[str]:
    """Normalize a date/datetime/ISO string to `YYYY-MM-DDTHH:MM:SSZ` (UTC). None if unparseable."""
    try:
        if isinstance(value, datetime):
            dt = value
        elif isinstance(value, date):
            dt = datetime(value.year, value.month, value.day)
        elif isinstance(value, str):
            text = value.strip()
            dt = (
                datetime.combine(date.fromisoformat(text), datetime.min.time())
                if len(text) == 10
                else datetime.fromisoformat(text)
            )
        else:
            return None
    except ValueError:
        return None
    dt = dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _coerce_scalar(kind: IeKind, value: Any) -> Any:
    """Coerce one value to the Python type matching `kind`. None means 'drop'."""
    if value is None:
        return None
    if kind in (IeKind.ENUM, IeKind.STRING):
        if isinstance(value, (dict, list, tuple, set)):
            return None
        text = str(value).strip()
        return text or None
    if kind == IeKind.DATE:
        return to_rfc3339(value)
    if kind == IeKind.BOOLEAN:
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        return {"true": True, "false": False}.get(text)
    if isinstance(value, bool):  # bool is an int subclass; never a number here
        return None
    try:
        number = float(value.strip()) if isinstance(value, str) else float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    if kind == IeKind.INTEGER:
        return int(number) if number.is_integer() else None
    return number


def _coerce_key(k: IeKey, value: Any) -> Any:
    if k.is_list:
        if value is None:
            return None
        items = value if isinstance(value, (list, tuple, set)) else [value]
        coerced = [c for c in (_coerce_scalar(k.kind, v) for v in items) if c is not None]
        return coerced or None
    return _coerce_scalar(k.kind, value)


def build_ie_payload(
    schema: IeSchema,
    result: Mapping[str, Any],
    *,
    schema_version: Optional[str] = None,
    model: Optional[str] = None,
    extracted_at: Optional[Any] = None,
) -> dict[str, Any]:
    """Turn one IE result (YAML key -> value) into the Weaviate properties for a judgment.

    Null, empty and unparseable values are dropped, unknown keys are ignored. Enum values are kept
    as text even when they are not in `choices` (e.g. "Tak"/"Nie"); dates become RFC3339 UTC.
    Returns {} when nothing usable was extracted and no meta was given.
    """
    obj: dict[str, Any] = {}
    payload: dict[str, Any] = {}
    for k in schema.keys:
        raw = result.get(k.key, result.get(k.prop))
        value = _coerce_key(k, raw)
        if value is None:
            if raw not in (None, "", [], ()):
                logger.debug("{}: dropped unusable value for {!r}: {!r}", schema.name, k.key, raw)
            continue
        obj[k.prop] = value
        if k.flat_copy:
            payload[schema.flat_name(k)] = value
    if obj:
        payload[schema.object_name] = obj
    if schema_version is not None:
        payload[schema.meta_name("schema_version")] = str(schema_version)
    if model is not None:
        payload[schema.meta_name("model")] = str(model)
    if extracted_at is not None:
        if (normalized := to_rfc3339(extracted_at)) is not None:
            payload[schema.meta_name("extracted_at")] = normalized
    return payload


def ie_chunk_payload(schema: IeSchema, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Subset of a judgment payload that is denormalized onto each of its chunks."""
    names = {schema.flat_name(k) for k in schema.keys if k.on_chunks}
    return {name: value for name, value in payload.items() if name in names}


# --- collection updates ------------------------------------------------------------------------


def _add_missing_properties(collection: Any, props: Iterable[wvcc.Property]) -> list[str]:
    existing = {p.name: p.data_type for p in collection.config.get().properties}
    added: list[str] = []
    for prop in props:
        if prop.name in existing:
            if existing[prop.name] != prop.dataType:
                raise ValueError(
                    f"{collection.name}.{prop.name} exists as {existing[prop.name].value}, "
                    f"schema wants {prop.dataType.value}"
                )
            continue
        collection.config.add_property(prop)
        added.append(prop.name)
    return added


def add_ie_schema(
    collection: Any,
    yaml_path: Path,
    *,
    chunk_collection: Optional[Any] = None,
    policy: FlatCopyPolicy = DEFAULT_POLICY,
) -> list[str]:
    """Add the properties of one IE schema to an existing collection. Idempotent.

    Properties that already exist with the same type are skipped; with a different type it raises.
    Pass `chunk_collection` to also add the flat chunk copies. Do that before ingesting chunks:
    properties added later are indexed only for objects written (or updated) afterwards.

    Returns the names of the properties that were actually added (judgment collection first).
    """
    schema = load_ie_schema(yaml_path, policy)
    added = _add_missing_properties(collection, ie_properties(schema))
    if chunk_collection is not None:
        added += _add_missing_properties(chunk_collection, ie_chunk_properties(schema))
    logger.info("IE schema {}: added {} properties", schema.name, len(added))
    return added
