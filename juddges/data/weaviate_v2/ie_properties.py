"""Weaviate properties generated from the information-extraction schemas in configs/ie_schema.

For every schema file `<name>.yaml` a judgment (and optionally chunk) collection gets:

- `ie_<name>`: one `object` property holding the full extraction result. It is stored only,
  because nested-path filtering is a preview feature, off by default in Weaviate 1.40.
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
from decimal import Decimal, InvalidOperation
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
        chunk_keys: per-schema allowlist {schema name: keys}. A schema listed here ignores
            `chunk_kinds`: exactly the listed keys (YAML key or sanitized name) are copied to
            chunks, an empty set means none. Every listed key must be a scalar with a flat copy,
            otherwise parsing raises. Use it to cap chunk-collection size, where every copy costs
            an inverted index over ~54M objects.
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
    chunk_keys: Mapping[str, frozenset[str]] = field(default_factory=dict)


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
    allow = policy.chunk_keys.get(schema_name)
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
                on_chunks=flat
                and not is_list
                and (
                    kind in policy.chunk_kinds
                    if allow is None
                    else (str(key) in allow or prop in allow)
                ),
            )
        )
    schema = IeSchema(name=schema_name, keys=tuple(keys))
    _check_name_lengths(schema)
    if allow is not None:
        known = {k.key for k in keys} | {k.prop for k in keys}
        if unknown := sorted(allow - known):
            raise ValueError(f"schema {name!r}: chunk_keys has unknown keys {unknown}")
        if bad := sorted(
            k.key for k in keys if (k.key in allow or k.prop in allow) and not k.on_chunks
        ):
            raise ValueError(
                f"schema {name!r}: chunk_keys {bad} are not scalar keys with a flat copy"
            )
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
    # Not indexed: nested paths are not filterable by default in Weaviate 1.40, so indexes are waste.
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
        dt = dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except (ValueError, OverflowError):
        return None


INT64_MIN, INT64_MAX = -(2**63), 2**63 - 1
_INT_TEXT = re.compile(r"[+-]?[0-9]+")


def _to_int64(value: Any) -> Optional[int]:
    """Exact integer value of an int, integral float or integer-like string; None if not int64."""
    if isinstance(value, int):
        number = value
    elif isinstance(value, float):
        if not value.is_integer():  # also rejects nan/inf
            return None
        number = int(value)
    elif isinstance(value, str):
        text = value.strip()
        if _INT_TEXT.fullmatch(text):
            number = int(text)
        elif "_" in text:  # Decimal() would accept "1_000"
            return None
        else:  # "4.0", "1e3": exact decimal arithmetic, never via float
            try:
                dec = Decimal(text)
            except InvalidOperation:
                return None
            if not dec.is_finite() or dec.adjusted() > 19 or dec != dec.to_integral_value():
                return None
            number = int(dec)
    else:
        return None
    if not INT64_MIN <= number <= INT64_MAX:
        logger.debug("integer {} is outside int64, dropped", number)
        return None
    return number


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
    if kind == IeKind.INTEGER:
        return _to_int64(value)
    try:
        number = float(value.strip()) if isinstance(value, str) else float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
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

_FLAGS = (
    ("index_filterable", "indexFilterable"),
    ("index_searchable", "indexSearchable"),
    ("index_range_filters", "indexRangeFilters"),
)


def property_drift(existing: Any, wanted: wvcc.Property) -> list[str]:
    """Differences between an existing property (collection config) and the wanted one.

    Index flags and tokenization are immutable and nested properties of an `object` cannot be
    extended in Weaviate, so none of these can be fixed in place. Returns human-readable messages.
    """
    messages: list[str] = []
    for existing_attr, wanted_attr in _FLAGS:
        want = getattr(wanted, wanted_attr)
        if want is not None and bool(getattr(existing, existing_attr, None)) != bool(want):
            messages.append(
                f"{existing_attr} is {getattr(existing, existing_attr, None)}, wanted {want}"
            )
    want_tok, have_tok = wanted.tokenization, getattr(existing, "tokenization", None)
    if want_tok is not None and have_tok is not None and want_tok != have_tok:
        messages.append(f"tokenization is {have_tok.value}, wanted {want_tok.value}")
    if wanted.nestedProperties:
        have = {n.name: n.data_type for n in getattr(existing, "nested_properties", None) or ()}
        missing = sorted(n.name for n in wanted.nestedProperties if n.name not in have)
        changed = sorted(
            n.name for n in wanted.nestedProperties if n.name in have and have[n.name] != n.dataType
        )
        if missing:
            messages.append(f"nested keys missing and impossible to add: {missing}")
        if changed:
            messages.append(f"nested keys with a different type: {changed}")
    return messages


def add_missing_properties(collection: Any, props: Iterable[wvcc.Property]) -> list[str]:
    """Add the properties that are not on the collection yet. Idempotent.

    A property that exists with another type raises `ValueError`. One that exists with the same
    type but a different index config (or nested keys) is skipped with a warning, as Weaviate
    cannot change that in place. Returns the names of the added properties.
    """
    existing = {p.name: p for p in collection.config.get().properties}
    added: list[str] = []
    for prop in props:
        have = existing.get(prop.name)
        if have is None:
            collection.config.add_property(prop)
            added.append(prop.name)
            continue
        if have.data_type != prop.dataType:
            raise ValueError(
                f"{collection.name}.{prop.name} exists as {have.data_type.value}, "
                f"schema wants {prop.dataType.value}"
            )
        for message in property_drift(have, prop):
            logger.warning("{}.{} differs from the schema: {}", collection.name, prop.name, message)
    return added


def existing_ie_schema_names(collection: Any) -> list[str]:
    """Names of the IE schemas that already have their `ie_<name>` object on the collection."""
    return [
        p.name[len(IE_PREFIX) :]
        for p in collection.config.get().properties
        if p.name.startswith(IE_PREFIX) and p.data_type == wvcc.DataType.OBJECT
    ]


def check_ie_collisions(schemas: Iterable[IeSchema], existing_names: Iterable[str] = ()) -> None:
    """Raise `ValueError` if property names of different IE schemas can clash.

    Covers two stems that sanitize to the same name, any generated name produced twice, and
    schemas already on a collection (`existing_names`, see `existing_ie_schema_names`) whose
    reserved prefixes (`ie_<name>__`, `ie_<name>_meta__`) a new schema would write into, e.g.
    schema `a` meta `ie_a_meta__model` vs schema `a_meta` key `model`.
    """
    schemas = list(schemas)
    seen_names: set[str] = set()
    for s in schemas:
        if s.name in seen_names:
            raise ValueError(f"IE schema name {s.name!r} given twice (or two stems sanitize to it)")
        seen_names.add(s.name)
    owners: dict[str, str] = {}
    for s in schemas:
        for prop in ie_properties(s) + ie_chunk_properties(s):
            if (owner := owners.setdefault(prop.name, s.name)) != s.name:
                raise ValueError(f"IE schemas {owner!r} and {s.name!r} both produce {prop.name!r}")
    for s in schemas:
        for other in existing_names:
            if other == s.name:
                continue
            prefixes = (
                f"{IE_PREFIX}{other}{FLAT_SEP}",
                f"{IE_PREFIX}{other}{META_INFIX}{FLAT_SEP}",
            )
            for prop in ie_properties(s) + ie_chunk_properties(s):
                if prop.name.startswith(prefixes) or prop.name == f"{IE_PREFIX}{other}":
                    raise ValueError(
                        f"IE schema {s.name!r} property {prop.name!r} collides with "
                        f"existing IE schema {other!r}"
                    )


def add_ie_schema(
    collection: Any,
    yaml_path: Path,
    *,
    chunk_collection: Optional[Any] = None,
    policy: FlatCopyPolicy = DEFAULT_POLICY,
) -> list[str]:
    """Add the properties of one IE schema to an existing collection. Idempotent.

    Properties that already exist with the same type are skipped (warning when their index
    config or nested keys differ from the YAML, which Weaviate cannot change in place); a
    different type raises. Raises `ValueError` if the schema's names collide with another IE
    schema already on the collection.
    Pass `chunk_collection` to also add the flat chunk copies. Do that before ingesting chunks:
    properties added later are indexed only for objects written (or updated) afterwards.

    Returns the names of the properties that were actually added (judgment collection first).
    """
    schema = load_ie_schema(yaml_path, policy)
    check_ie_collisions([schema], existing_ie_schema_names(collection))
    added = add_missing_properties(collection, ie_properties(schema))
    if chunk_collection is not None:
        added += add_missing_properties(chunk_collection, ie_chunk_properties(schema))
    logger.info("IE schema {}: added {} properties", schema.name, len(added))
    return added
