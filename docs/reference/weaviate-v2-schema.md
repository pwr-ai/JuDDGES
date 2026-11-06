# Weaviate v2 schema reference

Collections of the v2 stack (`weaviate/v2/`, Weaviate 1.40.2) for Polish court judgments. The code lives in
`juddges/data/weaviate_v2/`. The legacy `LegalDocuments` / `DocumentChunks` collections and
`DocumentsWeaviateDB` are unchanged and keep serving the legacy stack.

The design follows the measurements in
[Embedding model and vector quantization](../explanation/embedding-and-quantization.md). To create the
collections, see [Deploy Weaviate v2](../how-to/infrastructure/deploy-weaviate-v2.md#create-the-collections).

## Collections

| Collection | Python constant | Objects | Shards | Vectors |
|---|---|---|---|---|
| `Judgments` | `JUDGMENTS_COLLECTION` | ~3M | 4 | `thesis`, `summary` (optional per object) |
| `JudgmentChunks` | `JUDGMENT_CHUNKS_COLLECTION` | ~54M | 8 | `default` |

Shard counts are fields of `SchemaConfig` (`judgment_shards`, `chunk_shards`). The count cannot change
after creation, so set it before ingesting. 54M chunks over 8 shards is ~7M chunks per shard, which keeps
each HNSW graph and LSM store at a size that builds, compacts and restarts in reasonable time, and queries
still fan out over shards in parallel.

`JudgmentChunks.parent_judgment` is a cross-reference to `Judgments`. Object ids are deterministic:
`judgment_uuid(judgment_id)` and `chunk_uuid(judgment_id, chunk_idx)`, so a re-ingest overwrites.

## Vectors

Vectors are computed by TEI (`sdadas/mmlw-retrieval-e5-large`, 1024d) and sent with each object
(`Configure.Vectors.self_provided`). Weaviate never vectorizes.

| Setting | Value |
|---|---|
| Index | HNSW, cosine |
| `ef_construction` | 128 |
| `max_connections` | 32 |
| `ef` | -1 (dynamic) |
| Quantizer | RQ, 4 bits (`SchemaConfig.rq_bits`) |

Every named vector of both collections uses the same index config. RQ4 showed no loss vs fp32 on our
judgments, while BQ/RQ1 lost recall (ANN recall 0.87).

Client rules:

- Queries are embedded as `"query: " + text` (`QUERY_PREFIX`), chunks as `"passage: " + text`
  (`PASSAGE_PREFIX`).
- Chunk with 480 tokens and 64 overlap (BGE-M3 tokenizer), `CHUNK_TOKENS` / `CHUNK_OVERLAP_TOKENS`.
- Search chunks with hybrid, `alpha=0.25` (`HYBRID_ALPHA`, 0 is BM25 only), `query_properties=["chunk_text"]`,
  `target_vector="default"`, then aggregate to the judgment by best chunk score (MaxP).
- `thesis` and `summary` vectors are embedded as `"passage: " + text`. Objects without a thesis or summary
  simply omit that vector.

## Inverted index

Both collections use BM25 `b=0.75`, `k1=1.2`, the English stopword preset plus the Polish stopwords
(`POLISH_STOPWORDS`, imported from `juddges.data.documents_weaviate_db`) and a 60 s cleanup interval.

## `Judgments` properties

Index column: `filter` = filterable, `BM25` = searchable, `range` = range filters.

| Property | Type | Index | Tokenization | Notes |
|---|---|---|---|---|
| `judgment_id` | `text` | filter | field | Unique identifier of the judgment (UUID5 is derived from it) |
| `document_number` | `text` | filter, BM25 | word | Case signature (docket number), searchable by tokens |
| `source_id` | `text` | filter | field | Identifier in the source system |
| `source` | `text` | filter | field | Data source, e.g. pl-court, nsa |
| `source_url` | `text` | stored only | - | Source URL |
| `document_type` | `text` | filter | field | judgment, tax_interpretation, legal_act |
| `language` | `text` | filter | field | Language code (pl, en) |
| `country` | `text` | filter | field | Country of origin |
| `court_type` | `text` | filter | field | Type of court |
| `judgment_type` | `text` | filter | field | Type of judgment |
| `outcome` | `text` | filter | field | Outcome of the case |
| `court_id` | `text` | filter | field | System id of the court |
| `department_id` | `text` | filter | field | System id of the court department |
| `court_name` | `text` | filter | field | Name of the court (exact match) |
| `department_name` | `text` | filter | field | Name of the court department (exact match) |
| `date_issued` | `date` | filter, range | - | Date of the judgment |
| `publication_date` | `date` | filter, range | - | Date the judgment was published |
| `ingestion_date` | `date` | filter, range | - | When the judgment was ingested |
| `full_text` | `text` | stored only | - | Full text; BM25 and vectors live on JudgmentChunks |
| `title` | `text` | BM25 | word | Title |
| `summary` | `text` | BM25 | word | Summary (also embedded as the `summary` vector) |
| `thesis` | `text` | BM25 | word | Legal thesis (also embedded as the `thesis` vector) |
| `keywords` | `text[]` | filter, BM25 | word | Keywords / topics |
| `factual_state` | `text` | BM25 | word | Factual circumstances (LLM-extracted) |
| `legal_state` | `text` | BM25 | word | Legal state (LLM-extracted) |
| `presiding_judge` | `text` | filter | word | Presiding judge |
| `judges` | `text[]` | filter | word | Judges of the panel |
| `legal_bases` | `text[]` | filter | word | Legal bases (plain text) |
| `references` | `text[]` | filter | word | Plain-text references to legal acts |
| `extracted_legal_bases` | `object[]` | stored only | - | Legal bases linked to the ISAP repository |
| `parties` | `text` | stored only | - | Parties with roles (JSON string) |
| `legal_concepts` | `text[]` | stored only | - | Legal concepts (LLM-extracted) |
| `legal_analysis` | `text` | stored only | - | Structured legal reasoning (JSON string) |
| `judgment_specific` | `text` | stored only | - | Judgment-specific fields (JSON string) |
| `x` | `number` | stored only | - | X coordinate for visualization |
| `y` | `number` | stored only | - | Y coordinate for visualization |

`extracted_legal_bases` is an `object[]` with the stored-only text properties `address`, `art`, `isap_id`,
`text` and `title` (the structure of `extracted_legal_bases` in `juddges/schema.py`).

### Differences from `LegalDocuments`

| Change | Reason |
|---|---|
| `full_text`: not searchable, not filterable | BM25 lives on `chunk_text`. A second inverted index over all full texts is the largest avoidable cost. |
| Dropped `raw_content` | Duplicate of `full_text` (the XML it was derived from stays in the source dataset). Nothing in v2 reads it. |
| Dropped `issuing_body` | Equals `court_name` for judgments. |
| Dropped `processing_status`, `last_updated` | The ingester wrote the constant `completed` and the same timestamp as `ingestion_date`. |
| Dropped `tax_interpretation_specific` | v2 holds judgments only. |
| Dropped `metadata`, `legal_references` | Not part of the legacy collection schema, shape undefined. |
| Added `court_id`, `department_id` | System ids from the judgment schema, exact filters that do not depend on name spelling. |
| `court_name`, `department_name`: FIELD instead of WORD tokenization | An exact filter on "Sąd Okręgowy w Krakowie" must not match "Sąd Apelacyjny w Krakowie". |
| `title`, `summary`, `thesis`: BM25 only, not filterable | Nothing filters on long text. |
| `source_url`: stored only | Looked up through `source_id`. |
| `extracted_legal_bases`: `object[]` instead of a JSON string | The structure is fixed (see above). Stored only, because nested paths cannot be filtered. |
| `parties`, `legal_analysis`, `judgment_specific`: JSON string, stored only | The extractor does not fix their shape, so there is no safe `nestedProperties`. |
| `legal_concepts`: `text[]`, stored only | The extractor returns an array of strings. |
| One vector over truncated `full_text` replaced by `thesis` and `summary` vectors | Full text is covered by chunk vectors. |

## `JudgmentChunks` properties

The filter fields are copied from the judgment at ingest, so a filtered ANN or hybrid query needs no join.

| Property | Type | Index | Tokenization | Notes |
|---|---|---|---|---|
| `chunk_text` | `text` | BM25 | word | Chunk text (BM25; the vector is computed from 'passage: ' + text) |
| `judgment_id` | `text` | filter | field | Parent judgment id |
| `chunk_idx` | `int` | filter | - | Position of the chunk in the judgment |
| `document_type` | `text` | filter | field | Copy of Judgments.document_type |
| `language` | `text` | filter | field | Copy of Judgments.language |
| `court_type` | `text` | filter | field | Copy of Judgments.court_type |
| `judgment_type` | `text` | filter | field | Copy of Judgments.judgment_type |
| `court_name` | `text` | filter | field | Copy of Judgments.court_name |
| `date_issued` | `date` | filter, range | - | Copy of Judgments.date_issued |

Additionally: reference `parent_judgment` -> `Judgments`, and the flat IE copies listed below when an IE schema
is enabled for chunks.

## Information-extraction (IE) properties

Generated from `configs/ie_schema/<name>.yaml` by `juddges/data/weaviate_v2/ie_properties.py`
(`load_ie_schema`, `ie_properties`, `ie_chunk_properties`, `build_ie_payload`, `ie_chunk_payload`,
`add_ie_schema`). Weaviate 1.40 stores `object` properties but fails to filter by a nested path
(`invalid where filter`), so each schema produces a stored object plus flat, filterable copies.

| Property | Type | Notes |
|---|---|---|
| `ie_<name>` | `object` | All keys as `nestedProperties`. Stored only, nested properties have no index. |
| `ie_<name>_meta__schema_version` | `text` | FIELD, filterable |
| `ie_<name>_meta__model` | `text` | FIELD, filterable |
| `ie_<name>_meta__extracted_at` | `date` | filterable, range |
| `ie_<name>__<key>` | per key | Flat copy, only for the keys selected by the rule below |

`<name>` is the YAML file stem. `<key>` is the sanitized key.

### Type mapping

| YAML | Nested property | Flat copy |
|---|---|---|
| `enum` | `text` | `text`, FIELD, filterable |
| `string` | `text` | none (free text) |
| `string` whose description contains `YYYY-MM-DD` or `ISO 8601`, or `type: date` | `date` | `date`, filterable, range |
| `int`, `integer` | `int` | `int`, filterable, range |
| `number` | `number` | `number`, filterable, range |
| `bool`, `boolean` | `boolean` | `boolean`, filterable |
| `list` of X | `X[]` | `text[]` FIELD for lists of enums only |

An enum named like a date (`data_rozpoczecia_odsetek`) stays an enum: only strings are date-detected.

### Which keys get a flat copy

The rule is `FlatCopyPolicy` (defaults: scalar enum, date, number, boolean and list of enums).
Override it per call:

- `include` / `exclude`: key names (YAML key or sanitized name). `exclude` wins.
- `kind_overrides`: force a kind, for example `{"data_wyroku": IeKind.DATE}`.
- `list_enum=False`, `enum=False`, ... switch whole categories off.
- `chunk_kinds`: scalar kinds whose flat copies are also written to chunks. Default `{IeKind.ENUM}`. Lists are
  never copied to chunks.

Add the IE schema to the chunk collection before ingesting chunks: a property added later is indexed only for
objects written afterwards.

### Names and values

- Names: Polish diacritics are transliterated, other invalid characters become `_`, repeated underscores are
  collapsed (`__` is the separator), a leading digit gets a `_` prefix and the first letter is lowercased
  (`ConvictPleaDate` -> `convictPleaDate`). Two keys that map to the same name raise `ValueError`.
  Unsupported YAML types raise as well.
- `build_ie_payload(schema, result, schema_version=, model=, extracted_at=)` returns the properties of one
  judgment. Null, empty and unparseable values are dropped, dates become `YYYY-MM-DDTHH:MM:SSZ` (UTC),
  numbers are coerced (`"4"` -> `4`, `True` is rejected), a scalar given for a list is wrapped.
  Enum values stay text even when not listed in `choices` (`"Tak"`, `"Nie"`).
