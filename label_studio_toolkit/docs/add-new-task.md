# Add a new annotation task

This is the toolkit's primary use case. The included Swiss Franc, personal rights, and English appeal court tasks are concrete examples of the same recipe.

A new task is four files:

| Step | File | What it defines |
|---|---|---|
| 1 | `label_studio_toolkit/schemas/<task>.py` | Pydantic schema for the structured output. |
| 2 | `label_studio_toolkit/form_templates/<task>.xml` | Label Studio XML form rendering the schema. |
| 3 | `configs/prompt/annotate_<task>.yaml` | Prompt template for the LLM. |
| 4 | `configs/annotate_data.yaml` *(or new variant)* | Hydra config wiring it all together. |

After that, run the same scripts you'd run for any task — see [preannotation.md](preannotation.md) and [upload-and-annotate.md](upload-and-annotate.md).

## 1. Define the Pydantic schema

Subclass `pydantic.BaseModel` and `SchemaUtilsMixin`. Make every field that can be empty `Optional`. Use `Enum` for closed vocabularies (single values or lists), `str` for free text, `int` for numeric scales.

```python
# label_studio_toolkit/schemas/contract_breach.py
from enum import Enum

from pydantic import BaseModel, Field

from label_studio_toolkit.schemas.utils import SchemaUtilsMixin


class TakNie(str, Enum):
    TAK = "Tak"
    NIE = "Nie"


class TypNaruszenia(str, Enum):
    NIETERMINOWE = "Nieterminowe wykonanie"
    NIENALEZYTE = "Nienależyte wykonanie"
    BRAK_WYKONANIA = "Brak wykonania"


class ContractBreachAnnotation(BaseModel, SchemaUtilsMixin):
    naruszenie_umowy: TakNie = Field(
        ..., description="Czy doszło do naruszenia umowy?"
    )
    typ_naruszenia: list[TypNaruszenia] | None = Field(
        None, description="Rodzaj naruszenia"
    )
    wartosc_szkody: int | None = Field(
        None, description="Wartość szkody w PLN", ge=0
    )
    opis: str | None = Field(None, description="Opis sytuacji")
```

### Type rules supported by the toolkit

`SchemaUtilsMixin.get_schema_string()` ([`label_studio_toolkit/schemas/utils.py`](../schemas/utils.py)) supports:

- `Enum` (single)
- `list[Enum]`
- `list[str]`
- `str`
- `int`
- `Optional[...]` (i.e. `T | None`)

Anything else raises `ValueError("Unknown field type: ...")` from `get_schema_string()`. Stick to the types above unless you also extend `utils.py`.

### Why `SchemaUtilsMixin`

Two utilities come with it:

- `coerce_single_enum_to_list()` — a `model_validator(mode="before")` that wraps a single string in a list when the field annotation is `list[Enum]`. This handles the case where the LLM returns one value as a string instead of a one-element list.
- `get_schema_string()` — renders the schema as YAML, used by the export scripts to write `schema.yaml` next to `dataset.json`.

## 2. Write the XML form template

The form maps the schema onto Label Studio's UI. **The `name` attribute on each `Choices` / `TextArea` / `Number` element must equal a field name on the schema.** This contract is enforced at upload time by `LabelStudioClient.create_prediction` ([`label_studio_toolkit/api/client.py:58`](../api/client.py#L58)).

Minimum useful template, mirroring the schema above:

```xml
<View>
  <Text name="text" value="$text" />

  <Header value="Naruszenie umowy" />
  <Choices name="naruszenie_umowy" toName="text" choice="single" required="true">
    <Choice value="Tak" />
    <Choice value="Nie" />
  </Choices>

  <Header value="Typ naruszenia" />
  <Choices name="typ_naruszenia" toName="text" choice="multiple">
    <Choice value="Nieterminowe wykonanie" />
    <Choice value="Nienależyte wykonanie" />
    <Choice value="Brak wykonania" />
  </Choices>

  <Header value="Wartość szkody (PLN)" />
  <Number name="wartosc_szkody" toName="text" min="0" />

  <Header value="Opis" />
  <TextArea name="opis" toName="text" rows="3" placeholder="Opis sytuacji..." />
</View>
```

For a richer layout (collapsible sections, two-column layout with the document on the left), copy [`label_studio_toolkit/form_templates/personal_rights.xml`](../form_templates/personal_rights.xml) and rename the controls.

### Mapping rules

| Pydantic type | XML element |
|---|---|
| `Enum` (single) | `<Choices name="..." choice="single">` with `<Choice value="..." />` per enum value |
| `list[Enum]` | `<Choices name="..." choice="multiple">` |
| `str` | `<TextArea name="..." rows="...">` |
| `list[str]` | `<TextArea name="..." rows="1" maxSubmissions="20">` (annotators submit each entry separately) |
| `int` | `<Number name="..." min="..." max="..." step="..." />` |

`<Choice value="..." />` strings must match the `Enum` `.value` exactly, otherwise Pydantic validation fails on export.

## 3. Write the prompt config

```yaml
# configs/prompt/annotate_contract_breach.yaml
template: |
    Jesteś analitykiem prawnym. Wyodrębnij ustrukturyzowane informacje
    z poniższego orzeczenia zgodnie z podanym schematem.

    Zasady:
    - Ekstrahuj tylko to, co jest jednoznacznie wskazane w tekście.
    - Użyj null, gdy nie znajdziesz informacji.
    - Wartości muszą odpowiadać typom i enumom ze schematu.

    Tekst orzeczenia:
    ====
    {text}
    ====
```

Requirements:

- Must contain `{text}`. This is asserted in [`LangChainAnnotator.__init__`](../annotator.py).
- May contain `{language}` if you want to parameterise the language via the `language` config field (used by `annotate_data.py`'s async path).

For richer prompts, see [`configs/prompt/annotate_personal_rights.yaml`](../../configs/prompt/annotate_personal_rights.yaml) — it bundles domain context (article excerpts, scoring rubric, examples).

## 4. Wire it into a Hydra config

You can either edit `configs/annotate_data.yaml` or create a new variant. New variant is cleaner:

```yaml
# configs/annotate_contract_breach.yaml
defaults:
  - model: ../api_llm/gpt_4o_mini
  - prompt: annotate_contract_breach
  - _self_

dataset:
  type: parquet
  train:
    path: data/contract_breach/train.parquet
  test:
    path: data/contract_breach/test.parquet

text_field: "text"
language: "polish"
skip_test: false
skip_train: false
model_version: ${model.name}
annotation_schema: label_studio_toolkit.schemas.contract_breach.ContractBreachAnnotation

output_dir: data/contract_breach/annotations/${model.name}

model:
  request_cache_db: .langchain_annotation_cache.db
```

Then run with `--config-name`:

```bash
python scripts/annotation/annotate_data.py --config-name annotate_contract_breach
```

For workflow A you also need a sibling config for the upload step:

```yaml
# configs/upload_contract_breach.yaml
ls_base_url: ${oc.env:LABEL_STUDIO_BASE_URL}
ls_api_key: ${oc.env:LABEL_STUDIO_API_KEY}
project_name: "Contract breach"
text_field: "text"
label_interface: label_studio_toolkit/form_templates/contract_breach.xml

model_version: ${model.name}
data_path: data/contract_breach/annotations/gpt-4o-mini/test_annotations.parquet

filter_fields: {}
annotation_schema: label_studio_toolkit.schemas.contract_breach.ContractBreachAnnotation

model:
  name: gpt-4o-mini
  request_cache_db: .langchain_annotation_cache.db
```

Run with:

```bash
python scripts/label_studio/upload_with_preannotation.py --config-name upload_contract_breach
```

## 5. End-to-end check

```bash
# 1. Generate preannotations
python scripts/annotation/annotate_data.py --config-name annotate_contract_breach

# 2. Upload to Label Studio
python scripts/label_studio/upload_with_preannotation.py --config-name upload_contract_breach

# 3. Open Label Studio in the browser, review and submit the tasks.

# 4. From the LS UI, export as JSON. Then convert it:
python scripts/label_studio/export_annotated_dataset.py \
  --input-file ./contract-breach-export.json \
  --output-path data/contract_breach/exports \
  --schema contract_breach
```

> `--schema contract_breach` requires an entry in `SCHEMA_MAP` inside [`scripts/label_studio/export_annotated_dataset.py`](../../scripts/label_studio/export_annotated_dataset.py):
> ```python
> SCHEMA_MAP = {
>     ...
>     "contract_breach": ContractBreachAnnotation,
> }
> ```

## Examples to copy

| Task | Schema | Form | Prompt | Annotate config |
|---|---|---|---|---|
| Polish CHF mortgage | [`schemas/swiss_frank.py`](../schemas/swiss_frank.py) | [`form_templates/swiss_frank.xml`](../form_templates/swiss_frank.xml) | [`configs/prompt/annotate_swiss_frank.yaml`](../../configs/prompt/annotate_swiss_frank.yaml) | [`configs/annotate_data.yaml`](../../configs/annotate_data.yaml) |
| Polish personal rights | [`schemas/personal_rights.py`](../schemas/personal_rights.py) | [`form_templates/personal_rights.xml`](../form_templates/personal_rights.xml) | [`configs/prompt/annotate_personal_rights.yaml`](../../configs/prompt/annotate_personal_rights.yaml) | — (use `preannotate_label_studio.yaml`) |
| English appeal court | [`schemas/en_appealcourt.py`](../schemas/en_appealcourt.py) | *(no form yet — schema-only example)* | [`configs/prompt/annotate_en_appealcourt.yaml`](../../configs/prompt/annotate_en_appealcourt.yaml) | [`configs/annotate_data_en_appealcourt.yaml`](../../configs/annotate_data_en_appealcourt.yaml) |
