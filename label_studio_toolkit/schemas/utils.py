from enum import Enum
from types import UnionType
from typing import get_args, get_origin

from pydantic import model_validator


class SchemaUtilsMixin:
    @model_validator(mode="before")
    @classmethod
    def coerce_single_enum_to_list(cls, values):
        for field_name, field_type in cls.model_fields.items():
            annotation = field_type.annotation

            # Check for Union[list[Enum], None] or just list[Enum]
            if get_origin(annotation) is UnionType:
                args = get_args(annotation)
                list_type = next((arg for arg in args if get_origin(arg) is list), None)
            elif get_origin(annotation) is list:
                list_type = annotation
            else:
                continue

            if list_type is None:
                continue

            enum_type = get_args(list_type)[0]
            if not isinstance(enum_type, type) or not issubclass(enum_type, Enum):
                continue

            value = values.get(field_name)
            if isinstance(value, str):
                values[field_name] = [value]

        return values

    @classmethod
    def get_schema_string(cls) -> str:
        schema_parts = []

        for field_name, field in cls.model_fields.items():
            field_type = field.annotation
            description = field.description or ""

            # Check if field is optional (Union with None)
            is_optional = get_origin(field_type) is UnionType and type(None) in get_args(field_type)

            # Handle Union types (e.g., Union[Enum, None])
            if get_origin(field_type) is UnionType:
                field_type = next(
                    (t for t in get_args(field_type) if t is not type(None)), field_type
                )

            # Handle list types
            if get_origin(field_type) is list:
                item_type = get_args(field_type)[0]
                schema_parts.append(f"{field_name}:")
                schema_parts.append("  type: list")
                schema_parts.append("  items:")

                if isinstance(item_type, type) and issubclass(item_type, Enum):
                    schema_parts.append("    type: enum")
                    schema_parts.append(f"    choices: {[e.value for e in item_type]}")
                elif item_type is str:
                    schema_parts.append("    type: string")
                elif item_type is int:
                    schema_parts.append("    type: integer")
                else:
                    raise ValueError(f"Unknown list item type: {item_type}")

                schema_parts.append(f'  description: "{description}"')
                if is_optional:
                    schema_parts.append("  required: false")
            # Handle Enum types
            elif isinstance(field_type, type) and issubclass(field_type, Enum):
                schema_parts.append(f"{field_name}:")
                schema_parts.append("  type: enum")
                schema_parts.append(f"  choices: {[e.value for e in field_type]}")
                schema_parts.append(f'  description: "{description}"')
                if is_optional:
                    schema_parts.append("  required: false")
            # Handle string types
            elif field_type is str:
                schema_parts.append(f"{field_name}:")
                schema_parts.append("  type: string")
                schema_parts.append(f'  description: "{description}"')
                if is_optional:
                    schema_parts.append("  required: false")
            else:
                raise ValueError(f"Unknown field type: {field_type}")

            schema_parts.append("")  # empty line between fields

        return "\n".join(schema_parts)
