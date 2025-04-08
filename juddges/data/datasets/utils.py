from typing import Any


def format_to_conversations(
    entry: dict[str, Any],
    prompt_field: str,
    context_field: str,
    output_field: str,
) -> dict[str, list[dict[str, str]]]:
    first_message = entry[prompt_field].format(context=entry[context_field])
    messages = [
        {"role": "user", "content": first_message},
        {"role": "assistant", "content": entry[output_field]},
    ]
    return {"messages": messages}
