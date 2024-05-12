from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, Iterable

from jsonlines import jsonlines


def save_jsonl(records: Iterable[Dict[str, Any]], out: Path | str, mode="w") -> None:
    """Save a list of dictionaries to a jsonl file."""
    with jsonlines.open(out, mode=mode) as writer:
        writer.write_all(records)


def read_jsonl(path: Path | str) -> Generator[Dict[str, Any], None, None]:
    """Read a jsonl file and yield dictionaries."""
    with jsonlines.open(path) as reader:
        yield from reader


def path_safe_udate() -> str:
    """Generate a unique timestamp string for file naming.

    Returns:
        str: A string with the current date and time in the %Y%m%d_%H%M%Sf%f format
    """
    return datetime.now().strftime("%Y%m%d_%H%M%Sf%f")
