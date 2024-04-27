from pathlib import Path
from typing import Iterable, Generator
from jsonlines import jsonlines
from datetime import datetime


def save_jsonl(records: Iterable[dict], out: Path | str):
    """Save a list of dictionaries to a jsonl file."""
    with jsonlines.open(out, mode="w") as writer:
        writer.write_all(records)


def read_jsonl(path: Path | str) -> Generator[dict, None, None]:
    """Read a jsonl file and yield dictionaries."""
    with jsonlines.open(path) as reader:
        yield from reader


def path_safe_udate() -> str:
    """Generate a unique timestamp string for file naming.

    Returns:
        str: A string with the current date and time in the %Y%m%d_%H%M%Sf%f format
    """
    return datetime.now().strftime("%Y%m%d_%H%M%Sf%f")
