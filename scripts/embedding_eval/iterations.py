"""Layout resolver for the iterative-scaling eval (nested, delta-only datasets).

Iteration 0 is the original set and keeps its original flat paths (byte-identical); iteration
k >= 1 lives in `iter_k/` subdirectories holding only the delta over iteration k-1.

    data:        <base>/{queries,judgments,chunks_*}.parquet          (k = 0)
                 <base>/iter_k/{queries,judgments,chunks_*}.parquet   (k >= 1)
    embeddings:  <base>/embeddings/<key>/                             (k = 0)
                 <base>/embeddings/<key>/iter_k/                      (k >= 1)
    results:     <base>/results/iter_k/                               (all k)
"""

import json
import os
import re
import subprocess
from pathlib import Path


def data_dir(base: str | Path, k: int) -> Path:
    base = Path(base)
    return base if k == 0 else base / f"iter_{k}"


def emb_dir(base: str | Path, key: str, k: int) -> Path:
    d = Path(base) / "embeddings" / key
    return d if k == 0 else d / f"iter_{k}"


def results_dir(base: str | Path, k: int) -> Path:
    return Path(base) / "results" / f"iter_{k}"


def git_sha() -> str:
    """Git sha of the code; `EVAL_GIT_SHA` wins (set it when running inside a container)."""
    if os.environ.get("EVAL_GIT_SHA"):
        return os.environ["EVAL_GIT_SHA"]
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=10
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def is_placeholder(meta: dict) -> bool:
    """A `meta.json` with `status` starting with `in_progress` marks a run that has not finished."""
    return str(meta.get("status", "")).startswith("in_progress")


def read_done_meta(path: Path) -> dict | None:
    """Parsed `meta.json`, or None when it is missing, unreadable or an in-progress placeholder."""
    if not path.exists():
        return None
    try:
        meta = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    return None if is_placeholder(meta) else meta


def parse_units(units: list[str] | tuple[str, ...]) -> dict[str, list[str]]:
    """`["bgem3_c512", "mmlw_e5_c512", "bgem3_c2048"]` -> `{"bgem3": ["c512", "c2048"], "mmlw_e5": ["c512"]}`."""
    out: dict[str, list[str]] = {}
    for u in units:
        m = re.match(r"^(.+)_(c\d+)$", u)
        if not m:
            raise ValueError(f"bad embedding unit {u!r}, expected <model>_c<size>")
        chunkings = out.setdefault(m.group(1), [])
        if m.group(2) not in chunkings:
            chunkings.append(m.group(2))
    return out


def embed_needed(base: str | Path, k: int) -> dict[str, list[str]]:
    """Model -> chunkings to embed for delta k, from `results/iter_{k-1}/decisions.json` `embed_needed`."""
    path = results_dir(base, k - 1) / "decisions.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: iteration {k - 1} has not been evaluated")
    return parse_units(json.loads(path.read_text())["embed_needed"])
