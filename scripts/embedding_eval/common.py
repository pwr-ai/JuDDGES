"""Shared config and paths for the embedding eval scripts."""

import os
from dataclasses import fields
from pathlib import Path
from typing import Any

DEFAULT_OUT_DIR = "data/experiments/embedding_eval"
HF_COURT_RAW_GLOB = (
    "/mnt/ai/cache/huggingface/hub/datasets--JuDDGES--pl-court-raw/snapshots/*/data/*.parquet"
)


def env_override(cfg: Any, prefix: str) -> Any:
    """Override dataclass fields in place from `<PREFIX><FIELD_NAME>` env vars (types inferred from defaults)."""
    for f in fields(cfg):
        raw = os.environ.get(f"{prefix}{f.name.upper()}")
        if raw is None:
            continue
        current = getattr(cfg, f.name)
        if isinstance(current, bool):
            value: Any = raw.lower() in {"1", "true", "yes"}
        elif isinstance(current, int):
            value = int(raw)
        elif isinstance(current, float):
            value = float(raw)
        elif isinstance(current, tuple):
            value = tuple(x.strip() for x in raw.split(",") if x.strip())
        else:
            value = raw
        setattr(cfg, f.name, value)
    return cfg


def out_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
