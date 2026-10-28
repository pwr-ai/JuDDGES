"""Layout resolver for the iterative-scaling eval (nested, delta-only datasets).

Iteration 0 is the original set and keeps its original flat paths (byte-identical); iteration
k >= 1 lives in `iter_k/` subdirectories holding only the delta over iteration k-1.

    data:        <base>/{queries,judgments,chunks_*}.parquet          (k = 0)
                 <base>/iter_k/{queries,judgments,chunks_*}.parquet   (k >= 1)
    embeddings:  <base>/embeddings/<key>/                             (k = 0)
                 <base>/embeddings/<key>/iter_k/                      (k >= 1)
    results:     <base>/results/iter_k/                               (all k)
"""

import os
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
