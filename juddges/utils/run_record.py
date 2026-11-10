import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

RUN_RECORD_FILE = "run_record.json"
TRACKED_LIBRARIES = [
    "torch",
    "vllm",
    "transformers",
    "peft",
    "trl",
    "bitsandbytes",
    "accelerate",
    "xgrammar",
]
REPO_ROOT = Path(__file__).resolve().parents[2]


def file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def library_versions(libraries: list[str] = TRACKED_LIBRARIES) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for lib in libraries:
        try:
            versions[lib] = metadata.version(lib)
        except metadata.PackageNotFoundError:
            versions[lib] = None
    return versions


def git_state() -> dict[str, Any]:
    def _git(*args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", "-C", str(REPO_ROOT), *args],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = _git("status", "--porcelain")
    return {"sha": _git("rev-parse", "HEAD"), "dirty": None if status is None else bool(status)}


def model_revision(model_name: str) -> str | None:
    """Commit hash of the model snapshot in the local HF cache (no network access)."""
    try:
        from huggingface_hub import try_to_load_from_cache

        cached = try_to_load_from_cache(model_name, "config.json")
    except Exception:
        return None
    if not isinstance(cached, str):
        return None
    return Path(cached).parent.name


def gpu_name() -> str | None:
    try:
        import torch

        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        return None


def write_run_record(
    output_dir: Path,
    config: dict[str, Any],
    seed: int,
    model_name: str,
    split_files: dict[str, Path],
    start_time: float,
    adapter_path: Path | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Writes `run_record.json` with everything needed to reproduce and compare a run.

    Unknown values are written as null. `start_time` is a `time.time()` taken at script start.
    `adapter_path` is the resolved adapter directory of a fine-tuned run; `extra` is merged in as-is
    (e.g. the guided decoding backend).
    """
    record = {
        "config": config,
        "seed": seed,
        "git": git_state(),
        "model": {"name": model_name, "revision": model_revision(model_name)},
        "split_files_sha256": {name: file_sha256(Path(p)) for name, p in split_files.items()},
        "adapter": None
        if adapter_path is None
        else {
            "path": str(Path(adapter_path).resolve()),
            "sha256": file_sha256(Path(adapter_path) / "adapter_model.safetensors"),
        },
        "library_versions": library_versions(),
        "gpu_name": gpu_name(),
        "started_at": datetime.fromtimestamp(start_time, tz=timezone.utc).isoformat(),
        "wall_clock_seconds": round(time.time() - start_time, 1),
        **(extra or {}),
    }
    path = Path(output_dir) / RUN_RECORD_FILE
    with path.open("w") as f:
        json.dump(record, f, indent="\t", ensure_ascii=False, default=str)
    return path


def run_record_matches(output_dir: Path, seed: int, **expected_config: Any) -> bool:
    """True when `output_dir` has a run record of the same seed and the same config values."""
    path = Path(output_dir) / RUN_RECORD_FILE
    if not path.is_file():
        return False
    try:
        record = json.loads(path.read_text())
    except json.JSONDecodeError:
        return False
    config = record.get("config", {})
    return record.get("seed") == seed and all(
        config.get(key) == value for key, value in expected_config.items()
    )


def write_json_atomic(data: Any, path: Path, **kwargs: Any) -> None:
    """Writes JSON to a temp file and renames it, so `path` is never half-written."""
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w") as f:
        json.dump(data, f, **kwargs)
    os.replace(tmp_path, path)
