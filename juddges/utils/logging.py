from pathlib import Path
import sys
from typing import Any
from loguru import logger


def setup_loguru(extra: dict[str, Any] | None = None, log_file: Path | None = None):
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.configure(extra=extra)  # Default values
    format_log = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | " "<level>{level: <8}</level> | "
    for key in extra.keys():
        format_log += f"<level>{{extra[{key}]}}</level> | "
    format_log += (
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    logger.add(sys.stdout, format=format_log, level="INFO")
    if log_file:
        logger.add(log_file, format=format_log, level="INFO")
