"""Logging setup.

Log records deliberately carry no hostname, username or working directory, so that a log
file shared alongside results does not reveal where it was produced.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_DATE_FORMAT = "%H:%M:%S"


def configure_logging(
    level: str = "INFO",
    *,
    log_file: Optional[str] = None,
    quiet_libraries: bool = True,
) -> None:
    """Install a single stream handler, plus an optional file handler."""
    numeric = getattr(logging, str(level).upper(), None)
    if not isinstance(numeric, int):
        raise ValueError(f"unknown log level {level!r}")

    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)

    formatter = logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT)
    stream = logging.StreamHandler(stream=sys.stderr)
    stream.setFormatter(formatter)
    root.addHandler(stream)

    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    root.setLevel(numeric)

    if quiet_libraries:
        for name in ("transformers", "accelerate", "peft", "urllib3", "filelock", "PIL"):
            logging.getLogger(name).setLevel(max(numeric, logging.WARNING))


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


__all__ = ["configure_logging", "get_logger"]
