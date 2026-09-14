"""Path resolution.

No model or output path is ever baked into the source tree. Every path arrives from a CLI
argument, an environment variable or a YAML config, resolved here so the precedence is
stated once. Relative paths are kept relative and resolve against the working directory.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional

#: Environment variables recognised for the common roots.
ENV_MODEL_PATH = "OST_MODEL_PATH"
ENV_OUTPUT_DIR = "OST_OUTPUT_DIR"
ENV_CACHE_DIR = "OST_CACHE_DIR"


class PathError(ValueError):
    """Raised when a required path is neither configured nor discoverable."""


def resolve_path(
    explicit: Optional[str],
    *,
    env_var: Optional[str] = None,
    what: str = "path",
    must_exist: bool = True,
) -> Path:
    """Resolve a path from an explicit value then an environment variable.

    The error message names the concrete ways to supply the path, because "file not found"
    on a research codebase is otherwise an unhelpful dead end.
    """
    candidate = explicit or (os.environ.get(env_var) if env_var else None)
    if not candidate:
        hint = f" or export {env_var}" if env_var else ""
        raise PathError(f"no {what} configured: pass it explicitly{hint}")
    path = Path(candidate).expanduser()
    if must_exist and not path.exists():
        raise PathError(f"{what} does not exist: {path}")
    return path


def resolve_model_path(explicit: Optional[str] = None) -> Path:
    return resolve_path(explicit, env_var=ENV_MODEL_PATH, what="model path")


def resolve_output_dir(explicit: Optional[str] = None, *, default: Optional[str] = None) -> Path:
    """Resolve an output directory, creating it.

    Output never defaults into the source tree: the caller must supply a destination, set
    ``OST_OUTPUT_DIR``, or pass an explicit default that lives outside the repository.
    """
    candidate = explicit or os.environ.get(ENV_OUTPUT_DIR) or default
    if not candidate:
        raise PathError(
            "no output directory configured: pass --output_dir or export OST_OUTPUT_DIR"
        )
    path = Path(candidate).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_cache_dir(explicit: Optional[str] = None) -> Path:
    """Resolve a scratch cache directory, defaulting to the OS temp area."""
    candidate = explicit or os.environ.get(ENV_CACHE_DIR)
    if candidate:
        path = Path(candidate).expanduser()
    else:
        import tempfile

        path = Path(tempfile.gettempdir()) / "ost-cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def first_existing(candidates: Iterable[str | os.PathLike[str]]) -> Optional[Path]:
    for candidate in candidates:
        path = Path(candidate).expanduser()
        if path.exists():
            return path
    return None


__all__ = [
    "ENV_CACHE_DIR",
    "ENV_MODEL_PATH",
    "ENV_OUTPUT_DIR",
    "PathError",
    "first_existing",
    "resolve_cache_dir",
    "resolve_model_path",
    "resolve_output_dir",
    "resolve_path",
]
