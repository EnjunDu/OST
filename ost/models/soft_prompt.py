"""Loading a learned prompt prefix.

The forecast lane runs behind a learned prefix that is prepended in embedding space
(Eq. (13), App. A.5). At inference the prefix is a fixed tensor read from a checkpoint,
so this module only needs to read one and report its geometry.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import torch

LOGGER = logging.getLogger(__name__)

#: Filename a soft-prompt checkpoint directory is expected to contain.
SOFT_PROMPT_FILENAME = "soft_prompt.pt"


class SoftPromptError(RuntimeError):
    """Raised when a soft-prompt checkpoint cannot be read."""


def load_soft_prompt(
    path: str | Path, *, map_location: str = "cpu"
) -> Tuple[torch.Tensor, int, int]:
    """Read a prefix from ``path``, returning ``(embeddings, num_tokens, hidden_size)``.

    ``path`` may be the checkpoint file itself or a directory containing
    :data:`SOFT_PROMPT_FILENAME`. The stored geometry is checked against the tensor rather
    than trusted, because a prefix that silently loads at the wrong width would shift
    every forecast without raising.
    """
    target = Path(path).expanduser()
    file = target / SOFT_PROMPT_FILENAME if target.is_dir() else target
    if not file.exists():
        raise SoftPromptError(f"no soft prompt at {file}")

    payload = torch.load(file, map_location=map_location)
    if not isinstance(payload, dict) or "embeddings" not in payload:
        raise SoftPromptError(
            f"{file} is not a soft-prompt checkpoint: expected a mapping with an "
            "'embeddings' entry"
        )

    embeddings = payload["embeddings"]
    if not isinstance(embeddings, torch.Tensor) or embeddings.dim() != 2:
        raise SoftPromptError(
            f"{file} must hold a 2-D (num_tokens, hidden_size) tensor, got "
            f"{tuple(getattr(embeddings, 'shape', ()))}"
        )

    num_tokens, hidden_size = (int(embeddings.shape[0]), int(embeddings.shape[1]))
    for key, actual in (("num_tokens", num_tokens), ("hidden_size", hidden_size)):
        declared = payload.get(key)
        if declared is not None and int(declared) != actual:
            raise SoftPromptError(
                f"{file} declares {key}={int(declared)} but its tensor has {actual}"
            )

    LOGGER.info(
        "loaded a %d x %d soft prompt from %s", num_tokens, hidden_size, file
    )
    return embeddings, num_tokens, hidden_size


__all__ = ["SOFT_PROMPT_FILENAME", "SoftPromptError", "load_soft_prompt"]
