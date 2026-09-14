"""Deterministic seeding."""

from __future__ import annotations

import os
import random
from typing import Optional

import torch


def set_seed(seed: int, *, deterministic_algorithms: bool = False) -> None:
    """Seed Python, NumPy and PyTorch.

    ``deterministic_algorithms`` additionally asks cuBLAS and PyTorch for deterministic
    kernels. It is off by default because it materially slows large-model inference and is
    only needed when bit-exact reproducibility is being verified.
    """
    seed = int(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2**32))
    except ImportError:  # pragma: no cover - NumPy is a hard dependency in practice
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_algorithms:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def torch_generator(seed: Optional[int]) -> Optional[torch.Generator]:
    if seed is None:
        return None
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


__all__ = ["set_seed", "torch_generator"]
