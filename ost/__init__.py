"""Omni-Streaming Thinking (OST).

A streaming audio-visual reasoning system that records unresolved interpretations as
claims linked to a future evidence interval, a verifying modality, and the states that
depend on them. When the interval closes, the claim is checked against evidence from the
specified modality; a refutation reduces the influence of the claim and its dependent
states and guides a corrected state update.

This package is the inference implementation: it reads checkpoints from the paths named in
a configuration and runs the streaming loop over a video.

Public surface:

* :mod:`ost.config` — configuration objects mirroring the paper's default table
* :mod:`ost.types` — Omni-State, claim, span and verdict types
* :mod:`ost.runtime` — assembles an orchestrator from a config plus its checkpoints
* :mod:`ost.policy.orchestrator` — the online inference loop
"""

from __future__ import annotations

__version__ = "1.0.0"

__all__ = ["__version__"]
