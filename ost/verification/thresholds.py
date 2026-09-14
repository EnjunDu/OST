"""Verdict bands, the contradiction margin, and the uninformative-interval cap.

A verifier score is turned into a verdict by a per-modality band. The band is not a
single threshold: between refutation and confirmation there is an explicit unresolved
region, and calibration keeps confirmation unavailable for an interval that carries no
information about the claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Tuple

from ost.config import AXIOM_CAP_MARGIN, ModalityThresholds
from ost.types import MODALITIES, Modality, Verdict


class ThresholdError(ValueError):
    """Raised when a band is inconsistent with the calibration constraints."""


def contradiction_margin(score: float, tau_low: float) -> float:
    """Contradiction margin ``gamma_i``.

    Implements Eq. (7): ``gamma_i = [(tau_low - s_i) / tau_low]`` clamped to ``[0, 1]``.
    The margin is zero for any score at or above the refutation threshold, and reaches
    one only for a score of zero, so it grades how far below the threshold the evidence
    pushed the claim.
    """
    if not 0.0 < tau_low < 1.0:
        raise ThresholdError(f"tau_low must lie in (0, 1); got {tau_low}")
    raw = (tau_low - float(score)) / tau_low
    return min(max(raw, 0.0), 1.0)


@dataclass(frozen=True)
class ModalityBand:
    """A per-modality verdict band."""

    modality: Modality
    low: float
    high: float
    axiom_cap: float

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not 0.0 < self.low < self.high < 1.0:
            raise ThresholdError(
                f"band for {self.modality} must satisfy 0 < low < high < 1; "
                f"got low={self.low}, high={self.high}"
            )
        # Implements the App. A.3 cap constraint, which keeps confirmation unavailable for
        # an uninformative interval while preserving the unresolved band:
        #   tau_low <= AXIOM_CAP <= tau_high - 0.05
        tolerance = 1e-9
        if self.low > self.axiom_cap + tolerance:
            raise ThresholdError(
                f"band for {self.modality}: low={self.low} exceeds the cap "
                f"{self.axiom_cap}, so a capped score could still refute"
            )
        if self.high - AXIOM_CAP_MARGIN < self.axiom_cap - tolerance:
            raise ThresholdError(
                f"band for {self.modality}: high={self.high} leaves less than "
                f"{AXIOM_CAP_MARGIN} above the cap {self.axiom_cap}, so a capped score "
                "could confirm"
            )

    def verdict(self, score: float) -> Verdict:
        """Map a score to a verdict.

        Sec. 3.3 and App. A.3: scores above the upper threshold confirm the claim, scores
        below the lower threshold refute it, and the intervening band is unresolved.
        """
        value = float(score)
        if value > self.high:
            return Verdict.CONFIRMED
        if value < self.low:
            return Verdict.REFUTED
        return Verdict.UNRESOLVED

    def margin(self, score: float) -> float:
        return contradiction_margin(score, self.low)

    def cap(self, score: float) -> Tuple[float, bool]:
        """Clamp a score for an uninformative interval.

        Returns the possibly-clamped score and whether clamping occurred.
        """
        value = float(score)
        if value > self.axiom_cap:
            return self.axiom_cap, True
        return value, False


class VerdictBands:
    """The per-modality bands used by the verifier."""

    def __init__(
        self,
        thresholds: Mapping[str, ModalityThresholds],
        axiom_cap: float,
    ) -> None:
        if not 0.0 < axiom_cap < 1.0:
            raise ThresholdError("axiom_cap must lie in (0, 1)")
        self.axiom_cap = float(axiom_cap)
        self._bands: Dict[str, ModalityBand] = {}
        for modality in MODALITIES:
            if modality not in thresholds:
                raise ThresholdError(f"missing verifier band for modality {modality!r}")
            entry = thresholds[modality]
            self._bands[modality] = ModalityBand(
                modality=modality,  # type: ignore[arg-type]
                low=float(entry.low),
                high=float(entry.high),
                axiom_cap=self.axiom_cap,
            )

    def __getitem__(self, modality: str) -> ModalityBand:
        try:
            return self._bands[modality]
        except KeyError as exc:
            raise ThresholdError(f"unknown modality {modality!r}") from exc

    def as_dict(self) -> Dict[str, Dict[str, float]]:
        return {
            modality: {"low": band.low, "high": band.high}
            for modality, band in self._bands.items()
        }

    @classmethod
    def from_config(cls, verifier_config) -> "VerdictBands":  # noqa: ANN001
        return cls(verifier_config.thresholds, verifier_config.axiom_cap)


__all__ = [
    "ModalityBand",
    "ThresholdError",
    "VerdictBands",
    "contradiction_margin",
]
