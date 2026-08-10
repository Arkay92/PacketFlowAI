"""Reference-window drift checks for operational evidence channels."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DriftResult:
    channel: str
    distance: float
    threshold: float
    drifted: bool


class DriftDetector:
    def __init__(self, threshold: float = 0.2, bins: int = 20):
        self.threshold = threshold
        self.bins = bins
        self.references: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def fit(self, channel: str, values: Sequence[float]) -> None:
        if not values:
            raise ValueError("drift reference cannot be empty")
        histogram, edges = np.histogram(values, bins=self.bins, density=False)
        distribution = histogram / max(1, histogram.sum())
        self.references[channel] = distribution, edges

    def evaluate(self, channel: str, values: Sequence[float]) -> DriftResult:
        if channel not in self.references:
            raise ValueError(f"no drift reference for {channel}")
        reference, edges = self.references[channel]
        observed, _ = np.histogram(values, bins=edges, density=False)
        observed = observed / max(1, observed.sum())
        distance = float(0.5 * np.abs(reference - observed).sum())
        return DriftResult(channel, distance, self.threshold, distance >= self.threshold)

    def evaluate_channels(self, channels: Mapping[str, Sequence[float]]) -> list[DriftResult]:
        return [self.evaluate(channel, values) for channel, values in channels.items() if channel in self.references]
