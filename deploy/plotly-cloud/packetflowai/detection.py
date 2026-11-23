"""Local prototype, OOD, anomaly, and calibration components."""

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace

import numpy as np
import torch

from .clustering import UnknownClusterer
from .domain import LocalPrediction


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator else 0.0


@dataclass(frozen=True)
class PrototypeResult:
    label: str
    similarity: float
    margin: float
    scores: dict[str, float]


class PrototypeClassifier:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self._sums: dict[str, np.ndarray] = {}
        self._counts: dict[str, int] = {}

    def update(self, label: str, vector: np.ndarray) -> None:
        values = np.asarray(vector, dtype=np.float64)
        if values.shape != (self.dimension,):
            raise ValueError(f"expected vector shape {(self.dimension,)}, got {values.shape}")
        self._sums[label] = self._sums.get(label, np.zeros(self.dimension)) + values
        self._counts[label] = self._counts.get(label, 0) + 1

    def fit(self, vectors: Iterable[np.ndarray], labels: Iterable[str]) -> None:
        for vector, label in zip(vectors, labels, strict=False):
            self.update(label, vector)

    def prototypes(self) -> dict[str, np.ndarray]:
        return {label: np.where(values >= 0, 1, -1) for label, values in self._sums.items()}

    def predict(self, vector: np.ndarray) -> PrototypeResult:
        scores = {label: cosine_similarity(vector, prototype) for label, prototype in self.prototypes().items()}
        if not scores:
            raise RuntimeError("prototype classifier has not been fitted")
        ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        margin = ordered[0][1] - ordered[1][1] if len(ordered) > 1 else ordered[0][1]
        return PrototypeResult(ordered[0][0], ordered[0][1], margin, scores)


@dataclass(frozen=True)
class OODResult:
    is_unknown: bool
    reasons: tuple[str, ...]
    entropy: float


class OODDetector:
    def __init__(self, minimum_similarity: float = 0.15, minimum_margin: float = 0.03,
                 maximum_entropy: float = 1.5, maximum_anomaly: float = 4.0):
        self.minimum_similarity = minimum_similarity
        self.minimum_margin = minimum_margin
        self.maximum_entropy = maximum_entropy
        self.maximum_anomaly = maximum_anomaly

    def evaluate(self, prototype: PrototypeResult, probabilities: Sequence[float], anomaly_score: float) -> OODResult:
        entropy = -sum(probability * math.log(max(probability, 1e-12)) for probability in probabilities)
        reasons = []
        if prototype.similarity < self.minimum_similarity:
            reasons.append("low_prototype_similarity")
        if prototype.margin < self.minimum_margin:
            reasons.append("low_prototype_margin")
        if entropy > self.maximum_entropy:
            reasons.append("high_entropy")
        if anomaly_score > self.maximum_anomaly:
            reasons.append("high_anomaly")
        return OODResult(bool(reasons), tuple(reasons), entropy)


class AnomalyBaseline:
    def __init__(self):
        self.count = 0
        self.mean: np.ndarray | None = None
        self.m2: np.ndarray | None = None

    def update(self, values: Sequence[float]) -> None:
        vector = np.asarray(values, dtype=np.float64)
        if self.mean is None:
            self.mean = np.zeros_like(vector)
            self.m2 = np.zeros_like(vector)
        if vector.shape != self.mean.shape:
            raise ValueError("anomaly feature shape changed")
        self.count += 1
        delta = vector - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (vector - self.mean)

    def score(self, values: Sequence[float]) -> float:
        if self.mean is None or self.m2 is None or self.count < 2:
            return 0.0
        variance = self.m2 / (self.count - 1)
        standard_deviation = np.sqrt(np.maximum(variance, 1e-9))
        return float(np.sqrt(np.mean(np.square((np.asarray(values) - self.mean) / standard_deviation))))


class TemperatureCalibrator:
    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature

    def fit(self, logits: torch.Tensor, targets: torch.Tensor) -> float:
        if logits.ndim != 2 or len(logits) != len(targets):
            raise ValueError("logits and targets have incompatible shapes")
        candidates = torch.linspace(0.25, 5.0, 96)
        losses = [torch.nn.functional.cross_entropy(logits / candidate, targets).item() for candidate in candidates]
        self.temperature = float(candidates[int(np.argmin(losses))].item())
        return self.temperature

    def probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.softmax(logits / self.temperature, dim=-1)

    @staticmethod
    def expected_calibration_error(probabilities: torch.Tensor, targets: torch.Tensor, bins: int = 10) -> float:
        confidence, predictions = probabilities.max(dim=1)
        total = len(targets)
        error = 0.0
        for lower in torch.linspace(0, 1, bins + 1)[:-1]:
            upper = lower + 1 / bins
            mask = (confidence > lower) & (confidence <= upper)
            if mask.any():
                accuracy = (predictions[mask] == targets[mask]).float().mean()
                error += float(mask.sum() / total) * abs(float(accuracy) - float(confidence[mask].mean()))
        return error

    @staticmethod
    def brier_score(probabilities: torch.Tensor, targets: torch.Tensor) -> float:
        expected = torch.nn.functional.one_hot(targets, probabilities.shape[1]).float()
        return float(torch.mean(torch.sum((probabilities - expected) ** 2, dim=1)).item())


class HybridLocalDetector:
    """Attach prototype, anomaly, OOD, and cluster evidence to a neural prediction."""

    def __init__(self, prototypes: PrototypeClassifier, anomaly: AnomalyBaseline,
                 ood: OODDetector, clusterer: UnknownClusterer | None = None):
        self.prototypes = prototypes
        self.anomaly = anomaly
        self.ood = ood
        self.clusterer = clusterer or UnknownClusterer()

    def enrich(self, neural: LocalPrediction, hypervector: np.ndarray, behavioral_values: Sequence[float],
               timestamp: float, source_ip: str) -> LocalPrediction:
        prototype = self.prototypes.predict(hypervector)
        anomaly_score = self.anomaly.score(behavioral_values)
        ood = self.ood.evaluate(prototype, neural.scores, anomaly_score)
        cluster_id = None
        if ood.is_unknown:
            cluster_id = self.clusterer.assign(
                hypervector, timestamp, source_ip,
                {"neural_label": neural.label, "prototype_label": prototype.label},
            ).cluster_id
        return replace(
            neural,
            prototype_label=prototype.label,
            prototype_similarity=prototype.similarity,
            anomaly_score=anomaly_score,
            is_unknown=ood.is_unknown,
            unknown_reasons=ood.reasons,
            cluster_id=cluster_id,
        )
