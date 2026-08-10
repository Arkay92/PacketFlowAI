"""Analyst-only feedback promotion and active-learning prioritization."""

from collections.abc import Iterable
from dataclasses import dataclass

from .domain import FeedbackRecord, LocalPrediction


def validate_supervised_feedback(record: FeedbackRecord) -> None:
    if not record.adjudicated or not record.analyst_label or not record.analyst_id:
        raise ValueError("supervised feedback requires an adjudicated analyst label and analyst identity")
    if record.provenance.get("label_source") == "nim":
        raise ValueError("NIM assessments cannot become supervised labels")


@dataclass(frozen=True)
class ReviewCandidate:
    event_id: str
    priority: float
    reasons: tuple[str, ...]


class ActiveLearningReviewSelector:
    def rank(self, event_id: str, prediction: LocalPrediction,
             local_nim_disagreement: bool = False, neural_hdc_disagreement: bool = False,
             novel_pattern: bool = False, containment_candidate: bool = False) -> ReviewCandidate:
        priority = 0.0
        reasons = []
        confidence = prediction.calibrated_confidence or prediction.confidence
        if (prediction.anomaly_score or 0.0) >= 3 and confidence < 0.7:
            priority += 30
            reasons.append("high_anomaly_low_confidence")
        if neural_hdc_disagreement:
            priority += 20
            reasons.append("neural_hdc_disagreement")
        if local_nim_disagreement:
            priority += 20
            reasons.append("local_nim_disagreement")
        if prediction.is_unknown or prediction.cluster_id:
            priority += 25
            reasons.append("unknown_or_new_cluster")
        if novel_pattern:
            priority += 15
            reasons.append("novel_pattern")
        if containment_candidate:
            priority += 30
            reasons.append("containment_candidate")
        return ReviewCandidate(event_id, min(priority, 100.0), tuple(reasons))

    def select(self, predictions: Iterable[LocalPrediction]) -> list[str]:
        candidates = [self.rank(str(index), prediction) for index, prediction in enumerate(predictions)]
        return [
            candidate.event_id
            for candidate in sorted(candidates, key=lambda value: -value.priority)
            if candidate.priority
        ]
