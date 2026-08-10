"""Deterministic evidence fusion with semantically separate channels."""

from dataclasses import asdict, dataclass
from enum import IntEnum
from typing import Any

from .domain import LocalPrediction, NIMAssessment

FUSION_VERSION = "deterministic-fusion-v1"


class PolicyLevel(IntEnum):
    NORMAL = 0
    OBSERVE = 1
    SUSPICIOUS = 2
    LIKELY_MALICIOUS = 3
    HIGH_CONFIDENCE_ATTACK = 4
    CONTAIN = 5


@dataclass(frozen=True)
class EvidenceChannels:
    classifier_label: str
    classifier_confidence: float
    calibrated_confidence: float | None
    prototype_similarity: float | None
    anomaly_score: float | None
    is_unknown: bool
    nim_verdict: str | None
    nim_reasoning_strength: float | None
    nim_mode: str


@dataclass(frozen=True)
class FusionDecision:
    policy_level: PolicyLevel
    risk_score: float
    evidence: EvidenceChannels
    rules_fired: tuple[str, ...]
    explanation: str
    provenance: dict[str, Any]


class DeterministicFusionEngine:
    def decide(self, local: LocalPrediction, nim: NIMAssessment | None = None,
               containment_enabled: bool = False) -> FusionDecision:
        confidence = local.calibrated_confidence if local.calibrated_confidence is not None else local.confidence
        risk = 0.0
        rules = []
        if local.label != "benign":
            risk += 45.0 * confidence
            rules.append("local_malicious")
        elif confidence >= 0.9:
            risk -= 15.0
            rules.append("local_high_confidence_benign")
        if local.prototype_similarity is not None:
            if local.prototype_label and local.prototype_label != "benign":
                risk += max(0.0, local.prototype_similarity) * 25.0
                rules.append("prototype_malicious")
            elif local.prototype_similarity < 0.15:
                risk += 10.0
                rules.append("prototype_low_similarity")
        if local.anomaly_score is not None:
            risk += min(20.0, max(0.0, local.anomaly_score) * 4.0)
            if local.anomaly_score >= 3.0:
                rules.append("high_anomaly")
        if local.is_unknown:
            risk += 20.0
            rules.append("unknown_or_ood")
        if nim is not None and nim.mode == "influence":
            if nim.verdict == "malicious":
                risk += 12.0
                rules.append("nim_malicious_corroboration")
            elif nim.verdict == "benign":
                risk -= 5.0
                rules.append("nim_benign_challenge")
            elif nim.verdict in {"suspicious", "unknown"}:
                risk += 5.0
                rules.append("nim_uncertainty")
        risk = min(100.0, max(0.0, risk))
        if risk < 10:
            level = PolicyLevel.NORMAL
        elif risk < 25:
            level = PolicyLevel.OBSERVE
        elif risk < 45:
            level = PolicyLevel.SUSPICIOUS
        elif risk < 70:
            level = PolicyLevel.LIKELY_MALICIOUS
        else:
            level = PolicyLevel.HIGH_CONFIDENCE_ATTACK
        local_containment_gate = (
            containment_enabled
            and local.label != "benign"
            and confidence >= 0.95
            and (local.prototype_similarity or 0.0) >= 0.7
            and risk >= 85
        )
        if local_containment_gate:
            level = PolicyLevel.CONTAIN
            rules.append("deterministic_containment_gate")
        channels = EvidenceChannels(
            classifier_label=local.label,
            classifier_confidence=local.confidence,
            calibrated_confidence=local.calibrated_confidence,
            prototype_similarity=local.prototype_similarity,
            anomaly_score=local.anomaly_score,
            is_unknown=local.is_unknown,
            nim_verdict=nim.verdict if nim else None,
            nim_reasoning_strength=nim.self_reported_confidence if nim else None,
            nim_mode=nim.mode if nim else "disabled",
        )
        explanation = self.explain(level, risk, tuple(rules), channels)
        return FusionDecision(
            level,
            risk,
            channels,
            tuple(rules),
            explanation,
            {"fusion_version": FUSION_VERSION, "evidence_channels": asdict(channels)},
        )

    @staticmethod
    def explain(level: PolicyLevel, risk: float, rules: tuple[str, ...], evidence: EvidenceChannels) -> str:
        rule_text = ", ".join(rules) if rules else "no risk rules"
        return (
            f"Policy {level.name} at risk {risk:.1f}; rules: {rule_text}. "
            f"Local={evidence.classifier_label} confidence={evidence.classifier_confidence:.3f}; "
            f"prototype={evidence.prototype_similarity}; anomaly={evidence.anomaly_score}; "
            f"NIM={evidence.nim_verdict} mode={evidence.nim_mode}."
        )
