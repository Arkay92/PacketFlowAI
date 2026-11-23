"""Typed domain boundaries shared across PacketFlowAI services."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from .features import PacketFeatures


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class FlowFeatures:
    flow_id: str
    source_ip: str
    destination_ip: str
    source_port: int | None
    destination_port: int | None
    protocol: str
    packet_count: int = 0
    byte_count: int = 0
    duration_seconds: float = 0.0
    packets_per_second: float = 0.0
    bytes_per_second: float = 0.0
    forward_packets: int = 0
    reverse_packets: int = 0
    forward_bytes: int = 0
    reverse_bytes: int = 0
    packet_length_mean: float = 0.0
    packet_length_std: float = 0.0
    packet_length_min: int = 0
    packet_length_max: int = 0
    inter_arrival_mean: float = 0.0
    inter_arrival_std: float = 0.0
    syn_count: int = 0
    ack_count: int = 0
    fin_count: int = 0
    rst_count: int = 0
    retransmission_count: int = 0
    burstiness: float = 0.0
    state: str = "NEW"
    unique_destination_hosts: int = 0
    unique_destination_ports: int = 0
    host_connection_rate: float = 0.0
    host_failure_rate: float = 0.0
    protocol_entropy: float = 0.0
    outbound_ratio: float = 0.0
    protocol_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PacketObservation:
    timestamp: float
    source_ip: str
    destination_ip: str
    source_port: int | None
    destination_port: int | None
    protocol: str
    length: int
    tcp_flags: int = 0
    tcp_sequence: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LocalPrediction:
    label: str
    label_index: int
    confidence: float
    scores: tuple[float, ...]
    model_id: str
    model_version: str
    prototype_label: str | None = None
    prototype_similarity: float | None = None
    anomaly_score: float | None = None
    calibrated_confidence: float | None = None
    is_unknown: bool = False
    unknown_reasons: tuple[str, ...] = ()
    cluster_id: str | None = None
    mitre_techniques: tuple[str, ...] = ()


@dataclass(frozen=True)
class NIMAssessment:
    provider: str
    model: str
    assessment: str
    evidence: tuple[str, ...] = ()
    self_reported_confidence: float | None = None
    mode: str = "shadow"
    verdict: str = "unknown"
    attack_family: str | None = None
    contradictions: tuple[str, ...] = ()
    unknown_indicators: tuple[str, ...] = ()
    mitre_techniques: tuple[str, ...] = ()
    recommended_action: str = "observe"
    reason: str = ""
    latency_ms: float | None = None
    cached: bool = False


@dataclass(frozen=True)
class ThreatAssessment:
    event_id: str
    packet_features: PacketFeatures
    local_prediction: LocalPrediction
    risk_score: float
    source_ip: str
    nim_assessment: NIMAssessment | None = None
    created_at: str = field(default_factory=utc_now)


@dataclass(frozen=True)
class ResponseDecision:
    action: str
    reason: str
    policy_level: int
    reversible: bool
    requires_confirmation: bool
    ttl_seconds: int | None = None
    executed: bool = False


@dataclass(frozen=True)
class FeedbackRecord:
    event_id: str
    model_prediction: str
    analyst_label: str | None
    analyst_id: str | None
    adjudicated: bool
    notes: str | None = None
    nim_assessment: str | None = None
    disagreement_state: str | None = None
    analyst_decision: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)
