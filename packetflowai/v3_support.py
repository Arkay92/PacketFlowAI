"""Supporting v3 intelligence, learning, investigation, and platform contracts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

import numpy as np

from .clustering import UnknownCluster, cosine_similarity


@dataclass(frozen=True)
class CanonicalEvidenceEvent:
    event_id: str
    sensor: str
    event_type: str
    observed_at: str
    source: str | None
    target: str | None
    identity: str | None
    service: str | None
    attributes: dict[str, Any]
    confidence: float
    provenance: dict[str, Any]


class SensorAdapter(Protocol):
    def adapt(self, record: Mapping[str, Any]) -> CanonicalEvidenceEvent: ...


def _event_id(sensor: str, record: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(
        json.dumps(dict(record), sort_keys=True, default=str).encode()
    ).hexdigest()[:14]
    return f"{sensor}-{digest}"


class ZeekAdapter:
    def adapt(self, record: Mapping[str, Any]) -> CanonicalEvidenceEvent:
        timestamp = str(record.get("ts") or datetime.now(UTC).isoformat())
        return CanonicalEvidenceEvent(
            _event_id("zeek", record), "zeek", str(record.get("_path", "conn")), timestamp,
            str(record.get("id.orig_h")) if record.get("id.orig_h") else None,
            str(record.get("id.resp_h")) if record.get("id.resp_h") else None,
            str(record.get("user")) if record.get("user") else None,
            str(record.get("service")) if record.get("service") else None,
            dict(record), .9, {"adapter": "zeek-v1", "native_id": record.get("uid")},
        )


class SuricataAdapter:
    def adapt(self, record: Mapping[str, Any]) -> CanonicalEvidenceEvent:
        raw_alert = record.get("alert")
        alert: Mapping[str, Any] = raw_alert if isinstance(raw_alert, Mapping) else {}
        return CanonicalEvidenceEvent(
            _event_id("suricata", record), "suricata", str(record.get("event_type", "flow")),
            str(record.get("timestamp") or datetime.now(UTC).isoformat()),
            str(record.get("src_ip")) if record.get("src_ip") else None,
            str(record.get("dest_ip")) if record.get("dest_ip") else None,
            None, str(record.get("app_proto")) if record.get("app_proto") else None,
            dict(record), float(alert.get("severity", 1)) / 3 if alert else .6,
            {"adapter": "suricata-eve-v1", "signature_id": alert.get("signature_id")},
        )


class PacketFlowAdapter:
    def adapt(self, record: Mapping[str, Any]) -> CanonicalEvidenceEvent:
        raw_payload = record.get("payload")
        payload: Mapping[str, Any] = raw_payload if isinstance(raw_payload, Mapping) else record
        return CanonicalEvidenceEvent(
            str(payload.get("flow_id") or _event_id("packetflowai", record)), "packetflowai", "network_flow",
            str(record.get("created_at") or datetime.now(UTC).isoformat()),
            str(payload.get("source_ip")) if payload.get("source_ip") else None,
            str(payload.get("destination_ip")) if payload.get("destination_ip") else None,
            None, str(payload.get("destination_port")) if payload.get("destination_port") is not None else None,
            dict(payload), 1.0, {"adapter": "packetflowai-v1"},
        )


class PrivacyTransformer:
    def __init__(self, salt: str = "packetflowai"):
        self.salt = salt

    def represent(self, value: str, mode: str, role: str | None = None) -> str:
        if mode == "raw":
            return value
        if mode == "role":
            return role or "unknown-role"
        digest = hashlib.sha256(f"{self.salt}:{value}".encode()).hexdigest()
        if mode == "hashed":
            return digest
        if mode == "pseudonymous":
            return f"entity-{digest[:12]}"
        raise ValueError(f"unsupported privacy mode: {mode}")


@dataclass(frozen=True)
class ThreatIntelEntity:
    indicator: str
    entity_type: str
    relationships: tuple[tuple[str, str], ...]
    confidence: float
    source: str
    observed_at: str


class STIXThreatIntelAdapter:
    def adapt(self, bundle: Mapping[str, Any]) -> list[ThreatIntelEntity]:
        entities = []
        for item in bundle.get("objects", []):
            if not isinstance(item, Mapping) or item.get("type") not in {
                "indicator", "malware", "campaign", "domain-name",
            }:
                continue
            indicator = str(item.get("pattern") or item.get("name") or item.get("value") or item.get("id"))
            entities.append(ThreatIntelEntity(
                indicator,
                str(item.get("type")),
                tuple(
                    (str(key), str(value))
                    for key, value in item.items()
                    if key in {"created_by_ref", "object_marking_refs"}
                ),
                min(1.0, max(0.0, float(item.get("confidence", 50)) / 100)),
                str(item.get("created_by_ref", "stix")),
                str(item.get("created") or datetime.now(UTC).isoformat()),
            ))
        return entities


@dataclass
class ThreatMemoryConcept:
    concept_id: str
    cluster_id: str
    first_seen: float
    instances: int
    hosts: int
    internal_similarity: float
    traits: tuple[str, ...]
    status: str = "provisional"


class EmergentThreatMemory:
    def __init__(self, minimum_instances: int = 20, minimum_similarity: float = .85):
        self.minimum_instances = minimum_instances
        self.minimum_similarity = minimum_similarity
        self.concepts: dict[str, ThreatMemoryConcept] = {}

    def observe(self, cluster: UnknownCluster) -> ThreatMemoryConcept | None:
        if cluster.sample_count < self.minimum_instances or cluster.internal_similarity < self.minimum_similarity:
            return None
        concept_id = "PF-UNKNOWN-" + hashlib.sha256(cluster.cluster_id.encode()).hexdigest()[:6].upper()
        concept = ThreatMemoryConcept(
            concept_id, cluster.cluster_id, cluster.first_seen, cluster.sample_count, len(cluster.source_ips),
            cluster.internal_similarity,
            tuple(key for key, _ in sorted(cluster.common_characteristics.items(), key=lambda item: -item[1])[:6]),
        )
        self.concepts[concept_id] = concept
        return concept

    def adjudicate(self, concept_id: str, family: str) -> ThreatMemoryConcept:
        concept = self.concepts[concept_id]
        concept.status = f"confirmed:{family}"
        return concept


@dataclass(frozen=True)
class PrototypeVersion:
    version: int
    vector: tuple[float, ...]
    sample_count: int
    parent_version: int | None
    holdout_score: float
    created_at: str


class ContinualPrototypeStore:
    def __init__(self, minimum_holdout_score: float = .8):
        self.minimum_holdout_score = minimum_holdout_score
        self.history: dict[str, list[PrototypeVersion]] = {}

    def update(self, label: str, samples: Iterable[np.ndarray], holdout_score: float) -> PrototypeVersion:
        if holdout_score < self.minimum_holdout_score:
            raise ValueError("prototype update rejected by holdout guardrail")
        vectors = [np.asarray(sample, dtype=np.float64) for sample in samples]
        if not vectors:
            raise ValueError("prototype update requires samples")
        prior = self.history.get(label, [])
        vector = np.mean(vectors, axis=0)
        if prior:
            previous = np.asarray(prior[-1].vector)
            vector = (
                previous * prior[-1].sample_count + vector * len(vectors)
            ) / (prior[-1].sample_count + len(vectors))
        version = PrototypeVersion(
            len(prior) + 1, tuple(float(value) for value in vector),
            (prior[-1].sample_count if prior else 0) + len(vectors), prior[-1].version if prior else None,
            holdout_score, datetime.now(UTC).isoformat(),
        )
        self.history.setdefault(label, []).append(version)
        return version

    def rollback(self, label: str) -> PrototypeVersion:
        if len(self.history.get(label, [])) < 2:
            raise RuntimeError("no prototype version available for rollback")
        self.history[label].pop()
        return self.history[label][-1]


@dataclass(frozen=True)
class FederatedArtifact:
    site_id: str
    concept_id: str
    hypervector_fingerprint: str
    similarity_signature: tuple[float, ...]
    sample_count: int
    confidence: float
    provenance: str


class FederatedConsensusEngine:
    def assess(self, artifacts: Iterable[FederatedArtifact]) -> dict[str, Any]:
        items = list(artifacts)
        if not items:
            return {"score": 0.0, "sites": 0, "assessment": "insufficient evidence"}
        total_weight = sum(max(1, item.sample_count) * item.confidence for item in items)
        weighted = sum(max(1, item.sample_count) * item.confidence ** 2 for item in items)
        score = weighted / total_weight if total_weight else 0.0
        return {
            "score": score,
            "sites": len({item.site_id for item in items}),
            "assessment": "high consensus" if score >= .8 else "mixed consensus" if score >= .5 else "weak consensus",
            "provenance": [item.provenance for item in items],
            "raw_traffic_exchanged": False,
        }


class DisagreementEngine:
    def find(self, decisions: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
        disagreements = []
        for decision in decisions:
            channels = decision.get("evidence", {})
            local = str(channels.get("classifier_label", "unknown"))
            nim = str(channels.get("nim_verdict", "unknown"))
            anomaly = float(channels.get("anomaly_score") or 0.0)
            if (nim not in {"unknown", "none", "None", local}) or (local == "benign" and anomaly >= .75):
                disagreements.append({
                    "event_id": decision.get("event_id"), "local": local, "nim": nim,
                    "anomaly": anomaly, "recommendation": "analyst attention recommended",
                })
        return disagreements


class ReadOnlyInvestigator:
    def query(self, question: str, records: Mapping[str, Any]) -> dict[str, Any]:
        lowered = question.lower()
        if any(token in lowered for token in {"delete", "block", "quarantine", "change", "execute"}):
            return {"mode": "read_only", "refused": True, "reason": "investigation cannot modify records or act"}
        matched = []
        for category, values in records.items():
            if not isinstance(values, list):
                continue
            for value in values:
                text = json.dumps(value, default=str).lower()
                if any(token in text for token in lowered.split() if len(token) > 3):
                    matched.append({"category": category, "record": value})
        return {"mode": "read_only", "refused": False, "matches": matched[:20], "evidence_count": len(matched)}

    def evaluate_hypothesis(self, name: str, evidence: Mapping[str, Any]) -> dict[str, Any]:
        features = {str(key).lower(): value for key, value in evidence.items()}
        supporting, contradicting = [], []
        if "credential" in name.lower():
            if features.get("authentication_failures", 0) > 5:
                supporting.append("repeated authentication attempts")
            if features.get("unique_usernames", 0) > 3:
                supporting.append("multiple usernames observed")
            if features.get("unique_destination_ports", 0) > 10:
                contradicting.append("wide destination port range")
            if features.get("authentication_protocol_ratio", 1.0) < .15:
                contradicting.append("authentication protocol present in few flows")
        strength = len(supporting) - len(contradicting)
        return {
            "hypothesis": name,
            "supporting": supporting,
            "contradicting": contradicting,
            "assessment": "strong support" if strength >= 2 else "weak support" if strength > 0 else "not supported",
            "alternative": "network reconnaissance" if "credential" in name.lower() else "unknown",
        }


class CaseNarrativeBuilder:
    def build(self, campaign_id: str, events: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
        ordered = sorted(events, key=lambda item: str(item.get("created_at", "")))
        timeline = []
        for event in ordered:
            payload = event.get("payload", {})
            label = payload.get("evidence", {}).get("classifier_label") or payload.get("action") or "activity"
            timeline.append({
                "timestamp": event.get("created_at"),
                "event_id": event.get("event_id") or payload.get("event_id") or payload.get("flow_id"),
                "summary": f"PacketFlowAI observed {str(label).replace('_', ' ')}.",
            })
        return {
            "campaign_id": campaign_id,
            "executive_summary": f"{len(timeline)} correlated events form the current campaign assessment.",
            "technical_timeline": timeline,
            "unresolved_questions": ["Was access successful?", "Is additional infrastructure involved?"],
            "immutable_evidence_grounded": True,
        }


@dataclass(frozen=True)
class PlaybookStep:
    order: int
    action: str
    authority_level: int
    purpose: str


class BoundedPlaybookEngine:
    PORT_SCAN = (
        PlaybookStep(1, "increase_observation", 0, "raise local sensor fidelity"),
        PlaybookStep(2, "gather_flow_evidence", 0, "extend flow history"),
        PlaybookStep(3, "query_prior_behavior", 0, "compare host baseline"),
        PlaybookStep(4, "check_asset_status", 0, "identify protected dependencies"),
        PlaybookStep(5, "alert_analyst", 1, "request human review"),
        PlaybookStep(6, "rate_limit", 2, "reduce scan velocity"),
    )

    def plan(self, authority_level: int) -> dict[str, Any]:
        executable = [asdict(step) for step in self.PORT_SCAN if step.authority_level <= authority_level]
        proposed = [asdict(step) for step in self.PORT_SCAN if step.authority_level > authority_level]
        return {"playbook": "Possible Port Scan", "executable": executable, "awaiting_authority": proposed}


class AdaptiveSensorController:
    def profile(self, risk_score: float) -> dict[str, Any]:
        if risk_score >= 70:
            return {"fidelity": "high", "history_seconds": 3600, "metadata": "extended", "window_seconds": 5}
        if risk_score >= 35:
            return {"fidelity": "elevated", "history_seconds": 1800, "metadata": "standard+", "window_seconds": 15}
        return {"fidelity": "light", "history_seconds": 300, "metadata": "minimal", "window_seconds": 60}


class FastPathBackend(Protocol):
    name: str

    def capability(self) -> dict[str, Any]: ...


class EBPFXDPBackend:
    name = "ebpf_xdp"

    def capability(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "status": "platform_contract",
            "kernel_role": "minimal flow telemetry and coarse filtering",
            "userspace_role": "HDC, anomaly, fusion, policy, evidence",
            "requires": ["Linux", "CAP_BPF or root", "compiled BPF object"],
        }


class HDCAccelerationRegistry:
    def capabilities(self) -> list[dict[str, Any]]:
        return [
            {"backend": "cpu", "status": "active", "operations": ["bind", "permute", "similarity"]},
            {"backend": "cuda", "status": "available_when_torch_cuda", "operations": ["bind", "similarity"]},
            {"backend": "binary_simd", "status": "research_contract", "operations": ["xor", "popcount"]},
            {"backend": "fpga", "status": "research_contract", "operations": ["bind", "bundle", "similarity"]},
        ]


class AttackLab:
    def evaluate(self, versions: Mapping[str, Iterable[Mapping[str, Any]]]) -> dict[str, Any]:
        reports = {}
        for version, events in versions.items():
            ordered = sorted(events, key=lambda item: float(item.get("timestamp", 0.0)))
            first = ordered[0]["timestamp"] if ordered else None
            sufficient = next(
                (
                    event["timestamp"]
                    for event in ordered
                    if float(event.get("risk_score", 0)) >= 45
                ),
                None,
            )
            reports[version] = {
                "detected": sufficient is not None,
                "time_to_understand_seconds": (
                    sufficient - first
                    if first is not None and sufficient is not None
                    else None
                ),
                "events_processed": len(ordered),
                "policy_overreaction_count": sum(bool(event.get("false_positive")) for event in ordered),
            }
        understood = {
            version: float(report["time_to_understand_seconds"])
            for version, report in reports.items()
            if report["time_to_understand_seconds"] is not None
        }
        return {
            "reports": reports,
            "best": min(understood, key=lambda version: understood[version]) if understood else None,
        }


def similarity_fingerprint(vector: np.ndarray) -> str:
    packed = np.packbits(np.asarray(vector) > 0).tobytes()
    return hashlib.sha256(packed).hexdigest()


def compare_federated(left: FederatedArtifact, right: FederatedArtifact) -> float:
    return cosine_similarity(np.asarray(left.similarity_signature), np.asarray(right.similarity_signature))
