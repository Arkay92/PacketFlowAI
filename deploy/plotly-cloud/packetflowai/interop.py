"""Security sensor, CTI, detection-as-code, and SOC export interoperability."""

from __future__ import annotations

import json
import math
import urllib.request
from collections.abc import Mapping
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .v3_support import CanonicalEvidenceEvent


def _canonical(
    sensor: str,
    event_type: str,
    record: Mapping[str, Any],
    source: str | None = None,
    target: str | None = None,
    identity: str | None = None,
    service: str | None = None,
    confidence: float = 0.8,
) -> CanonicalEvidenceEvent:
    token = json.dumps(dict(record), sort_keys=True, default=str)
    import hashlib

    event_id = f"{sensor}-" + hashlib.sha256(token.encode()).hexdigest()[:14]
    return CanonicalEvidenceEvent(
        event_id,
        sensor,
        event_type,
        str(record.get("timestamp") or datetime.now(UTC).isoformat()),
        source,
        target,
        identity,
        service,
        dict(record),
        confidence,
        {"adapter": f"{sensor}-ocsf-v1"},
    )


class EDRAdapter:
    def adapt(self, record: Mapping[str, Any]) -> CanonicalEvidenceEvent:
        return _canonical(
            "edr",
            str(record.get("event_type", "process_activity")),
            record,
            source=str(record.get("host")),
            identity=str(record.get("user")),
            service=str(record.get("process")),
            confidence=0.9,
        )


class IdentityAdapter:
    def adapt(self, record: Mapping[str, Any]) -> CanonicalEvidenceEvent:
        return _canonical(
            "identity",
            str(record.get("event_type", "authentication")),
            record,
            source=str(record.get("source_ip")),
            target=str(record.get("resource")),
            identity=str(record.get("user")),
            confidence=0.95,
        )


class CloudAuditAdapter:
    def adapt(self, record: Mapping[str, Any]) -> CanonicalEvidenceEvent:
        provider = str(record.get("provider", "cloud"))
        return _canonical(
            provider,
            str(record.get("event_name", "api_activity")),
            record,
            source=str(record.get("source_ip")),
            target=str(record.get("resource")),
            identity=str(record.get("principal")),
            service=str(record.get("service")),
        )


class DNSAdapter:
    def adapt(self, record: Mapping[str, Any]) -> CanonicalEvidenceEvent:
        return _canonical(
            "dns",
            "dns_activity",
            record,
            source=str(record.get("client")),
            target=str(record.get("answer")),
            service=str(record.get("query")),
        )


class ApplicationAdapter:
    def adapt(self, record: Mapping[str, Any]) -> CanonicalEvidenceEvent:
        return _canonical(
            "application",
            str(record.get("event_type", "application_activity")),
            record,
            source=str(record.get("client_ip")),
            target=str(record.get("application")),
            identity=str(record.get("user")),
            service=str(record.get("route")),
        )


class OCSFMapper:
    CLASS_UID = {
        "network_flow": 4001,
        "authentication": 3002,
        "dns_activity": 4003,
        "process_activity": 1007,
        "api_activity": 6003,
        "application_activity": 6001,
    }

    def map(self, event: CanonicalEvidenceEvent) -> dict[str, Any]:
        return {
            "class_uid": self.CLASS_UID.get(event.event_type, 0),
            "category_uid": 4,
            "activity_name": event.event_type,
            "time": event.observed_at,
            "severity_id": round(event.confidence * 5),
            "src_endpoint": {"ip": event.source},
            "dst_endpoint": {"ip": event.target, "svc_name": event.service},
            "actor": {"user": {"name": event.identity}},
            "metadata": {
                "product": {"name": "PacketFlowAI"},
                "original_event_uid": event.event_id,
                "source": event.sensor,
            },
            "unmapped": event.attributes,
        }


class STIX21Exchange:
    def export(self, concepts: list[dict[str, Any]]) -> dict[str, Any]:
        objects = []
        for concept in concepts:
            concept_id = str(concept.get("concept_id", "unknown")).lower()
            objects.append(
                {
                    "type": "indicator",
                    "spec_version": "2.1",
                    "id": f"indicator--{concept_id}",
                    "created": datetime.now(UTC).isoformat(),
                    "modified": datetime.now(UTC).isoformat(),
                    "name": concept.get("name", concept_id),
                    "pattern_type": "stix",
                    "pattern": concept.get("pattern", "[network-traffic:dst_port > 0]"),
                    "confidence": round(float(concept.get("confidence", 0.5)) * 100),
                    "external_references": concept.get("provenance", []),
                }
            )
        return {"type": "bundle", "id": "bundle--packetflowai", "objects": objects}

    def decay(self, confidence: float, age_days: float, half_life_days: float = 30) -> float:
        return confidence * math.pow(0.5, age_days / half_life_days)


class TAXIIClient:
    def __init__(self, base_url: str, token: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.token = token

    def request(self, collection: str, bundle: dict[str, Any] | None = None) -> dict[str, Any]:
        data = json.dumps(bundle).encode() if bundle else None
        request = urllib.request.Request(
            f"{self.base_url}/collections/{collection}/objects/",
            data=data,
            method="POST" if data else "GET",
            headers={
                "Accept": "application/taxii+json;version=2.1",
                **({"Authorization": f"Bearer {self.token}"} if self.token else {}),
            },
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.loads(response.read())


class SigmaRuleEngine:
    def import_rule(self, rule: Mapping[str, Any]) -> dict[str, Any]:
        detection = rule.get("detection")
        if not isinstance(detection, Mapping) or "condition" not in detection:
            raise ValueError("Sigma rule requires detection and condition")
        return {
            "id": str(rule.get("id")),
            "title": str(rule.get("title")),
            "status": rule.get("status", "experimental"),
            "detection": dict(detection),
            "tags": list(rule.get("tags", [])),
            "provenance": rule.get("author"),
        }

    def evaluate(self, rule: Mapping[str, Any], event: Mapping[str, Any]) -> bool:
        detection = rule["detection"]
        selections = {
            name: value for name, value in detection.items() if name != "condition" and isinstance(value, Mapping)
        }
        matches = {
            name: all(self._match(event.get(key), expected) for key, expected in values.items())
            for name, values in selections.items()
        }
        condition = str(detection["condition"])
        if condition == "all of them":
            return all(matches.values())
        if condition == "1 of them":
            return any(matches.values())
        return bool(matches.get(condition, False))

    @staticmethod
    def _match(actual: Any, expected: Any) -> bool:
        values = expected if isinstance(expected, list) else [expected]
        return any(str(value).lower() in str(actual).lower() for value in values)

    def generate_candidate(self, pattern: Mapping[str, Any], technique: str) -> dict[str, Any]:
        return {
            "title": "PacketFlowAI validated behavioural candidate",
            "status": "test",
            "author": "PacketFlowAI",
            "detection": {"selection": dict(pattern), "condition": "selection"},
            "tags": [f"attack.{technique.lower()}"],
            "packetflow": {"shadow": True, "requires_review": True},
        }

    def simulate(
        self, rule: Mapping[str, Any], events: list[Mapping[str, Any]], known_false_positives: set[int] | None = None
    ) -> dict[str, Any]:
        matches = [index for index, event in enumerate(events) if self.evaluate(rule, event)]
        false_positives = len(set(matches) & (known_false_positives or set()))
        return {
            "matches": matches,
            "match_count": len(matches),
            "false_positive_count": false_positives,
            "false_positive_estimate": false_positives / max(1, len(matches)),
            "mode": "shadow",
        }


class SIEMExporter:
    def export(self, event: CanonicalEvidenceEvent, target: str) -> str:
        payload = asdict(event)
        if target in {"json", "webhook", "elastic", "splunk", "sentinel", "chronicle"}:
            return json.dumps(payload, sort_keys=True, default=str)
        if target == "cef":
            severity = round(event.confidence * 10)
            return (
                f"CEF:0|PacketFlowAI|Sensor Fabric|4.0|{event.event_type}|"
                f"{event.event_type}|{severity}|src={event.source} dst={event.target}"
            )
        if target == "syslog":
            return f"<134>1 {event.observed_at} packetflowai - - - {json.dumps(payload, default=str)}"
        raise ValueError(f"unsupported SIEM target: {target}")


class AgreementMatrix:
    def build(self, sigma: bool, decision: Mapping[str, Any]) -> dict[str, Any]:
        evidence = decision.get("evidence", {})
        return {
            "sigma": "MATCH" if sigma else "NO MATCH",
            "hdc": evidence.get("prototype_label", "UNKNOWN"),
            "neural": evidence.get("classifier_label", "UNKNOWN"),
            "anomaly": evidence.get("anomaly_score"),
            "nim": evidence.get("nim_verdict", "NOT INVOKED"),
        }


class DetectionRepository:
    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)

    def save(self, rule: Mapping[str, Any], tests: Mapping[str, Any]) -> Path:
        rule_id = str(rule.get("id") or rule.get("title", "candidate")).replace("/", "-")
        version = len(list(self.root.glob(f"{rule_id}-*.json"))) + 1
        path = self.root / f"{rule_id}-{version}.json"
        path.write_text(
            json.dumps(
                {
                    "version": version,
                    "rule": dict(rule),
                    "tests": dict(tests),
                    "created_at": datetime.now(UTC).isoformat(),
                },
                sort_keys=True,
                indent=2,
            ),
            encoding="utf-8",
        )
        return path

    def promote(self, path: Path) -> dict[str, Any]:
        record = json.loads(path.read_text(encoding="utf-8"))
        if not record["tests"].get("passed"):
            raise ValueError("detection rule cannot be promoted without passing tests")
        record["rule"]["status"] = "stable"
        (self.root / "active.json").write_text(json.dumps(record, sort_keys=True, indent=2), encoding="utf-8")
        return record


class InfrastructureGraph:
    def __init__(self):
        self.entities: dict[str, dict[str, Any]] = {}
        self.relationships: list[dict[str, Any]] = []

    def add(self, value: str, kind: str, confidence: float, provenance: str, observed_at: str) -> None:
        self.entities[value] = {
            "value": value,
            "kind": kind,
            "confidence": confidence,
            "provenance": provenance,
            "observed_at": observed_at,
        }

    def connect(self, source: str, target: str, relationship: str) -> None:
        self.relationships.append({"source": source, "target": target, "relationship": relationship})

    def current_confidence(self, value: str, as_of: str) -> float:
        entity = self.entities[value]
        age = (datetime.fromisoformat(as_of) - datetime.fromisoformat(entity["observed_at"])).total_seconds() / 86400
        return STIX21Exchange().decay(float(entity["confidence"]), max(0, age))


class OTelEventExporter:
    def log_event(self, event: CanonicalEvidenceEvent) -> dict[str, Any]:
        return {
            "timestamp": event.observed_at,
            "severity_text": "WARN" if event.confidence >= 0.7 else "INFO",
            "body": event.event_type,
            "attributes": {
                "event.name": "packetflowai.evidence",
                "event.domain": "security",
                "packetflow.event_id": event.event_id,
                "packetflow.sensor": event.sensor,
                "packetflow.confidence": event.confidence,
            },
        }


class CaseManagementExporter:
    def export(self, narrative: Mapping[str, Any], target: str) -> dict[str, Any]:
        if target not in {"generic", "jira", "servicenow", "thehive"}:
            raise ValueError(f"unsupported case target: {target}")
        return {
            "target": target,
            "title": f"PacketFlowAI {narrative.get('campaign_id', 'investigation')}",
            "description": narrative.get("executive_summary"),
            "timeline": narrative.get("technical_timeline", []),
            "labels": ["packetflowai", "security-incident"],
            "evidence_grounded": True,
        }
