"""Evidence reconstruction, integrity sealing, and explicit action authority."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()


@dataclass(frozen=True)
class SealedEvent:
    sequence: int
    event_id: str
    timestamp: str
    previous_hash: str
    evidence_hash: str
    record_hash: str
    model_manifest: str
    policy_version: str
    authority: str
    action: str


class EvidenceLedger:
    def __init__(self) -> None:
        self.events: list[SealedEvent] = []

    def append(
        self,
        event_id: str,
        timestamp: str,
        evidence: Any,
        model_manifest: str,
        policy_version: str,
        authority: str,
        action: str,
    ) -> SealedEvent:
        previous_hash = self.events[-1].record_hash if self.events else "0" * 64
        evidence_hash = hashlib.sha256(canonical_json(evidence)).hexdigest()
        body = {
            "sequence": len(self.events),
            "event_id": event_id,
            "timestamp": timestamp,
            "previous_hash": previous_hash,
            "evidence_hash": evidence_hash,
            "model_manifest": model_manifest,
            "policy_version": policy_version,
            "authority": authority,
            "action": action,
        }
        record_hash = hashlib.sha256(canonical_json(body)).hexdigest()
        sealed = SealedEvent(
            sequence=len(self.events),
            event_id=event_id,
            timestamp=timestamp,
            previous_hash=previous_hash,
            evidence_hash=evidence_hash,
            record_hash=record_hash,
            model_manifest=model_manifest,
            policy_version=policy_version,
            authority=authority,
            action=action,
        )
        self.events.append(sealed)
        return sealed

    def merkle_root(self) -> str:
        hashes = [event.record_hash for event in self.events]
        if not hashes:
            return hashlib.sha256(b"").hexdigest()
        while len(hashes) > 1:
            if len(hashes) % 2:
                hashes.append(hashes[-1])
            hashes = [
                hashlib.sha256((hashes[index] + hashes[index + 1]).encode()).hexdigest()
                for index in range(0, len(hashes), 2)
            ]
        return hashes[0]

    def verify(self) -> dict[str, Any]:
        previous = "0" * 64
        for event in self.events:
            body = asdict(event)
            record_hash = body.pop("record_hash")
            if event.previous_hash != previous or hashlib.sha256(canonical_json(body)).hexdigest() != record_hash:
                return {"verified": False, "failed_sequence": event.sequence, "modifications": "DETECTED"}
            previous = event.record_hash
        return {
            "verified": True,
            "events": len(self.events),
            "merkle_root": self.merkle_root(),
            "evidence_chain": "VERIFIED",
            "decision_record": "VERIFIED",
            "model_artifact": "VERIFIED",
            "policy_version": "VERIFIED",
            "modifications": "NONE",
        }


class EvidenceTimeMachine:
    def reconstruct(
        self,
        timestamp: str,
        flows: list[dict[str, Any]],
        decisions: list[dict[str, Any]],
        evidence: list[dict[str, Any]],
        nim: list[dict[str, Any]],
    ) -> dict[str, Any]:
        cutoff = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))

        def partition(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
            available: list[dict[str, Any]] = []
            future: list[dict[str, Any]] = []
            for record in records:
                observed = datetime.fromisoformat(str(record["created_at"]).replace("Z", "+00:00"))
                (available if observed <= cutoff else future).append(record)
            return available, future

        available_flows, future_flows = partition(flows)
        available_decisions, future_decisions = partition(decisions)
        available_evidence, future_evidence = partition(evidence)
        available_nim, future_nim = partition(nim)
        latest = available_decisions[0].get("payload", {}) if available_decisions else {}
        channels = latest.get("evidence", {})
        return {
            "as_of": timestamp,
            "known": {
                "flows": len(available_flows),
                "decisions": len(available_decisions),
                "evidence_points": len(available_evidence),
                "nim_assessments": len(available_nim),
                "channels": channels,
                "policy": latest.get("policy_level", "NORMAL"),
            },
            "not_yet_known": {
                "flows": len(future_flows),
                "decisions": len(future_decisions),
                "evidence_points": len(future_evidence),
                "nim_assessments": len(future_nim),
            },
        }


@dataclass(frozen=True)
class AuthorityRule:
    action: str
    level: int
    authority_scope: str
    autonomous: bool
    approver_role: str | None
    default_ttl: int | None
    rollback_required: bool


class AuthorityGraph:
    RULES = (
        AuthorityRule("OBSERVE", 0, "telemetry", True, None, None, False),
        AuthorityRule("ALERT", 1, "notification", True, None, None, False),
        AuthorityRule("RATE_LIMIT", 2, "network_segment", True, "policy", 300, True),
        AuthorityRule("TEMP_BLOCK", 3, "source_address", False, "soc_analyst", 300, True),
        AuthorityRule("QUARANTINE", 4, "managed_host", False, "senior_analyst", 300, True),
    )

    def authorize(
        self,
        action: str,
        requested_by: str,
        approver_role: str | None = None,
        reason: str = "",
    ) -> dict[str, Any]:
        rule = next((item for item in self.RULES if item.action == action), None)
        if rule is None:
            raise ValueError(f"unknown action: {action}")
        permitted = rule.autonomous or approver_role == rule.approver_role
        return {
            "action": action,
            "requested_by": requested_by,
            "permitted": permitted,
            "permitted_by": "policy-v3" if permitted else None,
            "policy_id": "authority-policy-v3",
            "authority_scope": rule.authority_scope,
            "approver": approver_role,
            "reason": reason,
            "expiry_seconds": rule.default_ttl,
            "rollback_required": rule.rollback_required,
        }

    def serialize(self) -> dict[str, Any]:
        return {
            "rules": [asdict(rule) for rule in self.RULES],
            "edges": [
                {"source": self.RULES[index].action, "target": self.RULES[index + 1].action,
                 "relationship": "REQUIRES_GREATER_AUTHORITY"}
                for index in range(len(self.RULES) - 1)
            ],
        }
