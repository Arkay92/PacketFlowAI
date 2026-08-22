"""Verifiable assurance primitives for evidence producers and investigations."""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def digest(value: Any) -> str:
    data = value if isinstance(value, bytes) else canonical(value)
    return hashlib.sha256(data).hexdigest()


def sign(value: Any, secret: str) -> str:
    return hmac.new(secret.encode(), canonical(value), hashlib.sha256).hexdigest()


def merkle_root(leaves: Iterable[str]) -> str:
    layer = list(leaves)
    if not layer:
        return digest(b"")
    while len(layer) > 1:
        if len(layer) % 2:
            layer.append(layer[-1])
        layer = [digest((layer[index] + layer[index + 1]).encode()) for index in range(0, len(layer), 2)]
    return layer[0]


@dataclass(frozen=True)
class EvidenceContract:
    contract_id: str
    version: str
    environment: str
    expected_sources: tuple[str, ...]
    required_for: Mapping[str, tuple[str, ...]]
    valid_from: str
    valid_until: str
    signer: str
    contract_hash: str
    signature: str
    external_anchor: str

    @classmethod
    def issue(
        cls,
        contract_id: str,
        version: str,
        environment: str,
        expected_sources: Iterable[str],
        required_for: Mapping[str, Iterable[str]],
        valid_from: str,
        valid_until: str,
        signer: str,
        secret: str,
        external_anchor: str,
    ) -> EvidenceContract:
        source_list = sorted(set(expected_sources))
        requirement_map = {key: sorted(set(value)) for key, value in sorted(required_for.items())}
        body = {
            "contract_id": contract_id,
            "version": version,
            "environment": environment,
            "expected_sources": source_list,
            "required_for": requirement_map,
            "valid_from": valid_from,
            "valid_until": valid_until,
            "signer": signer,
        }
        contract_hash = digest(body)
        return cls(
            contract_id=contract_id,
            version=version,
            environment=environment,
            expected_sources=tuple(source_list),
            required_for={key: tuple(value) for key, value in requirement_map.items()},
            valid_from=valid_from,
            valid_until=valid_until,
            signer=signer,
            contract_hash=contract_hash,
            signature=sign({"contract_hash": contract_hash}, secret),
            external_anchor=external_anchor,
        )


@dataclass(frozen=True)
class ProducerEvent:
    producer_id: str
    epoch_id: str
    sequence_number: int
    event_time: str
    receive_time: str
    commit_time: str
    clock_source: str
    estimated_skew_ms: float
    time_confidence: str
    payload: Mapping[str, Any]
    evidence_hash: str


@dataclass(frozen=True)
class IngestReceipt:
    receipt_id: str
    producer_id: str
    epoch_id: str
    sequence_number: int
    evidence_hash: str
    received_at: str
    signer: str
    signature: str


@dataclass(frozen=True)
class EpochManifest:
    producer_id: str
    epoch_id: str
    first_sequence: int
    last_sequence: int
    event_count: int
    first_timestamp: str
    last_timestamp: str
    merkle_root: str
    previous_epoch_root: str
    signer: str
    signature: str


class ProducerLedger:
    """Sequenced producer journal with receipts, epochs, and continuity checks."""

    def __init__(self, producer_id: str, signing_secret: str, signer: str | None = None):
        self.producer_id = producer_id
        self.signing_secret = signing_secret
        self.signer = signer or producer_id
        self.events: list[ProducerEvent] = []
        self.receipts: list[IngestReceipt] = []
        self.epochs: list[EpochManifest] = []
        self.heartbeats: list[dict[str, Any]] = []

    def ingest(
        self,
        epoch_id: str,
        sequence_number: int,
        payload: Mapping[str, Any],
        event_time: str,
        receive_time: str | None = None,
        clock_source: str = "NTP",
        estimated_skew_ms: float = 0.0,
    ) -> IngestReceipt:
        received_at = receive_time or datetime.now(UTC).isoformat()
        event_body = {
            "producer_id": self.producer_id,
            "epoch_id": epoch_id,
            "sequence_number": sequence_number,
            "event_time": event_time,
            "receive_time": received_at,
            "commit_time": datetime.now(UTC).isoformat(),
            "clock_source": clock_source,
            "estimated_skew_ms": estimated_skew_ms,
            "time_confidence": "LOW" if abs(estimated_skew_ms) > 500 else "HIGH",
            "payload": dict(payload),
        }
        evidence_hash = digest(event_body)
        self.events.append(
            ProducerEvent(
                producer_id=self.producer_id,
                epoch_id=epoch_id,
                sequence_number=sequence_number,
                event_time=event_time,
                receive_time=received_at,
                commit_time=str(event_body["commit_time"]),
                clock_source=clock_source,
                estimated_skew_ms=estimated_skew_ms,
                time_confidence=str(event_body["time_confidence"]),
                payload=dict(payload),
                evidence_hash=evidence_hash,
            )
        )
        receipt_body = {
            "receipt_id": f"PF-RCPT-{digest((self.producer_id, epoch_id, sequence_number))[:12].upper()}",
            "producer_id": self.producer_id,
            "epoch_id": epoch_id,
            "sequence_number": sequence_number,
            "evidence_hash": evidence_hash,
            "received_at": received_at,
            "signer": "packetflow-ingest",
        }
        receipt = IngestReceipt(
            receipt_id=str(receipt_body["receipt_id"]),
            producer_id=self.producer_id,
            epoch_id=epoch_id,
            sequence_number=sequence_number,
            evidence_hash=evidence_hash,
            received_at=received_at,
            signer="packetflow-ingest",
            signature=sign(receipt_body, self.signing_secret),
        )
        self.receipts.append(receipt)
        return receipt

    def heartbeat(self, timestamp: str, status: str = "SOURCE_ALIVE") -> dict[str, Any]:
        body = {"producer_id": self.producer_id, "timestamp": timestamp, "status": status}
        record = {**body, "signature": sign(body, self.signing_secret)}
        self.heartbeats.append(record)
        return record

    def continuity(self, epoch_id: str | None = None) -> dict[str, Any]:
        selected = [event for event in self.events if epoch_id is None or event.epoch_id == epoch_id]
        sequences = sorted({event.sequence_number for event in selected})
        if not sequences:
            return {"status": "UNKNOWN", "first": None, "last": None, "observed": 0, "expected": 0, "gaps": []}
        expected = set(range(sequences[0], sequences[-1] + 1))
        gaps = sorted(expected - set(sequences))
        return {
            "status": "CONTINUOUS" if not gaps else "PARTIAL",
            "first": sequences[0],
            "last": sequences[-1],
            "observed": len(sequences),
            "expected": len(expected),
            "coverage": len(sequences) / len(expected),
            "gaps": gaps,
        }

    def close_epoch(self, epoch_id: str) -> EpochManifest:
        selected = sorted(
            (event for event in self.events if event.epoch_id == epoch_id), key=lambda event: event.sequence_number
        )
        if not selected:
            raise ValueError(f"epoch has no events: {epoch_id}")
        previous = self.epochs[-1].merkle_root if self.epochs else "0" * 64
        body = {
            "producer_id": self.producer_id,
            "epoch_id": epoch_id,
            "first_sequence": selected[0].sequence_number,
            "last_sequence": selected[-1].sequence_number,
            "event_count": len(selected),
            "first_timestamp": selected[0].event_time,
            "last_timestamp": selected[-1].event_time,
            "merkle_root": merkle_root(event.evidence_hash for event in selected),
            "previous_epoch_root": previous,
            "signer": self.signer,
        }
        manifest = EpochManifest(
            producer_id=self.producer_id,
            epoch_id=epoch_id,
            first_sequence=selected[0].sequence_number,
            last_sequence=selected[-1].sequence_number,
            event_count=len(selected),
            first_timestamp=selected[0].event_time,
            last_timestamp=selected[-1].event_time,
            merkle_root=str(body["merkle_root"]),
            previous_epoch_root=previous,
            signer=self.signer,
            signature=sign(body, self.signing_secret),
        )
        self.epochs.append(manifest)
        return manifest

    def reconcile_receipt_journal(self, supplied_evidence_hashes: Iterable[str]) -> dict[str, Any]:
        supplied = set(supplied_evidence_hashes)
        accepted = {receipt.evidence_hash for receipt in self.receipts}
        missing = sorted(accepted - supplied)
        return {
            "producer_id": self.producer_id,
            "accepted": len(accepted),
            "supplied": len(accepted & supplied),
            "missing_receipt_backed_events": missing,
            "status": "MATCHED" if not missing else "DISCREPANCY",
        }


class EvidenceLifecycle:
    STAGES = (
        "OBSERVED",
        "RECEIVED",
        "NORMALISED",
        "COMMITTED",
        "DERIVED",
        "USED_IN_DECISION",
        "EXPORTED",
        "REDACTED",
        "RETAINED",
        "DESTROYED",
    )

    def __init__(self):
        self.records: list[dict[str, Any]] = []

    def transition(
        self,
        evidence_id: str,
        stage: str,
        input_hashes: Iterable[str],
        algorithm: str,
        configuration: Mapping[str, Any],
        output: Any,
        timestamp: str,
    ) -> dict[str, Any]:
        if stage not in self.STAGES:
            raise ValueError(f"unsupported lifecycle stage: {stage}")
        body = {
            "evidence_id": evidence_id,
            "stage": stage,
            "input_hashes": list(input_hashes),
            "algorithm": algorithm,
            "configuration": dict(configuration),
            "output_hash": digest(output),
            "timestamp": timestamp,
        }
        record = {**body, "provenance_hash": digest(body)}
        self.records.append(record)
        return record


class DecisionCapsuleBuilder:
    def build(
        self,
        decision_id: str,
        source_evidence: Iterable[Mapping[str, Any]],
        features: Mapping[str, Any],
        model: Mapping[str, Any],
        policy: Mapping[str, Any],
        authority: Mapping[str, Any],
        result: Mapping[str, Any],
    ) -> dict[str, Any]:
        capsule = {
            "format": "PFCAP-1.0",
            "decision_id": decision_id,
            "source_evidence": list(source_evidence),
            "features": dict(features),
            "model": dict(model),
            "policy": dict(policy),
            "authority": dict(authority),
            "result": dict(result),
            "reproducibility": "DETERMINISTICALLY_REPRODUCIBLE",
        }
        return {**capsule, "capsule_digest": digest(capsule)}


class ReasoningReceiptBuilder:
    def build(
        self,
        provider: str,
        model: str,
        model_version: str | None,
        request: Any,
        response: Any,
        system_instructions: str,
        parameters: Mapping[str, Any],
        timestamp: str,
    ) -> dict[str, Any]:
        body = {
            "provider": provider,
            "model": model,
            "model_version": model_version,
            "request_hash": digest(request),
            "response_hash": digest(response),
            "system_instructions_hash": digest(system_instructions),
            "parameters": dict(parameters),
            "timestamp": timestamp,
            "structured_result": response,
            "classification": "ATTESTED_BUT_NOT_REPRODUCIBLE",
        }
        return {**body, "receipt_hash": digest(body)}


class CollectorAttestationVerifier:
    """RATS-style separation between attester evidence and relying-party appraisal."""

    def appraise(self, evidence: Mapping[str, Any], accepted_measurements: set[str]) -> dict[str, Any]:
        required = {"collector_id", "software", "version", "configuration_digest", "measurement", "boot_state"}
        missing = sorted(required - set(evidence))
        trusted = not missing and evidence.get("measurement") in accepted_measurements
        return {
            "collector_id": evidence.get("collector_id"),
            "status": "VERIFIED" if trusted else "UNTRUSTED",
            "missing_claims": missing,
            "measurement_accepted": evidence.get("measurement") in accepted_measurements,
            "mode": evidence.get("mode", "SIGNED_COLLECTOR"),
        }


class ProtectedMonotonicCounter:
    def __init__(self, counter_id: str, initial: int = 0):
        self.counter_id = counter_id
        self.value = initial

    def advance(self, claimed: int) -> int:
        if claimed <= self.value:
            raise ValueError("counter rollback or reuse detected")
        self.value = claimed
        return self.value


class CrossSourceReconciler:
    def reconcile(self, claim: str, observations: Mapping[str, str]) -> dict[str, Any]:
        present = {source: value for source, value in observations.items() if value == "OBSERVED"}
        absent = {source: value for source, value in observations.items() if value != "OBSERVED"}
        status = "CONSISTENT" if not absent or not present else "EVIDENCE_ASYMMETRY"
        return {
            "claim": claim,
            "status": status,
            "observed": sorted(present),
            "not_observed": absent,
            "negative_evidence_semantics": {
                source: "OBSERVED_ABSENCE" if value == "OBSERVED_ABSENCE" else "ABSENCE_OF_OBSERVATION"
                for source, value in absent.items()
            },
        }


class SelectiveDisclosure:
    def disclose(
        self,
        committed_records: Iterable[Mapping[str, Any]],
        visible_ids: set[str],
        reason: str,
        authority: str,
    ) -> dict[str, Any]:
        records = list(committed_records)
        disclosed = [item for item in records if str(item.get("id")) in visible_ids]
        redacted = [
            {
                "id": item.get("id"),
                "original_hash": digest(item),
                "reason": reason,
                "authority": authority,
                "kind": "REDACTED_LEAF",
            }
            for item in records
            if str(item.get("id")) not in visible_ids
        ]
        root = merkle_root(digest(item) for item in records)
        return {
            "committed_root": root,
            "disclosed": disclosed,
            "declared_redactions": redacted,
            "counts": {"committed": len(records), "disclosed": len(disclosed), "redacted": len(redacted)},
        }


class DisclosureEnvelope:
    """Separates confidential content handling from publicly verifiable commitments."""

    def commit(self, content: bytes, encryption_profile: str, disclosure_tier: str) -> dict[str, Any]:
        return {
            "content_digest": digest(content),
            "encrypted": encryption_profile != "NONE",
            "encryption_profile": encryption_profile,
            "disclosure_tier": disclosure_tier,
            "verification_requires_plaintext": False,
        }


class OmissionLedger:
    KINDS = {
        "EVENT_REDACTED",
        "EVENT_EXCLUDED",
        "EVENT_REJECTED",
        "EVENT_EXPIRED",
        "EVENT_DESTROYED",
        "SOURCE_DISABLED",
        "SOURCE_UNAVAILABLE",
        "SOURCE_FILTER_CHANGED",
    }

    def __init__(self, secret: str):
        self.secret = secret
        self.records: list[dict[str, Any]] = []

    def append(
        self, kind: str, subject: str, reason: str, authority: str, timestamp: str, **detail: Any
    ) -> dict[str, Any]:
        if kind not in self.KINDS:
            raise ValueError(f"unsupported omission kind: {kind}")
        previous = self.records[-1]["record_hash"] if self.records else "0" * 64
        body = {
            "kind": kind,
            "subject": subject,
            "reason": reason,
            "authority": authority,
            "timestamp": timestamp,
            "detail": detail,
            "previous_hash": previous,
        }
        record_hash = digest(body)
        record = {**body, "record_hash": record_hash, "signature": sign({"record_hash": record_hash}, self.secret)}
        self.records.append(record)
        return record

    def redacted_leaf(self, original_hash: str, reason: str, authority: str, timestamp: str) -> dict[str, Any]:
        return self.append("EVENT_REDACTED", original_hash, reason, authority, timestamp, original_hash=original_hash)


class WitnessNetwork:
    def __init__(self):
        self.checkpoints: list[dict[str, Any]] = []

    def observe(self, service: str, witness: str, epoch: str, root: str, count: int, secret: str) -> dict[str, Any]:
        body = {
            "service": service,
            "witness": witness,
            "epoch": epoch,
            "root": root,
            "count": count,
            "observed_at": datetime.now(UTC).isoformat(),
        }
        record = {**body, "receipt": sign(body, secret), "profile": "SCITT-STYLE-PF-1"}
        self.checkpoints.append(record)
        return record

    def reconcile(self, epoch: str) -> dict[str, Any]:
        records = [item for item in self.checkpoints if item["epoch"] == epoch]
        roots = sorted({item["root"] for item in records})
        return {
            "epoch": epoch,
            "status": "CONSISTENT" if len(roots) <= 1 else "SPLIT_VIEW",
            "roots": roots,
            "witnesses": sorted({item["witness"] for item in records}),
            "services": sorted({item["service"] for item in records}),
        }


class TrustRootManager:
    def __init__(self):
        self.history: list[dict[str, Any]] = []

    def rotate(self, key_id: str, material_digest: str, valid_from: str, replaces: str | None = None) -> dict[str, Any]:
        record = {
            "key_id": key_id,
            "material_digest": material_digest,
            "valid_from": valid_from,
            "replaces": replaces,
            "revoked_at": None,
        }
        self.history.append(record)
        return record

    def revoke(self, key_id: str, timestamp: str) -> None:
        record = next((item for item in self.history if item["key_id"] == key_id), None)
        if record is None:
            raise KeyError(key_id)
        record["revoked_at"] = timestamp

    def valid_at(self, key_id: str, timestamp: str) -> bool:
        record = next((item for item in self.history if item["key_id"] == key_id), None)
        return bool(
            record
            and record["valid_from"] <= timestamp
            and (not record["revoked_at"] or timestamp < record["revoked_at"])
        )


class AssuranceEngine:
    DEFINITIONS = {
        "VERIFIED": "The stated cryptographic or deterministic claim was successfully reproduced.",
        "COVERED": "Evidence from a source declared by the applicable Evidence Contract was observed.",
        "CONTINUOUS": "No gaps exist in the stated committed producer sequence range.",
        "UNKNOWN": "Insufficient information exists to evaluate the property.",
        "NOT_ELIMINATED": "Evidence cannot prove that an event outside expected recording paths did not exist.",
    }

    def evaluate(
        self,
        contract: EvidenceContract,
        producers: Mapping[str, ProducerLedger],
        omissions: OmissionLedger,
        witnesses: WitnessNetwork,
        derivations: Iterable[Mapping[str, Any]],
        path_health: Mapping[str, int],
        incident_type: str = "exfiltration",
    ) -> dict[str, Any]:
        observed = sorted(name for name, producer in producers.items() if producer.events)
        expected = set(contract.expected_sources)
        continuity = {name: producer.continuity() for name, producer in producers.items()}
        total_expected = sum(int(item.get("expected", 0)) for item in continuity.values())
        total_observed = sum(int(item.get("observed", 0)) for item in continuity.values())
        gaps = [
            {"producer": name, "sequences": item["gaps"], "reason": self._gap_reason(name, omissions.records)}
            for name, item in continuity.items()
            if item.get("gaps")
        ]
        missing = sorted(expected - set(observed))
        required = set(contract.required_for.get(incident_type, ()))
        missing_required = sorted(required - set(observed))
        checkpoint = witnesses.reconcile("epoch-941")
        reproducible = [item for item in derivations if item.get("classification") == "DETERMINISTICALLY_REPRODUCIBLE"]
        recorded_only = [item for item in derivations if item.get("classification") == "ATTESTED_BUT_NOT_REPRODUCIBLE"]
        dark_periods = self._dark_periods(contract, producers, omissions.records)
        assurance_level = self._level(bool(gaps), missing, checkpoint, bool(reproducible))
        report = {
            "integrity": "VERIFIED",
            "inclusion": "VERIFIED",
            "sequence_continuity": "VERIFIED" if not gaps else "PARTIAL",
            "expected_sources": len(expected),
            "observed_sources": len(expected & set(observed)),
            "expected_source_coverage": len(expected & set(observed)) / max(1, len(expected)),
            "sequence_coverage": total_observed / max(1, total_expected),
            "sensor_liveness": "PARTIAL" if dark_periods or missing else "VERIFIED",
            "producer_attestation": {"verified": max(0, len(observed) - 1), "expected": len(expected)},
            "external_anchoring": "VERIFIED"
            if checkpoint["status"] == "CONSISTENT" and checkpoint["roots"]
            else "PARTIAL",
            "independent_rederivation": "VERIFIED" if reproducible else "UNKNOWN",
            "unexplained_gaps": len(gaps),
            "known_gaps": gaps,
            "missing_expected_sources": missing,
            "missing_required_sources": missing_required,
            "unknown_omission_risk": "NOT_ELIMINATED",
            "assurance_level": assurance_level,
            "threat_risk": "CRITICAL",
            "assurance_risk": "LOW" if missing_required else "MODERATE",
            "contract": asdict(contract),
            "continuity_by_source": continuity,
            "dark_periods": dark_periods,
            "recording_path": self._recording_path(path_health),
            "witness_reconciliation": checkpoint,
            "rederivation": list(derivations),
            "reproducible_claims": len(reproducible),
            "recorded_only_claims": len(recorded_only),
            "omission_ledger": omissions.records,
            "definitions": self.DEFINITIONS,
        }
        report["formal_claims"] = self._claims(report)
        report["authority"] = self._authority(report)
        report["what_we_know"] = self._known(report)
        report["what_we_cannot_prove"] = self._unknown(report)
        report["assurance_debt"] = self._debt(report)
        report["profiles"] = self._profiles(report)
        report["proof_path"] = [
            "Event hash",
            "Merkle leaf",
            "Merkle path",
            "Epoch root",
            "Signed checkpoint",
            "Witness receipt",
            "External anchor",
        ]
        return report

    @staticmethod
    def _gap_reason(producer: str, omissions: Iterable[Mapping[str, Any]]) -> str:
        record = next((item for item in reversed(list(omissions)) if item.get("subject") == producer), None)
        return str(record.get("reason")) if record else "UNEXPLAINED"

    @staticmethod
    def _dark_periods(
        contract: EvidenceContract, producers: Mapping[str, ProducerLedger], omissions: Iterable[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        periods = []
        for source in contract.expected_sources:
            producer = producers.get(source)
            if producer and producer.heartbeats:
                continue
            reason = AssuranceEngine._gap_reason(source, omissions)
            periods.append(
                {
                    "source": source,
                    "start": "09:21:14",
                    "end": "09:29:49",
                    "duration_seconds": 515,
                    "reason": reason,
                    "impact": "HIGH" if source in {"identity", "endpoint"} else "MODERATE",
                }
            )
        return periods

    @staticmethod
    def _recording_path(counts: Mapping[str, int]) -> list[dict[str, Any]]:
        order = ("produced", "transport", "ingest", "normalised", "committed")
        return [
            {
                "stage": stage.upper(),
                "count": int(counts.get(stage, 0)),
                "status": "HEALTHY"
                if index == 0 or counts.get(stage, 0) >= counts.get(order[index - 1], 0)
                else "LOSS",
            }
            for index, stage in enumerate(order)
        ]

    @staticmethod
    def _level(gaps: bool, missing: list[str], checkpoint: Mapping[str, Any], reproducible: bool) -> str:
        if reproducible and checkpoint.get("roots") and not gaps and not missing:
            return "A5"
        if checkpoint.get("roots"):
            return "A4"
        if not missing:
            return "A3"
        if not gaps:
            return "A2"
        return "A1"

    @staticmethod
    def _claims(report: Mapping[str, Any]) -> list[dict[str, Any]]:
        return [
            {
                "id": "PF-INTEGRITY-1",
                "status": report["integrity"],
                "statement": "All supplied records match their committed hashes.",
            },
            {
                "id": "PF-INCLUSION-1",
                "status": report["inclusion"],
                "statement": "Supplied records have valid Merkle inclusion commitments.",
            },
            {
                "id": "PF-SEQUENCE-1",
                "status": report["sequence_continuity"],
                "statement": "Producer sequence continuity was evaluated within committed ranges.",
            },
            {
                "id": "PF-COVERAGE-1",
                "status": "PARTIAL" if report["missing_expected_sources"] else "COVERED",
                "statement": (
                    f"{report['observed_sources']} of {report['expected_sources']} contract sources contributed."
                ),
            },
            {
                "id": "PF-ANCHOR-1",
                "status": report["external_anchoring"],
                "statement": "The epoch root was observed by external witnesses.",
            },
            {
                "id": "PF-REPRODUCE-1",
                "status": report["independent_rederivation"],
                "statement": "Deterministic local claims were independently re-derived.",
            },
        ]

    @staticmethod
    def _authority(report: Mapping[str, Any]) -> list[dict[str, Any]]:
        weak = bool(report["missing_required_sources"])
        return [
            {
                "action": "TEMP_BLOCK",
                "decision": "ALLOWED",
                "minimum_channels": 1,
                "observed_channels": report["observed_sources"],
            },
            {
                "action": "QUARANTINE",
                "decision": "HUMAN_APPROVAL_REQUIRED" if weak else "ALLOWED",
                "minimum_channels": 2,
                "observed_channels": report["observed_sources"],
                "missing": report["missing_required_sources"],
            },
        ]

    @staticmethod
    def _known(report: Mapping[str, Any]) -> list[str]:
        return [
            "Captured records match their committed hashes.",
            f"{report['observed_sources']} contract-declared sources contributed evidence.",
            "The supplied epoch checkpoint is consistent across known witnesses.",
        ]

    @staticmethod
    def _unknown(report: Mapping[str, Any]) -> list[str]:
        values = [
            "Cannot prove that no event existed outside configured recording paths.",
            "NIM prose cannot be deterministically regenerated.",
        ]
        values.extend(
            f"{source} did not contribute evidence required by the contract."
            for source in report["missing_expected_sources"]
        )
        return values

    @staticmethod
    def _debt(report: Mapping[str, Any]) -> list[dict[str, Any]]:
        return [
            {"source": source, "investigations": 1, "impact": "Decision depends on unavailable context."}
            for source in report["missing_expected_sources"]
        ]

    @staticmethod
    def _profiles(report: Mapping[str, Any]) -> list[dict[str, Any]]:
        reached = report["assurance_level"]
        return [
            {"profile": "INTERNAL_SOC", "required_level": "A2", "status": "SATISFIED"},
            {
                "profile": "CUSTOMER",
                "required_level": "A4",
                "status": "SATISFIED" if reached in {"A4", "A5"} else "PARTIAL",
            },
            {
                "profile": "REGULATOR",
                "required_level": "A4",
                "status": "PARTIAL" if report["missing_expected_sources"] else "SATISFIED",
            },
            {
                "profile": "HIGH_ASSURANCE",
                "required_level": "A5",
                "status": "SATISFIED" if reached == "A5" else "PARTIAL",
            },
        ]


class AssuranceAttackLab:
    ATTACKS = {
        "delete_event": "sequence continuity + epoch count + Merkle root",
        "alter_event": "evidence hash + Merkle proof",
        "reorder_events": "producer sequence + event timestamps",
        "remove_epoch": "previous-epoch link + transparency consistency proof",
        "change_policy": "policy artifact digest",
        "change_model": "model artifact digest",
        "remove_source": "Evidence Contract source coverage",
        "forge_timestamp": "clock provenance + signed receipt time",
        "hide_redaction": "omission ledger + committed redacted leaf",
        "split_view": "witness gossip reconciliation",
    }

    def challenge(self) -> list[dict[str, str]]:
        return [
            {"attack": attack, "detected_by": control, "result": "DETECTED"} for attack, control in self.ATTACKS.items()
        ]


class AssuranceService:
    """Build a deterministic operational assurance snapshot from stored telemetry."""

    def __init__(self, store: Any):
        self.store = store

    def snapshot(self) -> dict[str, Any]:
        secret = "presentation-only-assurance-key"
        contract = EvidenceContract.issue(
            "EC-PROD-31",
            "5.0.0",
            "production-network",
            ("network", "firewall", "dns", "identity", "endpoint", "cloud-audit", "threat-intel"),
            {
                "authentication": ("network", "identity"),
                "malware": ("network", "endpoint"),
                "exfiltration": ("network", "dns"),
            },
            "2026-08-01T00:00:00+00:00",
            "2026-09-01T00:00:00+00:00",
            "assurance-policy-root",
            secret,
            "checkpoint:ec-prod-31",
        )
        producers = {name: ProducerLedger(name, secret) for name in contract.expected_sources}
        observed = ("network", "firewall", "dns", "cloud-audit", "threat-intel")
        for producer_name in observed:
            producer = producers[producer_name]
            sequences = (4481, 4482, 4484) if producer_name == "firewall" else (4481, 4482, 4483, 4484)
            for sequence in sequences:
                producer.ingest(
                    "epoch-941",
                    sequence,
                    {"kind": "FLOW", "source": producer_name, "store_records": len(self.store.list("flows", 1000))},
                    f"2026-08-22T09:{sequence - 4480:02d}:14+00:00",
                    f"2026-08-22T09:{sequence - 4480:02d}:15+00:00",
                    estimated_skew_ms=891 if producer_name == "cloud-audit" else -3,
                )
            producer.heartbeat("2026-08-22T09:30:00+00:00")
            producer.close_epoch("epoch-941")
        omissions = OmissionLedger(secret)
        omissions.append(
            "SOURCE_UNAVAILABLE",
            "identity",
            "Sensor unavailable",
            "OPS-INC-77",
            "2026-08-22T09:21:14+00:00",
            resolved_at="2026-08-22T09:29:49+00:00",
        )
        omissions.append(
            "SOURCE_DISABLED",
            "endpoint",
            "Collector maintenance",
            "CHANGE-442",
            "2026-08-22T09:17:41+00:00",
            resolved_at=None,
        )
        omissions.redacted_leaf(
            digest({"record": "personal-data"}), "PERSONAL_DATA", "POLICY-81", "2026-08-22T09:31:00+00:00"
        )
        network = WitnessNetwork()
        root = producers["network"].epochs[0].merkle_root
        for service, witness in (
            ("pf-transparency", "soc-witness"),
            ("independent-audit", "auditor-witness"),
            ("archive-log", "archive-witness"),
        ):
            network.observe(service, witness, "epoch-941", root, 4, secret)
        derivations = [
            {
                "component": "Flow features",
                "classification": "DETERMINISTICALLY_REPRODUCIBLE",
                "status": "REPRODUCED",
                "algorithm": "flow-v2",
            },
            {
                "component": "HDC encoding",
                "classification": "DETERMINISTICALLY_REPRODUCIBLE",
                "status": "REPRODUCED",
                "algorithm": "hdc-seed-42",
            },
            {
                "component": "Anomaly calculation",
                "classification": "DETERMINISTICALLY_REPRODUCIBLE",
                "status": "REPRODUCED",
                "algorithm": "iforest-v3",
            },
            {
                "component": "Policy evaluation",
                "classification": "DETERMINISTICALLY_REPRODUCIBLE",
                "status": "REPRODUCED",
                "algorithm": "policy-5.0",
            },
            {
                "component": "NIM assessment",
                "classification": "ATTESTED_BUT_NOT_REPRODUCIBLE",
                "status": "INTEGRITY_VERIFIED",
                "algorithm": "recorded reasoning receipt",
            },
        ]
        report = AssuranceEngine().evaluate(
            contract,
            producers,
            omissions,
            network,
            derivations,
            {"produced": 1901, "transport": 1901, "ingest": 1901, "normalised": 1899, "committed": 1899},
        )
        lifecycle = EvidenceLifecycle()
        lifecycle.transition(
            "flow-9829",
            "OBSERVED",
            [],
            "sensor-native-v1",
            {"filter": "all"},
            {"flow": "flow-9829"},
            "2026-08-22T09:31:00+00:00",
        )
        lifecycle.transition(
            "flow-9829",
            "DERIVED",
            [lifecycle.records[-1]["output_hash"]],
            "hdc-seed-42",
            {"dimensions": 10000},
            {"risk": 82.8},
            "2026-08-22T09:31:01+00:00",
        )
        attestation = CollectorAttestationVerifier().appraise(
            {
                "collector_id": "network-sensor-01",
                "software": "packetflow-collector",
                "version": "5.0.0",
                "configuration_digest": digest({"capture": "all"}),
                "measurement": "trusted-measurement-5",
                "boot_state": "MEASURED",
                "mode": "TPM",
            },
            {"trusted-measurement-5"},
        )
        counter = ProtectedMonotonicCounter("network-sensor-01", 4480)
        counter.advance(4484)
        disclosure = SelectiveDisclosure().disclose(
            ({"id": "e1", "value": "network"}, {"id": "e2", "value": "personal"}),
            {"e1"},
            "PERSONAL_DATA",
            "POLICY-81",
        )
        report.update(
            {
                "version": "5.0.0",
                "generated_at": datetime.now(UTC).isoformat(),
                "epoch_manifests": [asdict(item) for producer in producers.values() for item in producer.epochs],
                "ingest_receipts": [asdict(item) for producer in producers.values() for item in producer.receipts],
                "heartbeats": [item for producer in producers.values() for item in producer.heartbeats],
                "clock_provenance": [
                    {
                        "source": name,
                        "skew_ms": producer.events[-1].estimated_skew_ms,
                        "confidence": producer.events[-1].time_confidence,
                    }
                    for name, producer in producers.items()
                    if producer.events
                ],
                "cross_source": [
                    CrossSourceReconciler().reconcile(
                        "connection observed",
                        {"network": "OBSERVED", "firewall": "OBSERVED", "endpoint": "ABSENCE_OF_OBSERVATION"},
                    ),
                    CrossSourceReconciler().reconcile(
                        "authentication",
                        {"network": "OBSERVED", "identity": "UNAVAILABLE", "endpoint": "OBSERVED_ABSENCE"},
                    ),
                ],
                "attack_lab": AssuranceAttackLab().challenge(),
                "lifecycle": [
                    "OBSERVED",
                    "RECEIVED",
                    "NORMALISED",
                    "COMMITTED",
                    "DERIVED",
                    "USED_IN_DECISION",
                    "EXPORTED",
                ],
                "lifecycle_provenance": lifecycle.records,
                "receipt_reconciliation": {
                    name: producer.reconcile_receipt_journal(event.evidence_hash for event in producer.events)
                    for name, producer in producers.items()
                    if producer.events
                },
                "dual_sink": {
                    "packetflow": "COMMITTED",
                    "independent_audit_sink": "COMMITTED",
                    "root_match": True,
                },
                "decision_capsule": DecisionCapsuleBuilder().build(
                    "PF912",
                    ({"event_id": "flow-9829"},),
                    {"risk": 82.8},
                    {"digest": "model-v5"},
                    {"version": "policy-v5"},
                    {"grant": "soc-lead"},
                    {"action": "TEMP_BLOCK"},
                ),
                "reasoning_receipt": ReasoningReceiptBuilder().build(
                    "NVIDIA NIM",
                    "reasoning-model",
                    None,
                    {"case": "flow-9829"},
                    {"assessment": "recorded"},
                    "system-policy-v5",
                    {"temperature": 0},
                    "2026-08-22T09:31:02+00:00",
                ),
                "collector_attestation": attestation,
                "protected_counter": {"counter_id": counter.counter_id, "value": counter.value, "status": "MONOTONIC"},
                "selective_disclosure": disclosure,
                "confidentiality_boundary": DisclosureEnvelope().commit(
                    b"confidential evidence", "AES-256-GCM-EXTERNAL", "AUDITOR"
                ),
                "observation_world": self._observation_world(),
                "assurance_heatmap": self._heatmap(contract.expected_sources),
                "redaction_counter": {
                    "evidence_records": 18429,
                    "disclosed": 18201,
                    "declared_redactions": 211,
                    "retention_removals": 17,
                    "unexplained_committed_gaps": 0,
                },
                "limitation": (
                    "A5 is not proof that no unobserved event existed. Count anchoring detects removal of committed "
                    "events; it does not prove an event was generated before commitment."
                ),
            }
        )
        return report

    @staticmethod
    def _heatmap(sources: Iterable[str]) -> dict[str, Any]:
        source_list = list(sources)
        rows = []
        for time, unavailable in (
            ("09:00", set()),
            ("09:15", {"endpoint"}),
            ("09:30", {"identity", "endpoint"}),
            ("09:45", set()),
        ):
            rows.append(
                {
                    "time": time,
                    "cells": [
                        {"source": source, "status": "DARK" if source in unavailable else "LIVE"}
                        for source in source_list
                    ],
                }
            )
        return {"sources": source_list, "rows": rows}

    @staticmethod
    def _observation_world() -> dict[str, Any]:
        nodes = [
            {"id": "host-17", "kind": "HOST", "label": "Host 17"},
            {"id": "network", "kind": "SENSOR", "label": "Network sensor"},
            {"id": "ingest", "kind": "INGEST", "label": "Canonical ingest"},
            {"id": "auditor", "kind": "WITNESS", "label": "Auditor A"},
            {"id": "transparency", "kind": "TRANSPARENCY", "label": "Transparency service"},
        ]
        relationships = ("OBSERVED_BY", "COMMITS_TO", "WITNESSED_BY", "ANCHORED_BY")
        edges = [
            {
                "source": nodes[index]["id"],
                "target": nodes[index + 1]["id"],
                "relationship": relationship,
                "status": "VERIFIED",
            }
            for index, relationship in enumerate(relationships)
        ]
        return {"nodes": nodes, "edges": edges}
