"""PacketFlowAI v5 verifiable-assurance tests."""

from __future__ import annotations

import json
import tempfile
import time
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from threading import Thread
from urllib.request import urlopen

from packetflow_verifier import BundleVerifier
from packetflowai.api import APIServer
from packetflowai.assurance import (
    AssuranceEngine,
    AssuranceService,
    CollectorAttestationVerifier,
    CrossSourceReconciler,
    DecisionCapsuleBuilder,
    DisclosureEnvelope,
    EvidenceContract,
    EvidenceLifecycle,
    OmissionLedger,
    ProducerLedger,
    ProtectedMonotonicCounter,
    ReasoningReceiptBuilder,
    SelectiveDisclosure,
    TrustRootManager,
    WitnessNetwork,
    digest,
)
from packetflowai.config import AppConfig, ArtifactPaths
from packetflowai.forensics import EvidenceBundleExporter, FileTransparencyLog
from packetflowai.governance import EvidenceLedger
from packetflowai.registry import FilesystemModelRegistry
from packetflowai.storage import EventStore
from packetflowai.telemetry import MetricsRegistry


class ProducerAssuranceTests(unittest.TestCase):
    def test_sequences_receipts_epoch_and_clock_provenance(self):
        producer = ProducerLedger("firewall", "secret")
        first = producer.ingest("epoch-1", 194, {"event": "a"}, "2026-08-22T09:00:00+00:00")
        producer.ingest(
            "epoch-1",
            196,
            {"event": "b"},
            "2026-08-22T09:00:02+00:00",
            estimated_skew_ms=891,
        )
        epoch = producer.close_epoch("epoch-1")
        self.assertEqual(first.producer_id, "firewall")
        self.assertTrue(first.signature)
        self.assertEqual(producer.continuity()["gaps"], [195])
        self.assertEqual(epoch.event_count, 2)
        self.assertEqual(producer.events[-1].time_confidence, "LOW")

    def test_omissions_redactions_trust_history_and_split_view(self):
        ledger = OmissionLedger("secret")
        redaction = ledger.redacted_leaf(
            digest({"sensitive": True}), "PERSONAL_DATA", "POLICY-81", "2026-08-22T09:00:00Z"
        )
        self.assertEqual(redaction["kind"], "EVENT_REDACTED")
        self.assertTrue(redaction["signature"])
        trust = TrustRootManager()
        trust.rotate("key-1", digest(b"one"), "2026-08-01T00:00:00Z")
        trust.rotate("key-2", digest(b"two"), "2026-08-20T00:00:00Z", "key-1")
        trust.revoke("key-1", "2026-08-21T00:00:00Z")
        self.assertTrue(trust.valid_at("key-1", "2026-08-10T00:00:00Z"))
        self.assertFalse(trust.valid_at("key-1", "2026-08-22T00:00:00Z"))
        witnesses = WitnessNetwork()
        witnesses.observe("log-a", "auditor-a", "epoch-1", "a" * 64, 2, "one")
        witnesses.observe("log-b", "auditor-b", "epoch-1", "b" * 64, 2, "two")
        self.assertEqual(witnesses.reconcile("epoch-1")["status"], "SPLIT_VIEW")

    def test_provenance_attestation_capsules_and_selective_disclosure(self):
        lifecycle = EvidenceLifecycle()
        transition = lifecycle.transition("e1", "DERIVED", ["a" * 64], "hdc-v1", {"seed": 42}, {"risk": 70}, "now")
        self.assertTrue(transition["provenance_hash"])
        capsule = DecisionCapsuleBuilder().build("d1", ({"id": "e1"},), {}, {}, {}, {}, {"action": "ALERT"})
        self.assertEqual(capsule["format"], "PFCAP-1.0")
        receipt = ReasoningReceiptBuilder().build("nim", "model", None, {}, {}, "system", {}, "now")
        self.assertEqual(receipt["classification"], "ATTESTED_BUT_NOT_REPRODUCIBLE")
        appraisal = CollectorAttestationVerifier().appraise(
            {
                "collector_id": "c1",
                "software": "collector",
                "version": "1",
                "configuration_digest": "a",
                "measurement": "trusted",
                "boot_state": "measured",
            },
            {"trusted"},
        )
        self.assertEqual(appraisal["status"], "VERIFIED")
        counter = ProtectedMonotonicCounter("c1")
        counter.advance(2)
        with self.assertRaises(ValueError):
            counter.advance(1)
        reconciliation = CrossSourceReconciler().reconcile(
            "connection", {"network": "OBSERVED", "endpoint": "ABSENCE_OF_OBSERVATION"}
        )
        self.assertEqual(reconciliation["status"], "EVIDENCE_ASYMMETRY")
        disclosure = SelectiveDisclosure().disclose(({"id": "one"}, {"id": "two"}), {"one"}, "PRIVATE", "P1")
        self.assertEqual(disclosure["counts"]["redacted"], 1)
        envelope = DisclosureEnvelope().commit(b"secret", "EXTERNAL", "AUDITOR")
        self.assertFalse(envelope["verification_requires_plaintext"])


class AssuranceVectorTests(unittest.TestCase):
    def test_snapshot_separates_coverage_from_unknown_omission_risk(self):
        with tempfile.TemporaryDirectory() as directory:
            store = EventStore(Path(directory) / "events.db")
            result = AssuranceService(store).snapshot()
            store.close()
        self.assertEqual(result["version"], "5.0.0")
        self.assertEqual(result["integrity"], "VERIFIED")
        self.assertEqual(result["sequence_continuity"], "PARTIAL")
        self.assertEqual(result["expected_sources"], 7)
        self.assertEqual(result["observed_sources"], 5)
        self.assertEqual(result["unknown_omission_risk"], "NOT_ELIMINATED")
        self.assertNotIn("complete", result)
        self.assertIn("identity", result["missing_expected_sources"])
        self.assertEqual(result["witness_reconciliation"]["status"], "CONSISTENT")
        self.assertEqual(result["authority"][1]["decision"], "ALLOWED")

    def test_required_source_weakens_authority(self):
        contract = EvidenceContract.issue(
            "EC-1",
            "1",
            "prod",
            ("network", "endpoint"),
            {"malware": ("network", "endpoint")},
            "2026-08-01T00:00:00Z",
            "2026-09-01T00:00:00Z",
            "root",
            "secret",
            "anchor",
        )
        producer = ProducerLedger("network", "secret")
        producer.ingest("epoch-1", 1, {}, "2026-08-22T09:00:00Z")
        witnesses = WitnessNetwork()
        omissions = OmissionLedger("secret")
        result = AssuranceEngine().evaluate(
            contract,
            {"network": producer},
            omissions,
            witnesses,
            [],
            {"produced": 1, "transport": 1, "ingest": 1, "normalised": 1, "committed": 1},
            "malware",
        )
        self.assertEqual(result["authority"][1]["decision"], "HUMAN_APPROVAL_REQUIRED")
        self.assertEqual(result["missing_required_sources"], ["endpoint"])


class PFCaseV5Tests(unittest.TestCase):
    def _bundle(self, root: Path) -> tuple[Path, dict[str, str]]:
        secret = "case-secret"
        producer = ProducerLedger("network", secret)
        producer.ingest("epoch-941", 1, {"flow": "f1"}, "2026-08-22T09:00:00+00:00")
        producer.ingest("epoch-941", 2, {"flow": "f2"}, "2026-08-22T09:00:01+00:00")
        epoch = producer.close_epoch("epoch-941")
        evidence_contract = asdict(
            EvidenceContract.issue(
                "EC-PROD-31",
                "5.0.0",
                "production",
                ("network", "identity"),
                {"authentication": ("network", "identity")},
                "2026-08-01T00:00:00+00:00",
                "2026-09-01T00:00:00+00:00",
                "root",
                secret,
                "checkpoint",
            )
        )
        ledger = EvidenceLedger()
        ledger.append("f1", "2026-08-22T09:00:00+00:00", {}, "model", "policy", "soc", "alert")
        payload = {"risk": 82, "action": "TEMP_BLOCK"}
        replay_input = {"features": [1, 2, 3]}
        decision = {
            "decision_id": "d1",
            "payload": payload,
            "replay": {
                "input": replay_input,
                "input_digest": digest(replay_input),
                "output_digest": digest(payload),
            },
            "reproducibility": {"classification": "DETERMINISTICALLY_REPRODUCIBLE"},
        }
        events = [asdict(event) for event in producer.events]
        assurance = {
            "formal_claims": [{"id": "PF-INTEGRITY-1", "status": "VERIFIED"}],
            "unknown_omission_risk": "NOT_ELIMINATED",
        }
        bundle = root / "incident.pfcase"
        EvidenceBundleExporter().export(
            bundle,
            "PF-2026-00192",
            {
                "events": events,
                "observations": events,
                "sources": [{"producer_id": "network"}],
                "epochs": [asdict(epoch)],
                "decisions": [decision],
                "omissions": [{"kind": "SOURCE_UNAVAILABLE", "subject": "identity"}],
                "redactions": [{"original_hash": "a" * 64, "reason": "PERSONAL_DATA"}],
            },
            ledger,
            {"model_digest": "model-1", "classification": "DETERMINISTICALLY_REPRODUCIBLE"},
            {"soc": "witness-secret"},
            FileTransparencyLog(root / "transparency.jsonl"),
            assurance,
            evidence_contract,
            {"keys": [{"key_id": "case-key"}]},
            ("case-key", secret),
        )
        return bundle, {"case-key": secret}

    def test_pfcase_schema_signature_proof_replay_and_challenge(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle, signer_keys = self._bundle(Path(directory))
            verifier = BundleVerifier()
            result = verifier.verify(bundle, {"soc": "witness-secret"}, signer_keys)
            proof = verifier.inclusion_proof(bundle, "evidence/events.jsonl")
            autopsy = verifier.decision_autopsy(bundle, "d1")
            challenge = verifier.challenge(bundle)
            contract = verifier.audit_resource(bundle, "evidence-contract")
        self.assertTrue(result["verified"])
        self.assertEqual(result["schema"]["profile"], "PFCASE-1.0")
        self.assertEqual(result["manifest_signature"]["status"], "VALID")
        self.assertEqual(result["sequence_continuity"]["status"], "VERIFIED")
        self.assertEqual(result["expected_source_coverage"]["status"], "PARTIAL")
        self.assertTrue(proof["verified"])
        self.assertEqual(autopsy["feature_derivation"], "REPRODUCED")
        self.assertEqual(len(challenge["challenges"]), 10)
        self.assertEqual(contract["contract_id"], "EC-PROD-31")


class V5APITests(unittest.TestCase):
    def test_assurance_and_public_audit_resources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = replace(AppConfig(), artifacts=ArtifactPaths(root / "artifacts"))
            config.artifacts.create()
            store = EventStore(config.artifacts.event_database)
            server = APIServer(
                config, store, MetricsRegistry(), FilesystemModelRegistry(config.artifacts.registry), port=0
            )
            thread = Thread(target=server.serve_forever, daemon=True)
            thread.start()
            for _ in range(100):
                if server.server:
                    break
                time.sleep(0.01)
            assert server.server
            base = f"http://127.0.0.1:{server.server.server_address[1]}"
            health = json.loads(urlopen(f"{base}/health", timeout=5).read())
            assurance = json.loads(urlopen(f"{base}/v5/assurance", timeout=5).read())
            schema = json.loads(urlopen(f"{base}/audit/v1/schema", timeout=5).read())
            server.stop()
            thread.join(2)
            store.close()
        self.assertEqual(health["version"], "5.0.0")
        self.assertEqual(assurance["unknown_omission_risk"], "NOT_ELIMINATED")
        self.assertEqual(schema["identifier"], "https://packetflow.ai/spec/pfcase/1.0")


if __name__ == "__main__":
    unittest.main()
