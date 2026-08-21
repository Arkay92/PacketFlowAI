"""Cross-domain tests for PacketFlowAI's next-iteration platform."""

import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

from packetflow_verifier import BundleVerifier
from packetflowai.advanced_intelligence import (
    AuthorityGraphV2,
    DigitalTwinV2,
    EvidenceTimeMachineV2,
    InterventionSolver,
    PredictionEngineV2,
    TwinAsset,
)
from packetflowai.domain import FlowFeatures
from packetflowai.forensics import EvidenceBundleExporter, EvidenceDiffer, FileTransparencyLog
from packetflowai.governance import EvidenceLedger
from packetflowai.interop import (
    ApplicationAdapter,
    CloudAuditAdapter,
    DetectionRepository,
    DNSAdapter,
    EDRAdapter,
    IdentityAdapter,
    OCSFMapper,
    SIEMExporter,
    SigmaRuleEngine,
    STIX21Exchange,
)
from packetflowai.platform import PlatformIntelligenceService
from packetflowai.platform_engines import (
    AdaptiveRuntime,
    AttackLaboratoryV2,
    BinaryHDC,
    DeceptionEngine,
    ExplainabilityEngine,
    HyperdimensionalWorldMemory,
    RobustFederation,
    RobustnessLab,
    ThreatMemoryV2,
)
from packetflowai.storage import EventStore
from packetflowai.v3_support import FederatedArtifact
from packetflowai.world import Campaign, CyberWorldModel


class IndependentForensicsTests(unittest.TestCase):
    def test_bundle_verifies_independently_and_detects_tampering(self):
        ledger = EvidenceLedger()
        ledger.append("e1", "2026-08-21T10:00:00+00:00", {"risk": 70}, "model", "policy", "soc", "alert")
        replay_input = {"features": [1, 2]}
        decision = {"decision_id": "d1", "payload": {"risk": 70}}
        from packetflow_verifier.core import canonical, digest

        decision["replay"] = {
            "input": replay_input,
            "input_digest": digest(canonical(replay_input)),
            "output_digest": digest(canonical(decision["payload"])),
        }
        decision["reproducibility"] = {"model_digest": "abc", "feature_schema": "flow-v2"}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root / "case.pfcase"
            anchor = FileTransparencyLog(root / "transparency.jsonl")
            EvidenceBundleExporter().export(
                bundle,
                "PF-CASE-1",
                {"events": [{"event_id": "e1"}], "decisions": [decision]},
                ledger,
                {"model_digest": "abc"},
                {"soc": "secret"},
                anchor,
            )
            result = BundleVerifier().verify(bundle, {"soc": "secret"})
            replay = BundleVerifier().replay_decision(bundle, "d1")
            tampered = root / "tampered.pfcase"
            with zipfile.ZipFile(bundle) as source, zipfile.ZipFile(tampered, "w") as target:
                for name in source.namelist():
                    value = b"changed" if name == "events.jsonl" else source.read(name)
                    target.writestr(name, value)
            failed = BundleVerifier().verify(tampered, {"soc": "secret"})
        self.assertTrue(result["verified"])
        self.assertEqual(result["witnesses"]["count"], 1)
        self.assertTrue(replay["reproducible"])
        self.assertFalse(failed["verified"])

    def test_evidence_diff(self):
        result = EvidenceDiffer().compare([{"id": "a", "v": 1}, {"id": "b"}], [{"id": "a", "v": 2}, {"id": "c"}], "id")
        self.assertEqual(result, {"missing": ["b"], "unexpected": ["c"], "modified": ["a"]})


class IntelligenceV2Tests(unittest.TestCase):
    def test_horizons_calibration_and_belief_pruning(self):
        campaign = Campaign("c", "Campaign", ("e1",), (), ("T1110",), ("h1",), ("s1",), "a", "b", 0.8, "x")
        engine = PredictionEngineV2()
        prediction = engine.predict(campaign, 0.6)
        self.assertEqual(set(prediction["horizons"]), {"5m", "1h", "24h"})
        self.assertAlmostEqual(sum(item["probability"] for item in prediction["horizons"]["1h"]), 1)
        self.assertLess(
            engine.calibration([{"probability": 0.7, "occurred": True}])["expected_calibration_error"], 0.31
        )
        self.assertTrue(engine.belief_change({"lateral": 0.6}, {"lateral": 0.1})[0]["pruned"])

    def test_twin_paths_what_if_and_authority(self):
        twin = DigitalTwinV2()
        for name in ("workstation", "api", "database"):
            twin.add_asset(TwinAsset(name, "HOST", 0.8, 0.7, 0.6))
        twin.connect("workstation", "api", "CAN_CONNECT_TO")
        twin.connect("api", "database", "DEPENDS_ON")
        self.assertEqual(twin.paths("workstation", "database")[0], ["workstation", "api", "database"])
        self.assertEqual(twin.what_if({"database"})["dependencies_lost"], ["api"])
        authority = AuthorityGraphV2()
        authority.grant("soc", "lead", "QUARANTINE", "host-a", 300, ["MAY_APPROVE"])
        self.assertFalse(authority.authorize("QUARANTINE", "host-a", [{"subject": "one"}])["permitted"])
        self.assertTrue(
            authority.authorize("QUARANTINE", "host-a", [{"subject": "one"}, {"subject": "two"}])["permitted"]
        )
        self.assertTrue(authority.break_glass("commander", "active ransomware")["relationships"])

    def test_time_machine_has_strict_knowledge_boundary(self):
        state = {
            "events": [
                {"id": "old", "created_at": "2026-08-21T10:00:00+00:00"},
                {"id": "future", "created_at": "2026-08-21T11:00:00+00:00"},
            ]
        }
        replay = EvidenceTimeMachineV2().replay("2026-08-21T10:30:00+00:00", state)
        self.assertEqual(replay["known_then"]["events"][0]["id"], "old")
        self.assertEqual(replay["learned_later"]["events"][0]["id"], "future")
        self.assertFalse(replay["hindsight_leakage"])

    def test_intervention_solver(self):
        model = CyberWorldModel()
        model.add_node("SOURCE", "10.0.0.1")
        result = InterventionSolver().solve(model, "10.0.0.1", 80, 35)
        self.assertIsNotNone(result["minimum_intervention"])


class InteroperabilityTests(unittest.TestCase):
    def test_sensor_fabric_ocsf_stix_and_siem(self):
        events = [
            EDRAdapter().adapt({"host": "h1", "process": "cmd"}),
            IdentityAdapter().adapt({"user": "alice", "resource": "vpn"}),
            CloudAuditAdapter().adapt({"provider": "aws", "principal": "role"}),
            DNSAdapter().adapt({"client": "h1", "query": "example.test"}),
            ApplicationAdapter().adapt({"application": "portal", "route": "/login"}),
        ]
        self.assertTrue(all(OCSFMapper().map(event)["metadata"]["source"] for event in events))
        self.assertTrue(STIX21Exchange().export([{"concept_id": "c1", "confidence": 0.8}])["objects"])
        self.assertTrue(SIEMExporter().export(events[0], "cef").startswith("CEF:0"))

    def test_sigma_import_generate_shadow_simulation(self):
        engine = SigmaRuleEngine()
        rule = engine.import_rule(
            {
                "id": "r1",
                "title": "Shell",
                "detection": {"selection": {"process": "powershell"}, "condition": "selection"},
            }
        )
        self.assertTrue(engine.evaluate(rule, {"process": "PowerShell.exe"}))
        candidate = engine.generate_candidate({"destination_port": 445}, "T1021")
        report = engine.simulate(candidate, [{"destination_port": 445}, {"destination_port": 80}])
        self.assertEqual(report["match_count"], 1)
        self.assertEqual(report["mode"], "shadow")

    def test_detection_repository_requires_passing_tests(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = DetectionRepository(Path(directory))
            failed = repository.save({"id": "r1"}, {"passed": False})
            with self.assertRaises(ValueError):
                repository.promote(failed)
            passed = repository.save({"id": "r1"}, {"passed": True})
            self.assertEqual(repository.promote(passed)["rule"]["status"], "stable")


class PlatformEngineTests(unittest.TestCase):
    def test_memory_fork_decay_and_asset_baseline(self):
        memory = ThreatMemoryV2()
        concept = memory.create("x", np.ones(4), ["e1"], {"h1"}, {"port=445": 3})
        memory.promote(concept.concept_id, "analyst", "lateral")
        children = memory.fork(concept.concept_id, [[np.ones(4)], [np.zeros(4)]])
        self.assertEqual(len(children), 2)
        memory.observe_baseline("dns", 1755763200, np.ones(4))
        moment = __import__("datetime").datetime.fromtimestamp(1755763200, __import__("datetime").UTC)
        self.assertIsNotNone(memory.baseline("dns", moment.weekday(), moment.hour))

    def test_federation_rejects_poisoning(self):
        artifacts = [
            FederatedArtifact("good", "c", "h", (1.0, 1.0), 20, 0.8, "Scotland"),
            FederatedArtifact("poison", "c", "x", (-1.0, -1.0), 20, 1.0, "Unknown"),
        ]
        result = RobustFederation().aggregate(artifacts, {"good": 0.9, "poison": 0.1})
        self.assertIn("poison", result["rejected_as_poisoning"])
        self.assertFalse(result["raw_traffic_exchanged"])

    def test_runtime_deception_robustness_explainability_and_hdc(self):
        runtime = AdaptiveRuntime()
        self.assertEqual(len(runtime.shed([{"risk": 90}, {"risk": 2}], 1)["preserved"]), 1)
        self.assertAlmostEqual(
            sum(
                item["allocated_budget"]
                for item in runtime.sensor_budget([{"risk": 1, "uncertainty": 1}, {"risk": 2, "uncertainty": 1}], 100)
            ),
            100,
        )
        self.assertEqual(DeceptionEngine().assess({"canary": True}, {})["risk_multiplier"], 3)
        grounding = RobustnessLab().grounding([{"claim": "x", "evidence_ids": ["e1"]}], {"e1"})
        self.assertEqual(grounding["grounding_score"], 1)
        self.assertIn("identity", ExplainabilityEngine().completeness({"network": True})["missing"])
        binary = BinaryHDC()
        self.assertEqual(binary.similarity(np.ones(4), np.ones(4)), 1)
        memory = HyperdimensionalWorldMemory()
        self.assertGreater(memory.information_gain([0.5, 0.5], [0.9, 0.1]), 0)

    def test_attack_lab_timing_metrics(self):
        report = AttackLaboratoryV2().compare(
            {
                "v4": [
                    {"timestamp": 0, "risk": 10},
                    {
                        "timestamp": 5,
                        "risk": 70,
                        "correct_prediction": True,
                        "authority": True,
                        "predicted_outcome": 0.8,
                        "actual_outcome": 0.7,
                    },
                ]
            }
        )["versions"]["v4"]
        self.assertEqual(report["time_to_understand"], 5)
        self.assertEqual(report["time_to_predict"], 5)
        self.assertEqual(report["time_to_safe_action"], 5)


class PlatformSnapshotTests(unittest.TestCase):
    def test_v4_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            store = EventStore(Path(directory) / "events.db")
            flow = FlowFeatures("f1", "203.0.113.1", "10.0.0.4", 5000, 443, "TCP")
            store.add_flow(flow, "2026-08-21T10:00:00+00:00")
            store.add_decision(
                "d1",
                "f1",
                {"risk_score": 72, "action": "alert", "policy_level": 3, "evidence": {"classifier_label": "port_scan"}},
                "2026-08-21T10:00:01+00:00",
            )
            result = PlatformIntelligenceService(store).snapshot()
            store.close()
        self.assertEqual(result["version"], "4.0.0")
        self.assertTrue(result["platform_domains"])
        self.assertIn("why_graph", result["explainability"])


if __name__ == "__main__":
    unittest.main()
