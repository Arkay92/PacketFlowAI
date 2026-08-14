"""PacketFlowAI v3 world-model and predictive-defence tests."""

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from packetflowai.clustering import UnknownCluster
from packetflowai.domain import FlowFeatures
from packetflowai.governance import AuthorityGraph, EvidenceLedger, EvidenceTimeMachine
from packetflowai.intelligence import V3IntelligenceService
from packetflowai.predictive import CounterfactualResponseSimulator, NextMovePredictor
from packetflowai.storage import EventStore
from packetflowai.v3_support import (
    AdaptiveSensorController,
    AttackLab,
    BoundedPlaybookEngine,
    ContinualPrototypeStore,
    EBPFXDPBackend,
    EmergentThreatMemory,
    FederatedArtifact,
    FederatedConsensusEngine,
    PacketFlowAdapter,
    PrivacyTransformer,
    ReadOnlyInvestigator,
    STIXThreatIntelAdapter,
    SuricataAdapter,
    ZeekAdapter,
)
from packetflowai.world import CampaignCorrelator, WorldModelBuilder


def records():
    flows, decisions = [], []
    labels = ("port_scan", "credential_attack", "remote_services")
    for index, label in enumerate(labels):
        event_id = f"flow-{index}"
        created_at = f"2026-08-14T12:0{index}:00+00:00"
        flows.append({
            "flow_id": event_id,
            "created_at": created_at,
            "payload": {
                "flow_id": event_id,
                "source_ip": "203.0.113.42",
                "destination_ip": f"10.0.0.{10 + index}",
                "destination_port": (22, 445, 3389)[index],
                "protocol": "TCP",
                "protocol_metadata": {"account": "ops-admin"} if index == 1 else {},
            },
        })
        decisions.append({
            "decision_id": event_id,
            "event_id": event_id,
            "created_at": created_at,
            "payload": {
                "event_id": event_id,
                "risk_score": 42 + index * 21,
                "policy_level": index + 2,
                "action": "alert",
                "evidence": {"classifier_label": label, "anomaly_score": .45 + index * .2},
            },
        })
    return flows, decisions


class WorldModelTests(unittest.TestCase):
    def test_campaign_prediction_and_counterfactual_simulation(self):
        flows, decisions = records()
        model = WorldModelBuilder().build(flows, decisions, [])
        relationships = {edge.relationship for edge in model.edges.values()}
        self.assertTrue({"CONTACTED", "PRECEDED", "AUTHENTICATED_TO", "MAPS_TO"} <= relationships)
        campaigns = CampaignCorrelator().correlate(model)
        self.assertEqual(len(campaigns), 1)
        assessment = NextMovePredictor().predict(campaigns[0])
        self.assertAlmostEqual(sum(move.probability for move in assessment.predictions), 1.0)
        simulation = CounterfactualResponseSimulator().simulate(model, "203.0.113.42")
        self.assertEqual(len(simulation.alternatives), 4)
        self.assertIn(simulation.recommended_action, {item.action for item in simulation.alternatives})


class GovernanceTests(unittest.TestCase):
    def test_hash_chain_detects_tampering_and_authority_is_explicit(self):
        ledger = EvidenceLedger()
        ledger.append("one", "2026-08-14T12:00:00+00:00", {"signal": 1}, "model-a", "policy-v3", "auto", "observe")
        ledger.append("two", "2026-08-14T12:01:00+00:00", {"signal": 2}, "model-a", "policy-v3", "analyst", "alert")
        self.assertTrue(ledger.verify()["verified"])
        self.assertEqual(len(ledger.merkle_root()), 64)
        ledger.events[0] = replace(ledger.events[0], action="changed")
        self.assertFalse(ledger.verify()["verified"])
        self.assertTrue(AuthorityGraph().authorize("RATE_LIMIT", "engine", "policy")["permitted"])
        self.assertFalse(AuthorityGraph().authorize("QUARANTINE", "engine")["permitted"])

    def test_time_machine_hides_future_evidence(self):
        flows, decisions = records()
        snapshot = EvidenceTimeMachine().reconstruct(
            "2026-08-14T12:01:00+00:00", flows, decisions, [], [],
        )
        self.assertEqual(snapshot["known"]["flows"], 2)
        self.assertEqual(snapshot["not_yet_known"]["decisions"], 1)


class SupportSystemTests(unittest.TestCase):
    def test_sensor_stix_and_privacy_adapters(self):
        self.assertEqual(ZeekAdapter().adapt({"uid": "C1", "id.orig_h": "192.0.2.1"}).sensor, "zeek")
        self.assertEqual(SuricataAdapter().adapt({"event_type": "alert", "alert": {"severity": 2}}).sensor, "suricata")
        self.assertEqual(PacketFlowAdapter().adapt({"payload": {"flow_id": "f1"}}).event_id, "f1")
        entities = STIXThreatIntelAdapter().adapt({
            "objects": [{"type": "malware", "name": "Example", "confidence": 80}],
        })
        self.assertEqual(entities[0].confidence, .8)
        transformer = PrivacyTransformer("test")
        self.assertNotEqual(transformer.represent("10.0.0.1", "pseudonymous"), "10.0.0.1")
        self.assertEqual(transformer.represent("10.0.0.1", "role", "database"), "database")

    def test_emergent_memory_and_continual_prototype_guardrails(self):
        cluster = UnknownCluster("cluster", np.ones(4), 22, 1.0, 2.0, {"a", "b"}, 21.0, {"port=445": 22})
        memory = EmergentThreatMemory()
        concept = memory.observe(cluster)
        self.assertIsNotNone(concept)
        self.assertTrue(memory.adjudicate(concept.concept_id, "lateral-movement").status.startswith("confirmed"))
        prototypes = ContinualPrototypeStore()
        first = prototypes.update("unknown", [np.ones(4)], .9)
        second = prototypes.update("unknown", [np.full(4, 2)], .91)
        self.assertEqual((first.version, second.version), (1, 2))
        self.assertEqual(prototypes.rollback("unknown").version, 1)
        with self.assertRaises(ValueError):
            prototypes.update("unknown", [np.ones(4)], .4)

    def test_federation_investigator_playbook_and_platform_contracts(self):
        artifacts = [
            FederatedArtifact("site-a", "c1", "hash-a", (.9, .1), 20, .9, "signed:a"),
            FederatedArtifact("site-b", "c1", "hash-b", (.8, .2), 15, .85, "signed:b"),
        ]
        consensus = FederatedConsensusEngine().assess(artifacts)
        self.assertEqual(consensus["sites"], 2)
        self.assertFalse(consensus["raw_traffic_exchanged"])
        investigator = ReadOnlyInvestigator()
        self.assertTrue(investigator.query("block this host", {})["refused"])
        hypothesis = investigator.evaluate_hypothesis(
            "credential attack", {"authentication_failures": 9, "unique_usernames": 5},
        )
        self.assertEqual(hypothesis["assessment"], "strong support")
        self.assertTrue(BoundedPlaybookEngine().plan(1)["awaiting_authority"])
        self.assertEqual(AdaptiveSensorController().profile(80)["fidelity"], "high")
        self.assertEqual(EBPFXDPBackend().capability()["status"], "platform_contract")
        lab = AttackLab().evaluate({"v3": [{"timestamp": 1, "risk_score": 10}, {"timestamp": 4, "risk_score": 70}]})
        self.assertEqual(lab["reports"]["v3"]["time_to_understand_seconds"], 3)


class IntelligenceServiceTests(unittest.TestCase):
    def test_snapshot_persists_world_and_integrity_records(self):
        flows, decisions = records()
        with tempfile.TemporaryDirectory() as directory:
            store = EventStore(Path(directory) / "events.db")
            for flow_record, decision_record in zip(flows, decisions, strict=True):
                flow = flow_record["payload"]
                store.add_flow(FlowFeatures(
                    flow["flow_id"], flow["source_ip"], flow["destination_ip"],
                    49152, flow["destination_port"], flow["protocol"],
                    protocol_metadata=flow["protocol_metadata"],
                ), flow_record["created_at"])
                store.add_decision(
                    decision_record["decision_id"], decision_record["event_id"],
                    decision_record["payload"], decision_record["created_at"],
                )
            snapshot = V3IntelligenceService(store).snapshot()
            persisted = store.world_model()
            store.close()
        self.assertEqual(snapshot["version"], "3.0.0")
        self.assertTrue(snapshot["campaigns"])
        self.assertTrue(snapshot["integrity"]["verified"])
        self.assertEqual(len(persisted["sealed_events"]), 3)
        self.assertEqual(len(persisted["nodes"]), snapshot["world_model"]["counts"]["nodes"])


if __name__ == "__main__":
    unittest.main()
