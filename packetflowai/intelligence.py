"""PacketFlowAI v3 intelligence snapshot orchestration."""

from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any

from .governance import AuthorityGraph, EvidenceLedger, EvidenceTimeMachine
from .predictive import CounterfactualResponseSimulator, NextMovePredictor
from .storage import EventStore
from .v3_support import (
    AdaptiveSensorController,
    BoundedPlaybookEngine,
    CaseNarrativeBuilder,
    DisagreementEngine,
    EBPFXDPBackend,
    HDCAccelerationRegistry,
)
from .world import CampaignCorrelator, CausalGraphBuilder, WorldModelBuilder


class V3IntelligenceService:
    """Build a deterministic, persistent command snapshot from recorded evidence."""

    def __init__(self, store: EventStore):
        self.store = store

    def snapshot(self) -> dict[str, Any]:
        flows = list(reversed(self.store.list("flows", 1000)))
        decisions = list(reversed(self.store.list("decisions", 1000)))
        alerts = list(reversed(self.store.list("alerts", 1000)))
        evidence = list(reversed(self.store.list("evidence", 1000)))
        nim = list(reversed(self.store.list("nim_assessments", 1000)))

        model = WorldModelBuilder().build(flows, decisions, alerts)
        campaigns = CampaignCorrelator().correlate(model)
        world = model.serialize()
        campaign_records = [asdict(campaign) for campaign in campaigns]
        generated_at = datetime.now(UTC).isoformat()
        self.store.replace_world_model(world["nodes"], world["edges"], campaign_records, generated_at)

        predictions = [asdict(NextMovePredictor().predict(campaign)) for campaign in campaigns]
        target = self._primary_target(flows)
        simulation = CounterfactualResponseSimulator().simulate(model, target).serialize()
        causal_links = [asdict(link) for link in CausalGraphBuilder().build(model)]
        ledger = self._ledger(decisions, evidence)
        sealed = [asdict(event) for event in ledger.events]
        self.store.replace_sealed_events(sealed)
        timeline = self._timeline(flows, decisions, evidence, nim)
        primary_campaign = campaigns[0].campaign_id if campaigns else "campaign-none"
        narrative = CaseNarrativeBuilder().build(primary_campaign, decisions)
        risk = max((float(row.get("payload", {}).get("risk_score", 0)) for row in decisions), default=0.0)

        return {
            "version": "3.0.0",
            "generated_at": generated_at,
            "world_model": world,
            "campaigns": campaign_records,
            "causal_links": causal_links,
            "predictions": predictions,
            "simulation": simulation,
            "time_machine": timeline,
            "integrity": {**ledger.verify(), "sealed_events": sealed[-8:]},
            "authority": AuthorityGraph().serialize(),
            "disagreements": DisagreementEngine().find(
                [row.get("payload", {}) | {"event_id": row.get("event_id")} for row in decisions]
            ),
            "narrative": narrative,
            "playbook": BoundedPlaybookEngine().plan(2),
            "sensor_profile": AdaptiveSensorController().profile(risk),
            "capabilities": self._capabilities(),
        }

    @staticmethod
    def _primary_target(flows: list[dict[str, Any]]) -> str:
        if not flows:
            return "no-active-target"
        payload = flows[-1].get("payload", {})
        return str(payload.get("source_ip") or payload.get("destination_ip") or "unknown")

    @staticmethod
    def _ledger(decisions: list[dict[str, Any]], evidence: list[dict[str, Any]]) -> EvidenceLedger:
        ledger = EvidenceLedger()
        evidence_by_event: dict[str, list[dict[str, Any]]] = {}
        for row in evidence:
            evidence_by_event.setdefault(str(row.get("event_id")), []).append(row.get("payload", {}))
        for row in decisions:
            payload = row.get("payload", {})
            event_id = str(row.get("event_id") or payload.get("event_id") or "unknown")
            ledger.append(
                event_id,
                str(row.get("created_at")),
                evidence_by_event.get(event_id) or payload.get("evidence", {}),
                str(payload.get("model_version", "local-hdc")),
                "policy-v3",
                "autonomous" if not payload.get("requires_confirmation") else "human-required",
                str(payload.get("action", "observe")),
            )
        return ledger

    @staticmethod
    def _timeline(
        flows: list[dict[str, Any]],
        decisions: list[dict[str, Any]],
        evidence: list[dict[str, Any]],
        nim: list[dict[str, Any]],
    ) -> dict[str, Any]:
        timestamps = sorted({str(row.get("created_at")) for row in flows + decisions if row.get("created_at")})
        snapshots = []
        machine = EvidenceTimeMachine()
        for timestamp in timestamps:
            snapshots.append(machine.reconstruct(timestamp, flows, decisions, evidence, nim))
        return {
            "range": {"start": timestamps[0] if timestamps else None, "end": timestamps[-1] if timestamps else None},
            "snapshots": snapshots,
        }

    @staticmethod
    def _capabilities() -> list[dict[str, Any]]:
        return [
            {"name": "Canonical sensor fabric", "status": "ACTIVE", "detail": "Zeek, Suricata and native adapters"},
            {"name": "Emergent threat memory", "status": "ACTIVE", "detail": "Unknown-cluster concept lifecycle"},
            {"name": "Continual prototypes", "status": "GUARDED", "detail": "Holdout validation and rollback"},
            {"name": "Federated consensus", "status": "READY", "detail": "No raw traffic exchange"},
            {"name": "Read-only investigator", "status": "ACTIVE", "detail": "Evidence-grounded hypothesis testing"},
            {"name": "Adaptive sensing", "status": "ACTIVE", "detail": "Risk-directed fidelity"},
            {"name": "Attack laboratory", "status": "READY", "detail": "Version replay and time-to-understand"},
            {"name": "eBPF/XDP fast path", "status": "CONTRACT", "detail": EBPFXDPBackend().name},
            {"name": "HDC acceleration", "status": "TIERED", "detail": HDCAccelerationRegistry().capabilities()},
        ]
