"""Unified next-iteration platform snapshot."""

from __future__ import annotations

from typing import Any

from .advanced_intelligence import (
    AuthorityGraphV2,
    CausalReasoner,
    DigitalTwinV2,
    EvidenceTimeMachineV2,
    InterventionSolver,
    PredictionEngineV2,
    TwinAsset,
)
from .intelligence import V3IntelligenceService
from .platform_engines import AdaptiveRuntime, CaptureBackendRegistry, ExplainabilityEngine
from .storage import EventStore
from .world import CampaignCorrelator, WorldModelBuilder


class PlatformIntelligenceService:
    def __init__(self, store: EventStore):
        self.store = store

    def snapshot(self) -> dict[str, Any]:
        base = V3IntelligenceService(self.store).snapshot()
        flows = list(reversed(self.store.list("flows", 1000)))
        decisions = list(reversed(self.store.list("decisions", 1000)))
        alerts = list(reversed(self.store.list("alerts", 1000)))
        evidence = list(reversed(self.store.list("evidence", 1000)))
        nim = list(reversed(self.store.list("nim_assessments", 1000)))
        model = WorldModelBuilder().build(flows, decisions, alerts)
        campaigns = CampaignCorrelator().correlate(model)
        predictions = [PredictionEngineV2().predict(campaign) for campaign in campaigns]
        risk = max((float(item.get("payload", {}).get("risk_score", 0)) for item in decisions), default=0.0)
        target = V3IntelligenceService._primary_target(flows)
        twin = self._twin(model)
        authority = AuthorityGraphV2()
        authority.grant(
            "packetflowai",
            "detection_service",
            "RATE_LIMIT",
            "network",
            300,
            ["MAY_REQUEST", "MAY_EXECUTE", "MAY_ROLLBACK"],
        )
        latest = decisions[-1].get("payload", {}) | {"event_id": decisions[-1].get("event_id")} if decisions else {}
        time_state = {
            "flows": flows,
            "decisions": decisions,
            "evidence": evidence,
            "nim": nim,
            "world_nodes": [{**item, "created_at": base["generated_at"]} for item in base["world_model"]["nodes"]],
            "predictions": [{**item, "created_at": base["generated_at"]} for item in predictions],
        }
        return base | {
            "version": "4.0.0",
            "prediction_v2": predictions,
            "causal_v2": CausalReasoner().analyse(model, decisions),
            "intervention": InterventionSolver().solve(model, target, risk, 35),
            "digital_twin_v2": {
                "assets": [asset.__dict__ for asset in twin.assets.values()],
                "relationships": twin.relationships,
                "what_if": twin.what_if({target}),
            },
            "time_machine_v2": EvidenceTimeMachineV2().replay(base["generated_at"], time_state),
            "authority_v2": {
                "grants": authority.grants,
                "rate_limit": authority.authorize("RATE_LIMIT", "network", [{"subject": "soc-analyst"}]),
            },
            "explainability": {
                "why_graph": ExplainabilityEngine().why_graph(latest),
                "completeness": ExplainabilityEngine().completeness(
                    {
                        "network": bool(flows),
                        "identity": None,
                        "endpoint": None,
                        "dns": None,
                        "threat_intel": None,
                    }
                ),
            },
            "runtime_v2": {
                "capture_backends": CaptureBackendRegistry().capabilities(),
                "adaptive_batch": AdaptiveRuntime().batch_size(len(flows), 12),
            },
            "platform_domains": self._domains(),
        }

    @staticmethod
    def _twin(model: Any) -> DigitalTwinV2:
        twin = DigitalTwinV2()
        for node in model.nodes.values():
            if node.kind not in {"HOST", "SOURCE", "SERVICE", "ACCOUNT"}:
                continue
            criticality = 0.9 if node.kind in {"SERVICE", "ACCOUNT"} else 0.55
            twin.add_asset(TwinAsset(node.label, node.kind, criticality, 0.65, 0.8 if node.kind == "ACCOUNT" else 0.4))
        for edge in model.edges.values():
            source = model.nodes.get(edge.source)
            target = model.nodes.get(edge.target)
            if not source or not target or source.label not in twin.assets or target.label not in twin.assets:
                continue
            relationship = "DEPENDS_ON" if edge.relationship == "CONTAINS" else "CAN_CONNECT_TO"
            twin.connect(source.label, target.label, relationship)
        return twin

    @staticmethod
    def _domains() -> list[dict[str, str]]:
        return [
            {"domain": "Provable forensics", "status": "ACTIVE", "detail": ".pfcase + independent verifier"},
            {"domain": "Campaign intelligence", "status": "ACTIVE", "detail": "causal + multi-horizon prediction"},
            {"domain": "Counterfactual defence", "status": "ACTIVE", "detail": "minimum intervention solver"},
            {"domain": "Threat memory", "status": "ACTIVE", "detail": "lineage, forks, decay, baselines"},
            {"domain": "Collective defence", "status": "ACTIVE", "detail": "weighted consensus + poisoning defence"},
            {"domain": "SOC interoperability", "status": "ACTIVE", "detail": "OCSF, STIX/TAXII, Sigma, SIEM"},
            {"domain": "Adaptive runtime", "status": "AVAILABLE", "detail": "risk shedding + Linux fast paths"},
            {"domain": "Adversarial assurance", "status": "ACTIVE", "detail": "evasion, grounding, poisoning"},
            {"domain": "Neuro-symbolic research", "status": "ACTIVE", "detail": "HDC episodes + uncertainty"},
        ]
