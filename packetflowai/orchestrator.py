"""End-to-end flow detection, reasoning, fusion, policy, and evidence persistence."""

import logging
from dataclasses import asdict
from datetime import UTC, datetime

from .actions import AlertAdapter, ResponsePolicyEngine
from .domain import FlowFeatures, NIMAssessment
from .fusion import DeterministicFusionEngine, FusionDecision
from .inference import FlowInferenceService
from .reasoning import ReasoningProvider, UncertaintyGate
from .storage import EventStore
from .telemetry import MetricsRegistry


class DetectionOrchestrator:
    def __init__(self, inference: FlowInferenceService, store: EventStore,
                 metrics: MetricsRegistry, reasoning: ReasoningProvider | None = None,
                 nim_mode: str = "disabled", containment_enabled: bool = False):
        self.inference = inference
        self.store = store
        self.metrics = metrics
        self.reasoning = reasoning
        self.nim_mode = nim_mode
        self.gate = UncertaintyGate()
        self.fusion = DeterministicFusionEngine()
        self.policy = ResponsePolicyEngine(containment_enabled=containment_enabled)
        self.alert_adapter = AlertAdapter()
        self.last_decisions: list[dict] = []

    def handle_flow(self, flow: FlowFeatures) -> FusionDecision:
        created_at = datetime.now(UTC).isoformat()
        self.store.add_flow(flow, created_at)
        local = self.inference.predict(flow)
        self.store.add_evidence(flow.flow_id, "local", local, created_at)
        nim: NIMAssessment | None = None
        if self.reasoning is not None and self.nim_mode != "disabled" and self.gate.should_escalate(local):
            try:
                nim = self.reasoning.assess({"flow": asdict(flow), "local_prediction": asdict(local)})
                self.store.add_nim_assessment(flow.flow_id, nim, created_at)
                self.store.add_evidence(flow.flow_id, "nim", nim, created_at)
                self.metrics.increment("nim_assessments")
            except RuntimeError:
                logging.exception("NIM unavailable; continuing with local evidence")
                self.metrics.increment("nim_failures")
        decision = self.fusion.decide(local, nim, containment_enabled=self.policy.containment_enabled)
        self.store.add_decision(flow.flow_id, flow.flow_id, decision, created_at)
        self.metrics.increment("flows_classified")
        self.metrics.set("last_risk_score", decision.risk_score)
        requests = self.policy.decide(flow.flow_id, flow.source_ip, decision.policy_level, decision.explanation)
        for request in requests:
            self.store.add_alert(request.action_id, flow.flow_id, request, created_at)
            self.last_decisions.append({"flow_id": flow.flow_id, **asdict(decision)})
            if request.action == "alert":
                self.alert_adapter.execute(request)
        return decision
