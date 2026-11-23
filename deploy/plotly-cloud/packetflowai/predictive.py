"""Next-move prediction and counterfactual defensive simulation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .world import Campaign, CyberWorldModel


@dataclass(frozen=True)
class PredictedMove:
    technique_id: str
    label: str
    probability: float
    time_horizon: str
    supporting_evidence: tuple[str, ...]


@dataclass(frozen=True)
class NextMoveAssessment:
    campaign_id: str
    observed_sequence: tuple[str, ...]
    predictions: tuple[PredictedMove, ...]
    uncertainty: float
    model_version: str = "attack-progression-v1"


class NextMovePredictor:
    TRANSITIONS = {
        "T1046": (("T1110", "Brute Force", .45), ("T1021", "Remote Services", .31),
                   ("NONE", "No continuation", .24)),
        "T1110": (("T1021", "Remote Services", .61), ("CREDENTIAL_REUSE", "Credential Reuse", .24),
                   ("NONE", "No continuation", .15)),
        "T1021": (("T1087", "Account Discovery", .39), ("T1041", "Exfiltration Over C2 Channel", .34),
                   ("NONE", "No continuation", .27)),
        "T1105": (("T1059", "Command and Scripting Interpreter", .47), ("T1021", "Remote Services", .33),
                   ("NONE", "No continuation", .20)),
        "T1041": (("T1070", "Indicator Removal", .38), ("T1102", "Web Service", .22),
                   ("NONE", "No continuation", .40)),
    }

    def predict(self, campaign: Campaign) -> NextMoveAssessment:
        observed = campaign.techniques or ("UNKNOWN",)
        latest = observed[-1]
        transitions = self.TRANSITIONS.get(
            latest,
            (("T1046", "Network Service Discovery", .35), ("T1021", "Remote Services", .20),
             ("NONE", "No continuation", .45)),
        )
        evidence = tuple(campaign.event_ids[-3:]) or (campaign.campaign_id,)
        moves = tuple(
            PredictedMove(technique, label, probability, "next 30 minutes", evidence)
            for technique, label, probability in transitions
        )
        uncertainty = 1.0 - max(move.probability for move in moves)
        return NextMoveAssessment(campaign.campaign_id, observed, moves, uncertainty)


@dataclass(frozen=True)
class SimulationAlternative:
    action: str
    risk_reduction: float
    business_impact: str
    threat_paths_removed: int
    legitimate_flows_disrupted: int
    critical_dependency_affected: bool
    blast_radius: str
    evidence_gain: str
    authority_required: str


@dataclass(frozen=True)
class CounterfactualSimulation:
    target: str
    proposed_action: str
    alternatives: tuple[SimulationAlternative, ...]
    recommended_action: str
    rationale: str
    graph_version: str = "digital-twin-v1"

    def serialize(self) -> dict[str, Any]:
        return asdict(self)


class CounterfactualResponseSimulator:
    BASELINES = {
        "BLOCK_SOURCE": (.81, "HIGH", .85, "soc"),
        "RATE_LIMIT": (.58, "LOW", .45, "policy"),
        "ISOLATE_TARGET": (.93, "MEDIUM", .72, "senior"),
        "OBSERVE": (0.0, "NONE", 0.0, "autonomous"),
    }

    def simulate(
        self,
        model: CyberWorldModel,
        target: str,
        proposed_action: str = "BLOCK_SOURCE",
    ) -> CounterfactualSimulation:
        target_nodes = [node for node in model.nodes.values() if node.label == target]
        target_node = target_nodes[0] if target_nodes else None
        impacted = model.impact(target_node.node_id, depth=2) if target_node else set()
        dependency_count = sum(
            edge.relationship in {"DEPENDS_ON", "TRUSTS"} and (edge.source in impacted or edge.target in impacted)
            for edge in model.edges.values()
        )
        threat_paths = max(1, sum(
            edge.relationship in {"SCANNED", "AUTHENTICATED_TO", "CONTACTED", "DOWNLOADED_FROM"}
            and (not target_node or target_node.node_id in {edge.source, edge.target})
            for edge in model.edges.values()
        ))
        alternatives = []
        for action, (reduction, impact, disruption_factor, authority) in self.BASELINES.items():
            disrupted = 0 if action == "OBSERVE" else max(0, round(len(impacted) * disruption_factor))
            critical = dependency_count > 0 and action in {"BLOCK_SOURCE", "ISOLATE_TARGET"}
            blast = "HIGH" if disrupted >= 8 or critical else "MEDIUM" if disrupted >= 3 else "LOW"
            alternatives.append(SimulationAlternative(
                action,
                reduction,
                impact,
                round(threat_paths * reduction),
                disrupted,
                critical,
                blast,
                "HIGH" if action == "OBSERVE" else "MEDIUM",
                authority,
            ))
        safe = [item for item in alternatives if not item.critical_dependency_affected]
        recommended = max(
            safe or alternatives,
            key=lambda item: item.risk_reduction - item.legitimate_flows_disrupted * .03,
        )
        return CounterfactualSimulation(
            target,
            proposed_action,
            tuple(alternatives),
            recommended.action,
            f"{recommended.action} maximizes estimated risk reduction without crossing known critical dependencies.",
        )
