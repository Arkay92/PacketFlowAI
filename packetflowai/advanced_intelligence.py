"""Advanced prediction, causal, twin, temporal, authority, and decision analysis."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from .predictive import CounterfactualResponseSimulator, NextMovePredictor
from .world import Campaign, CausalGraphBuilder, CyberWorldModel


class PredictionEngineV2:
    HORIZON_WEIGHTS = {"5m": (1.15, 0.8), "1h": (1.0, 1.0), "24h": (0.72, 1.35)}

    def predict(self, campaign: Campaign, cross_site_support: float = 0.0) -> dict[str, Any]:
        base = NextMovePredictor().predict(campaign)
        horizons = {}
        for horizon, (active_weight, continuation_weight) in self.HORIZON_WEIGHTS.items():
            raw = [
                move.probability * (continuation_weight if move.technique_id == "NONE" else active_weight)
                for move in base.predictions
            ]
            total = sum(raw)
            horizons[horizon] = [
                {**asdict(move), "probability": score / total, "time_horizon": horizon}
                for move, score in zip(base.predictions, raw, strict=True)
            ]
        sequence_support = min(0.95, 0.45 + len(campaign.techniques) * 0.11)
        return {
            "campaign_id": campaign.campaign_id,
            "horizons": horizons,
            "confidence_decomposition": {
                "historical_sequence_similarity": sequence_support,
                "attack_path_support": min(0.95, 0.52 + len(campaign.techniques) * 0.08),
                "local_campaign_evidence": campaign.confidence,
                "cross_site_evidence": max(0.0, min(1.0, cross_site_support)),
            },
            "prediction_set": [move["technique_id"] for move in horizons["1h"] if move["probability"] >= 0.2],
            "configured_coverage": 0.9,
        }

    def calibration(self, outcomes: list[dict[str, Any]], bins: int = 5) -> dict[str, Any]:
        buckets = []
        error = 0.0
        for index in range(bins):
            low, high = index / bins, (index + 1) / bins
            selected = [item for item in outcomes if low <= float(item["probability"]) <= high]
            predicted = sum(float(item["probability"]) for item in selected) / len(selected) if selected else 0.0
            observed = sum(bool(item["occurred"]) for item in selected) / len(selected) if selected else 0.0
            error += len(selected) * abs(predicted - observed)
            buckets.append({"range": [low, high], "count": len(selected), "predicted": predicted, "observed": observed})
        return {"buckets": buckets, "expected_calibration_error": error / max(1, len(outcomes))}

    def belief_change(self, previous: dict[str, float], current: dict[str, float]) -> list[dict[str, Any]]:
        return [
            {
                "technique": key,
                "previous": previous.get(key, 0.0),
                "current": current.get(key, 0.0),
                "pruned": previous.get(key, 0.0) >= 0.2 and current.get(key, 0.0) < 0.2,
            }
            for key in sorted(previous.keys() | current.keys())
        ]


class CausalReasoner:
    def analyse(self, model: CyberWorldModel, decisions: list[dict[str, Any]]) -> dict[str, Any]:
        links = [asdict(link) | {"behavioural_support": 0.7} for link in CausalGraphBuilder().build(model)]
        alternatives = (
            [
                {"hypothesis": "attack progression", "probability": 0.71},
                {"hypothesis": "administrative activity", "probability": 0.18},
                {"hypothesis": "unrelated coincidence", "probability": 0.11},
            ]
            if links
            else []
        )
        ordered = sorted(decisions, key=lambda item: str(item.get("created_at", "")))
        sufficient = next((item for item in ordered if float(item.get("payload", {}).get("risk_score", 0)) >= 60), None)
        first_action = next(
            (item for item in ordered if item.get("payload", {}).get("action") not in {None, "observe"}), None
        )
        return {
            "links": links,
            "alternative_explanations": alternatives,
            "root_cause": links[0]["source_event"] if links else None,
            "earliest_intervention": sufficient.get("created_at") if sufficient else None,
            "missed_opportunity": {
                "detected": bool(sufficient and first_action and sufficient["created_at"] < first_action["created_at"]),
                "evidence_ready": sufficient.get("created_at") if sufficient else None,
                "authority_ready": first_action.get("created_at") if first_action else None,
            },
        }


@dataclass(frozen=True)
class TwinAsset:
    asset_id: str
    kind: str
    criticality: float
    exposure: float
    privilege: float
    decoy: bool = False


class DigitalTwinV2:
    def __init__(self):
        self.assets: dict[str, TwinAsset] = {}
        self.relationships: list[dict[str, str]] = []

    def add_asset(self, asset: TwinAsset) -> None:
        self.assets[asset.asset_id] = asset

    def connect(self, source: str, target: str, relationship: str) -> None:
        allowed = {"CAN_AUTHENTICATE_TO", "CAN_CONNECT_TO", "DEPENDS_ON", "ADMINISTERS", "TRUSTS", "ROUTES_TO"}
        if relationship not in allowed:
            raise ValueError(f"unsupported twin relationship: {relationship}")
        self.relationships.append({"source": source, "target": target, "relationship": relationship})

    def exposure_score(self, asset_id: str, observed_threat: float, reachable_surface: float) -> float:
        asset = self.assets[asset_id]
        score = asset.criticality * asset.exposure * observed_threat * reachable_surface * asset.privilege
        return min(100.0, score * 100)

    def paths(self, source: str, target: str, excluded: set[str] | None = None) -> list[list[str]]:
        excluded = excluded or set()
        paths: list[list[str]] = []
        frontier = [[source]]
        while frontier:
            path = frontier.pop(0)
            if len(path) > 7:
                continue
            for edge in self.relationships:
                if edge["source"] != path[-1] or edge["target"] in path or edge["target"] in excluded:
                    continue
                candidate = path + [edge["target"]]
                if edge["target"] == target:
                    paths.append(candidate)
                else:
                    frontier.append(candidate)
        return paths

    def what_if(self, removed_assets: set[str]) -> dict[str, Any]:
        impacted = {
            edge["source"]
            for edge in self.relationships
            if edge["target"] in removed_assets and edge["relationship"] == "DEPENDS_ON"
        }
        return {
            "removed": sorted(removed_assets),
            "dependencies_lost": sorted(impacted),
            "attack_surface_reduction": len(removed_assets) / max(1, len(self.assets)),
        }


class InterventionSolver:
    REVERSIBILITY = {
        "OBSERVE": 1.0,
        "RATE_LIMIT": 0.97,
        "BLOCK_SOURCE": 0.92,
        "ISOLATE_TARGET": 0.75,
        "REVOKE_CREDENTIAL": 0.48,
    }

    def solve(self, model: CyberWorldModel, target: str, current_risk: float, threshold: float) -> dict[str, Any]:
        simulation = CounterfactualResponseSimulator().simulate(model, target)
        candidates = []
        for item in simulation.alternatives:
            residual = current_risk * (1 - item.risk_reduction)
            candidates.append(
                asdict(item) | {"residual_risk": residual, "reversibility": self.REVERSIBILITY.get(item.action, 0.5)}
            )
        safe = [
            item
            for item in candidates
            if item["residual_risk"] <= threshold and not item["critical_dependency_affected"]
        ]
        selected = max(
            safe, key=lambda item: (item["reversibility"], -item["legitimate_flows_disrupted"]), default=None
        )
        return {"threshold": threshold, "scenarios": candidates, "minimum_intervention": selected}


class EvidenceTimeMachineV2:
    def replay(self, timestamp: str, state: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
        cutoff = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        known, learned = {}, {}
        for category, records in state.items():
            known[category] = [item for item in records if self._time(item) <= cutoff]
            learned[category] = [item for item in records if self._time(item) > cutoff]
        leakage = [
            {"category": category, "id": item.get("event_id") or item.get("id")}
            for category, records in known.items()
            for item in records
            if self._time(item) > cutoff
        ]
        return {"as_of": timestamp, "known_then": known, "learned_later": learned, "hindsight_leakage": leakage}

    @staticmethod
    def _time(item: dict[str, Any]) -> datetime:
        value = item.get("created_at") or item.get("timestamp") or datetime.min.replace(tzinfo=UTC).isoformat()
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))

    def regret(self, risk: float, action: str, outcome_harm: float) -> dict[str, Any]:
        reasonable = action != "OBSERVE" if risk >= 65 else action in {"OBSERVE", "ALERT", "RATE_LIMIT"}
        return {
            "reasonable_given_known_evidence": reasonable,
            "outcome_harm": outcome_harm,
            "hindsight_adjusted": False,
        }


class AuthorityGraphV2:
    def __init__(self):
        self.grants: list[dict[str, Any]] = []

    def grant(
        self, subject: str, role: str, action: str, scope: str, ttl_seconds: int, relationships: list[str]
    ) -> dict[str, Any]:
        grant = {
            "subject": subject,
            "role": role,
            "action": action,
            "scope": scope,
            "relationships": relationships,
            "issued_at": datetime.now(UTC).isoformat(),
            "expires_at": (datetime.now(UTC) + timedelta(seconds=ttl_seconds)).isoformat(),
        }
        self.grants.append(grant)
        return grant

    def authorize(
        self, action: str, scope: str, approvals: list[dict[str, str]], now: str | None = None
    ) -> dict[str, Any]:
        instant = datetime.fromisoformat(now) if now else datetime.now(UTC)
        valid = [
            grant
            for grant in self.grants
            if grant["action"] == action
            and grant["scope"] in {scope, "*"}
            and datetime.fromisoformat(grant["expires_at"]) >= instant
        ]
        required = 2 if action in {"QUARANTINE", "REVOKE_CREDENTIAL"} else 1
        approvers = {approval["subject"] for approval in approvals}
        permitted = bool(valid) and len(approvers) >= required
        return {
            "permitted": permitted,
            "required_approvals": required,
            "approvers": sorted(approvers),
            "grant": valid[0] if valid else None,
        }

    def break_glass(self, subject: str, reason: str, ttl_seconds: int = 300) -> dict[str, Any]:
        if not reason.strip():
            raise ValueError("break-glass requires an explicit reason")
        return self.grant(subject, "emergency", "*", "*", min(ttl_seconds, 900), ["MAY_OVERRIDE", "MANDATORY_REVIEW"])

    @staticmethod
    def policy_diff(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
        keys = left.keys() | right.keys()
        return {
            key: {"before": left.get(key), "after": right.get(key)} for key in keys if left.get(key) != right.get(key)
        }


class DecisionAutopsy:
    def build(self, decision: dict[str, Any], actual: dict[str, Any]) -> dict[str, Any]:
        evidence = decision.get("evidence", {})
        authority = decision.get("authority", {})
        risk = float(decision.get("risk_score", 0))
        return {
            "what_we_saw": evidence,
            "what_we_believed": {"risk": risk, "classification": evidence.get("classifier_label")},
            "what_we_predicted": decision.get("prediction"),
            "what_we_did": decision.get("action", "observe"),
            "who_authorised_it": authority,
            "what_actually_happened": actual,
            "what_we_learned": {
                "prediction_correct": decision.get("prediction", {}).get("technique_id") == actual.get("technique_id")
            },
            "why_not": None
            if decision.get("action") == "QUARANTINE"
            else {
                "authority": "insufficient" if not authority.get("permitted") else "available",
                "business_impact": decision.get("business_impact", "unknown"),
                "alternative": decision.get("recommended_action", "RATE_LIMIT"),
            },
        }
