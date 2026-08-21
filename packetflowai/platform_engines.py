"""Learning, federation, runtime, deception, robustness, explainability, and research engines."""

from __future__ import annotations

import hashlib
import math
import os
import socket
import subprocess
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np

from .clustering import cosine_similarity
from .v3_support import FederatedArtifact


@dataclass
class ThreatConceptV2:
    concept_id: str
    status: str
    first_seen: str
    last_seen: str
    vector: tuple[float, ...]
    supporting_evidence: list[str]
    affected_hosts: set[str]
    traits: dict[str, int]
    confidence: float
    lineage: list[str]


class ThreatMemoryV2:
    def __init__(self):
        self.concepts: dict[str, ThreatConceptV2] = {}
        self.asset_baselines: dict[str, list[tuple[int, np.ndarray]]] = {}

    def create(
        self, cluster_id: str, vector: np.ndarray, evidence: list[str], hosts: set[str], traits: dict[str, int]
    ) -> ThreatConceptV2:
        concept_id = "PF-UNKNOWN-" + hashlib.sha256(cluster_id.encode()).hexdigest()[:8].upper()
        now = datetime.now(UTC).isoformat()
        concept = ThreatConceptV2(
            concept_id, "PROVISIONAL", now, now, tuple(vector), evidence, hosts, traits, 0.65, [cluster_id]
        )
        self.concepts[concept_id] = concept
        return concept

    def promote(self, concept_id: str, analyst: str, family: str) -> ThreatConceptV2:
        concept = self.concepts[concept_id]
        concept.status = f"VALIDATED:{family}"
        concept.lineage.append(f"analyst:{analyst}")
        concept.confidence = max(0.9, concept.confidence)
        return concept

    def fork(self, concept_id: str, groups: list[list[np.ndarray]]) -> list[ThreatConceptV2]:
        parent = self.concepts[concept_id]
        children = []
        for index, samples in enumerate(groups):
            child = self.create(
                f"{concept_id}:{index}",
                np.mean(samples, axis=0),
                parent.supporting_evidence,
                set(parent.affected_hosts),
                dict(parent.traits),
            )
            child.lineage.insert(0, concept_id)
            children.append(child)
        return children

    def decay(self, as_of: str, half_life_days: float = 90) -> None:
        instant = datetime.fromisoformat(as_of)
        for concept in self.concepts.values():
            age = max(0.0, (instant - datetime.fromisoformat(concept.last_seen)).total_seconds() / 86400)
            concept.confidence *= math.pow(0.5, age / half_life_days)

    def observe_baseline(self, asset: str, timestamp: int, vector: np.ndarray) -> None:
        self.asset_baselines.setdefault(asset, []).append((timestamp, np.asarray(vector)))

    def baseline(self, asset: str, weekday: int, hour: int) -> np.ndarray | None:
        samples = [
            vector
            for timestamp, vector in self.asset_baselines.get(asset, [])
            if datetime.fromtimestamp(timestamp, UTC).weekday() == weekday
            and datetime.fromtimestamp(timestamp, UTC).hour == hour
        ]
        return np.mean(samples, axis=0) if samples else None


class RobustFederation:
    def aggregate(self, artifacts: Iterable[FederatedArtifact], reputation: Mapping[str, float]) -> dict[str, Any]:
        items = list(artifacts)
        accepted: list[tuple[FederatedArtifact, float]] = []
        rejected: list[tuple[FederatedArtifact, float]] = []
        signatures = (
            np.asarray([item.similarity_signature for item in items], dtype=float) if items else np.empty((0, 0))
        )
        median = np.median(signatures, axis=0) if len(signatures) else np.asarray([])
        for item in items:
            similarity = cosine_similarity(np.asarray(item.similarity_signature), median) if median.size else 0.0
            poison = similarity < 0.35 or item.confidence > 0.99 or item.sample_count <= 0
            (rejected if poison else accepted).append((item, similarity))
        weights = [
            max(0.01, reputation.get(item.site_id, 0.5)) * item.confidence * item.sample_count for item, _ in accepted
        ]
        consensus = sum(similarity * weight for (_, similarity), weight in zip(accepted, weights, strict=True)) / max(
            0.001, sum(weights)
        )
        return {
            "consensus": consensus,
            "peer_sites": len({item.site_id for item, _ in accepted}),
            "accepted": [item.site_id for item, _ in accepted],
            "rejected_as_poisoning": [item.site_id for item, _ in rejected],
            "raw_traffic_exchanged": False,
            "global_constellation": [item.provenance for item, _ in accepted],
        }


class AdaptiveRuntime:
    def batch_size(self, queue_depth: int, latency_ms: float, target_latency_ms: float = 20) -> int:
        pressure = queue_depth / 100 + max(0.0, latency_ms - target_latency_ms) / target_latency_ms
        return min(512, max(1, round(16 * (1 + pressure))))

    def shed(self, flows: list[Mapping[str, Any]], capacity: int) -> dict[str, Any]:
        ranked = sorted(
            flows, key=lambda item: (float(item.get("risk", 0)), float(item.get("uncertainty", 0))), reverse=True
        )
        return {"preserved": ranked[:capacity], "summarised": ranked[capacity:], "policy": "risk_uncertainty_priority"}

    def sensor_budget(self, assets: list[Mapping[str, Any]], budget: float) -> list[dict[str, Any]]:
        scores = [
            float(item.get("risk", 0)) * float(item.get("uncertainty", 0)) * float(item.get("criticality", 1))
            for item in assets
        ]
        total = sum(scores) or 1.0
        return [
            dict(item)
            | {"allocated_budget": budget * score / total, "sampling": "full" if score / total >= 0.25 else "dynamic"}
            for item, score in zip(assets, scores, strict=True)
        ]

    def next_observation(self, missing: list[Mapping[str, Any]]) -> dict[str, Any] | None:
        selected = max(missing, key=lambda item: float(item.get("information_gain", 0)), default=None)
        return dict(selected) if selected is not None else None


class CaptureBackendRegistry:
    def capabilities(self) -> list[dict[str, Any]]:
        return [
            {"backend": "scapy", "status": "active", "zero_copy": False},
            {"backend": "af_packet", "status": "linux", "zero_copy": True, "mode": "TPACKET_V3"},
            {"backend": "af_xdp", "status": "linux_optional", "zero_copy": True, "mode": "UMEM"},
            {
                "backend": "ebpf_xdp",
                "status": "linux_optional",
                "kernel_work": "coarse filter + flow telemetry",
                "userspace_work": "HDC + fusion + policy",
            },
        ]


class LinuxFastPathBackend:
    """Operational Linux raw-ring path with explicit XDP attachment lifecycle."""

    def __init__(self, interface: str, frame_size: int = 65536):
        self.interface = interface
        self.buffer = bytearray(frame_size)
        self.socket: socket.socket | None = None

    def open(self) -> None:
        if not sys.platform.startswith("linux") or not hasattr(socket, "AF_PACKET"):
            raise RuntimeError("AF_PACKET fast path requires Linux")
        self.socket = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.htons(3))
        self.socket.bind((self.interface, 0))

    def receive(self) -> memoryview:
        if self.socket is None:
            raise RuntimeError("fast path is not open")
        size = self.socket.recv_into(self.buffer)
        return memoryview(self.buffer)[:size]

    def attach_xdp(self, object_path: str, section: str = "xdp") -> None:
        if not sys.platform.startswith("linux") or os.geteuid() != 0:
            raise PermissionError("XDP attachment requires Linux root or delegated capabilities")
        subprocess.run(
            ["ip", "link", "set", "dev", self.interface, "xdp", "obj", object_path, "sec", section],
            check=True,
            capture_output=True,
        )

    def close(self) -> None:
        if self.socket:
            self.socket.close()
            self.socket = None


class BinaryHDC:
    def bind(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return np.bitwise_xor(np.asarray(left, dtype=np.uint8), np.asarray(right, dtype=np.uint8))

    def similarity(self, left: np.ndarray, right: np.ndarray) -> float:
        left_bits, right_bits = np.asarray(left, dtype=np.uint8), np.asarray(right, dtype=np.uint8)
        return 1.0 - float(np.count_nonzero(left_bits != right_bits)) / max(1, left_bits.size)

    def benchmark(self, dimensions: int = 10000, iterations: int = 100) -> dict[str, Any]:
        left = np.arange(dimensions, dtype=np.uint8) % 2
        right = 1 - left
        started = datetime.now(UTC)
        for _ in range(iterations):
            self.bind(left, right)
        elapsed = max(0.000001, (datetime.now(UTC) - started).total_seconds())
        return {
            "backend": "numpy_binary_simd",
            "operations_per_second": iterations / elapsed,
            "dimensions": dimensions,
            "energy_measurement": "not_available",
        }


class DeceptionEngine:
    def assess(self, asset: Mapping[str, Any], event: Mapping[str, Any]) -> dict[str, Any]:
        deception = bool(asset.get("honeypot") or asset.get("canary") or asset.get("decoy"))
        expected_users = set(asset.get("expected_users", []))
        unexpected = bool(event.get("identity") and event.get("identity") not in expected_users)
        return {
            "deception_signal": deception,
            "unexpected_identity": unexpected,
            "risk_multiplier": 3.0 if deception else 1.5 if unexpected else 1.0,
            "reason": "interaction with non-production defensive asset" if deception else "identity expectation check",
        }


class RobustnessLab:
    def evasion(self, classifier: Any, sample: np.ndarray, corruptions: list[float]) -> dict[str, Any]:
        baseline = classifier(sample)
        results = []
        rng = np.random.default_rng(42)
        for amount in corruptions:
            mask = rng.random(sample.shape) < amount
            changed = np.asarray(sample).copy()
            changed[mask] *= -1
            result = classifier(changed)
            results.append({"corruption": amount, "result": result, "identity_preserved": result == baseline})
        return {"baseline": baseline, "experiments": results}

    def feedback_poisoning(self, updates: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
        return [
            dict(item)
            | {
                "suspected_poisoning": float(item.get("label_flip_rate", 0)) > 0.35
                or float(item.get("vector_outlier", 0)) > 0.8
            }
            for item in updates
        ]

    def grounding(self, claims: list[Mapping[str, Any]], evidence_ids: set[str]) -> dict[str, Any]:
        classified = []
        for claim in claims:
            references = set(claim.get("evidence_ids", []))
            level = (
                "directly_supported"
                if references and references <= evidence_ids
                else "unsupported"
                if not references
                else "inferred"
            )
            classified.append(dict(claim) | {"grounding": level})
        supported = sum(item["grounding"] != "unsupported" for item in classified)
        return {
            "claims": classified,
            "grounding_score": supported / max(1, len(classified)),
            "hallucination_count": len(classified) - supported,
        }


class ExplainabilityEngine:
    def why_graph(self, decision: Mapping[str, Any]) -> dict[str, Any]:
        event_id = str(decision.get("event_id", "unknown"))
        nodes: list[dict[str, Any]] = [
            {"id": f"action:{event_id}", "kind": "ACTION", "value": decision.get("action")},
            {"id": f"policy:{event_id}", "kind": "POLICY", "value": decision.get("policy_level")},
            {"id": f"risk:{event_id}", "kind": "RISK", "value": decision.get("risk_score")},
            {"id": f"evidence:{event_id}", "kind": "EVIDENCE", "value": decision.get("evidence")},
            {"id": f"observation:{event_id}", "kind": "OBSERVATION", "value": event_id},
        ]
        return {
            "nodes": nodes,
            "edges": [
                {"source": nodes[index]["id"], "target": nodes[index + 1]["id"]} for index in range(len(nodes) - 1)
            ],
        }

    def feature_contributions(
        self, features: Mapping[str, float], weights: Mapping[str, float]
    ) -> list[dict[str, Any]]:
        values = [
            {"feature": key, "value": value, "contribution": value * weights.get(key, 0.0)}
            for key, value in features.items()
        ]
        return sorted(values, key=lambda item: abs(item["contribution"]), reverse=True)

    def completeness(self, channels: Mapping[str, Any]) -> dict[str, Any]:
        required = {"network", "identity", "endpoint", "dns", "threat_intel"}
        present = {key for key, value in channels.items() if value is not None}
        return {
            "score": len(required & present) / len(required),
            "missing": sorted(required - present),
            "suggested_questions": [
                f"Can we acquire {item.replace('_', ' ')} context?" for item in sorted(required - present)
            ],
        }

    def similar_incidents(
        self, vector: np.ndarray, incidents: Mapping[str, np.ndarray], limit: int = 5
    ) -> list[dict[str, Any]]:
        ranked: list[dict[str, Any]] = sorted(
            ({"incident_id": key, "similarity": cosine_similarity(vector, value)} for key, value in incidents.items()),
            key=lambda item: float(item["similarity"]),
            reverse=True,
        )
        return ranked[:limit]


class AnalystChallengeEngine:
    def hypotheses(
        self, theories: Mapping[str, Mapping[str, float]], evidence: Mapping[str, float]
    ) -> list[dict[str, Any]]:
        results = []
        for theory, expectations in theories.items():
            supporting = [key for key, weight in expectations.items() if evidence.get(key, 0) * weight > 0.25]
            contradicting = [key for key, weight in expectations.items() if evidence.get(key, 0) * weight < -0.15]
            score = sum(evidence.get(key, 0) * weight for key, weight in expectations.items())
            results.append(
                {"hypothesis": theory, "score": score, "supporting": supporting, "contradicting": contradicting}
            )
        return sorted(results, key=lambda item: float(item["score"]), reverse=True)

    def devils_advocate(self, action: str, simulation: Mapping[str, Any], missing: list[str]) -> dict[str, Any]:
        arguments = []
        if simulation.get("critical_dependency_affected"):
            arguments.append("The action crosses a critical dependency.")
        if float(simulation.get("legitimate_flows_disrupted", 0)) > 0:
            arguments.append("Legitimate communications will be disrupted.")
        arguments.extend(f"Missing {item} context could change this decision." for item in missing)
        return {
            "challenged_action": action,
            "strongest_arguments_against": arguments,
            "recommendation": "DEFER" if arguments else "NO MATERIAL OBJECTION",
        }


class ResearchBackendRegistry:
    def matrix(self) -> list[dict[str, Any]]:
        return [
            {"model": "HDC world model", "status": "implemented", "purpose": "structural similarity"},
            {"model": "HDC episodic memory", "status": "implemented", "purpose": "incident retrieval"},
            {"model": "Bayesian campaign updater", "status": "deterministic_prior", "purpose": "belief updates"},
            {"model": "Conformal prediction", "status": "prediction_sets", "purpose": "coverage bounds"},
            {"model": "GNN campaign model", "status": "adapter_contract", "purpose": "comparative research"},
            {"model": "Temporal graph network", "status": "adapter_contract", "purpose": "streaming graph research"},
            {"model": "Learned counterfactual", "status": "outcome_schema", "purpose": "future model fitting"},
        ]


class HyperdimensionalWorldMemory:
    def encode_relation(self, source: np.ndarray, relationship: np.ndarray, target: np.ndarray) -> np.ndarray:
        return np.sign(source * relationship * target)

    def episode(self, relations: Iterable[np.ndarray]) -> np.ndarray:
        values = list(relations)
        if not values:
            raise ValueError("episode requires relations")
        return np.sign(np.sum(values, axis=0))

    def uncertainty(self, probabilities: Iterable[float], ood: float, disagreement: float) -> dict[str, float]:
        probabilities = list(probabilities)
        aleatoric = 1 - max(probabilities, default=0.0)
        return {
            "aleatoric": aleatoric,
            "epistemic": min(1.0, disagreement * 0.7 + ood * 0.3),
            "ood": ood,
            "model_disagreement": disagreement,
        }

    def information_gain(self, prior: Iterable[float], posterior: Iterable[float]) -> float:
        def entropy(values: Iterable[float]) -> float:
            return -sum(value * math.log2(value) for value in values if value > 0)

        return entropy(prior) - entropy(posterior)


class AttackLaboratoryV2:
    def compare(self, versions: Mapping[str, list[Mapping[str, Any]]]) -> dict[str, Any]:
        reports = {}
        for version, events in versions.items():
            ordered = sorted(events, key=lambda item: float(item["timestamp"]))
            start = float(ordered[0]["timestamp"]) if ordered else 0.0
            understood = next((float(item["timestamp"]) for item in ordered if float(item.get("risk", 0)) >= 60), None)
            predicted = next((float(item["timestamp"]) for item in ordered if item.get("correct_prediction")), None)
            safe = next(
                (
                    float(item["timestamp"])
                    for item in ordered
                    if item.get("authority") and float(item.get("risk", 0)) >= 60
                ),
                None,
            )
            reports[version] = {
                "time_to_understand": understood - start if understood is not None else None,
                "time_to_predict": predicted - start if predicted is not None else None,
                "time_to_safe_action": safe - start if safe is not None else None,
                "counterfactual_error": sum(
                    abs(float(item.get("predicted_outcome", 0)) - float(item.get("actual_outcome", 0)))
                    for item in ordered
                )
                / max(1, len(ordered)),
            }
        return {"versions": reports}
