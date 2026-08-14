"""Temporal cyber world model, campaign correlation, and causal graph construction."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

RELATIONSHIPS = {
    "SCANNED",
    "AUTHENTICATED_TO",
    "CONTACTED",
    "DOWNLOADED_FROM",
    "PRECEDED",
    "SHARES_INFRASTRUCTURE_WITH",
    "SIMILAR_TO",
    "TRIGGERED",
    "OBSERVED_ON",
    "USES_SERVICE",
    "MAPS_TO",
    "CONTAINS",
    "DEPENDS_ON",
    "TRUSTS",
}


def stable_id(kind: str, value: str) -> str:
    digest = hashlib.sha256(f"{kind}:{value}".encode()).hexdigest()[:12]
    return f"{kind.lower()}-{digest}"


@dataclass(frozen=True)
class WorldNode:
    node_id: str
    kind: str
    label: str
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WorldEdge:
    edge_id: str
    source: str
    target: str
    relationship: str
    observed_at: str
    confidence: float = 1.0
    evidence_ids: tuple[str, ...] = ()
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Campaign:
    campaign_id: str
    title: str
    event_ids: tuple[str, ...]
    node_ids: tuple[str, ...]
    techniques: tuple[str, ...]
    affected_hosts: tuple[str, ...]
    sources: tuple[str, ...]
    first_seen: str
    last_seen: str
    confidence: float
    summary: str


@dataclass(frozen=True)
class CausalLink:
    source_event: str
    target_event: str
    temporal_support: float
    network_support: float
    identity_support: float
    confidence: float
    alternative_explanations: tuple[str, ...]


class CyberWorldModel:
    def __init__(self) -> None:
        self.nodes: dict[str, WorldNode] = {}
        self.edges: dict[str, WorldEdge] = {}

    def add_node(self, kind: str, label: str, attributes: dict[str, Any] | None = None) -> WorldNode:
        node_id = stable_id(kind, label)
        merged = {**self.nodes.get(node_id, WorldNode(node_id, kind, label)).attributes, **(attributes or {})}
        node = WorldNode(node_id, kind, label, merged)
        self.nodes[node_id] = node
        return node

    def add_edge(
        self,
        source: WorldNode | str,
        target: WorldNode | str,
        relationship: str,
        observed_at: str,
        confidence: float = 1.0,
        evidence_ids: Iterable[str] = (),
        attributes: dict[str, Any] | None = None,
    ) -> WorldEdge:
        if relationship not in RELATIONSHIPS:
            raise ValueError(f"unsupported relationship: {relationship}")
        source_id = source.node_id if isinstance(source, WorldNode) else source
        target_id = target.node_id if isinstance(target, WorldNode) else target
        token = f"{source_id}:{target_id}:{relationship}:{observed_at}"
        edge_id = "edge-" + hashlib.sha256(token.encode()).hexdigest()[:16]
        edge = WorldEdge(
            edge_id,
            source_id,
            target_id,
            relationship,
            observed_at,
            max(0.0, min(1.0, confidence)),
            tuple(evidence_ids),
            attributes or {},
        )
        self.edges[edge_id] = edge
        return edge

    def neighbors(self, node_id: str, relationship: str | None = None) -> set[str]:
        result = set()
        for edge in self.edges.values():
            if relationship and edge.relationship != relationship:
                continue
            if edge.source == node_id:
                result.add(edge.target)
            if edge.target == node_id:
                result.add(edge.source)
        return result

    def impact(self, node_id: str, depth: int = 2) -> set[str]:
        reached = {node_id}
        frontier = {node_id}
        for _ in range(max(0, depth)):
            frontier = {neighbor for current in frontier for neighbor in self.neighbors(current)} - reached
            reached.update(frontier)
        return reached - {node_id}

    def serialize(self) -> dict[str, Any]:
        return {
            "nodes": [asdict(node) for node in self.nodes.values()],
            "edges": [asdict(edge) for edge in self.edges.values()],
            "counts": {
                "nodes": len(self.nodes),
                "edges": len(self.edges),
                "hosts": sum(node.kind == "HOST" for node in self.nodes.values()),
                "identities": sum(node.kind == "ACCOUNT" for node in self.nodes.values()),
                "services": sum(node.kind == "SERVICE" for node in self.nodes.values()),
            },
        }


class WorldModelBuilder:
    TECHNIQUES = {
        "port_scan": ("T1046",),
        "credential_attack": ("T1110",),
        "brute_force": ("T1110",),
        "data_exfiltration": ("T1041",),
        "ddos": ("T1498",),
        "malware": ("T1105",),
        "remote_services": ("T1021",),
    }

    def build(
        self,
        flows: list[dict[str, Any]],
        decisions: list[dict[str, Any]],
        alerts: list[dict[str, Any]],
    ) -> CyberWorldModel:
        model = CyberWorldModel()
        decision_by_event = {row.get("event_id"): row for row in decisions}
        alert_by_event = {row.get("event_id"): row for row in alerts}
        ordered_events: list[tuple[str, str, WorldNode]] = []
        for record in flows:
            flow = record.get("payload", {})
            event_id = str(flow.get("flow_id") or record.get("flow_id") or "unknown")
            observed_at = str(record.get("created_at") or datetime.now(UTC).isoformat())
            source = model.add_node("SOURCE", str(flow.get("source_ip", "unknown")), {"role": "origin"})
            target = model.add_node("HOST", str(flow.get("destination_ip", "unknown")), {"role": "target"})
            flow_node = model.add_node("FLOW", event_id, flow)
            service_label = f"{flow.get('destination_ip', 'unknown')}:{flow.get('destination_port', '*')}"
            service = model.add_node("SERVICE", service_label, {"protocol": flow.get("protocol")})
            metadata = flow.get("protocol_metadata") or {}
            account_name = metadata.get("account") or metadata.get("user") or metadata.get("username")
            model.add_edge(source, flow_node, "CONTACTED", observed_at, evidence_ids=(event_id,))
            model.add_edge(flow_node, target, "OBSERVED_ON", observed_at, evidence_ids=(event_id,))
            model.add_edge(flow_node, service, "USES_SERVICE", observed_at, evidence_ids=(event_id,))
            model.add_edge(target, service, "CONTAINS", observed_at, evidence_ids=(event_id,))
            if account_name:
                account = model.add_node("ACCOUNT", str(account_name))
                model.add_edge(account, target, "AUTHENTICATED_TO", observed_at, evidence_ids=(event_id,))

            decision_record = decision_by_event.get(event_id)
            if decision_record:
                decision = decision_record.get("payload", {})
                label = str(decision.get("evidence", {}).get("classifier_label", "unknown"))
                decision_node = model.add_node("ALERT", f"decision:{event_id}", decision)
                model.add_edge(flow_node, decision_node, "TRIGGERED", observed_at, evidence_ids=(event_id,))
                for technique_id in self.TECHNIQUES.get(label.lower(), ()):
                    technique = model.add_node("TECHNIQUE", technique_id, {"attack_family": label})
                    model.add_edge(decision_node, technique, "MAPS_TO", observed_at, evidence_ids=(event_id,))
                if label == "port_scan":
                    model.add_edge(source, target, "SCANNED", observed_at, .9, (event_id,))
                if label in {"malware", "data_exfiltration"}:
                    model.add_edge(target, source, "DOWNLOADED_FROM", observed_at, .65, (event_id,))
            if event_id in alert_by_event:
                alert_node = model.add_node("CASE", f"case:{event_id}", alert_by_event[event_id].get("payload", {}))
                model.add_edge(flow_node, alert_node, "TRIGGERED", observed_at, evidence_ids=(event_id,))
            ordered_events.append((observed_at, event_id, flow_node))

        ordered_events.sort()
        for previous, current in zip(ordered_events, ordered_events[1:], strict=False):
            model.add_edge(previous[2], current[2], "PRECEDED", current[0], .75, (previous[1], current[1]))
        self._add_shared_infrastructure(model)
        return model

    @staticmethod
    def _add_shared_infrastructure(model: CyberWorldModel) -> None:
        sources = [node for node in model.nodes.values() if node.kind == "SOURCE"]
        for index, left in enumerate(sources):
            for right in sources[index + 1:]:
                left_prefix = ".".join(left.label.split(".")[:3])
                right_prefix = ".".join(right.label.split(".")[:3])
                if left_prefix and left_prefix == right_prefix:
                    model.add_edge(left, right, "SHARES_INFRASTRUCTURE_WITH", datetime.now(UTC).isoformat(), .7)


class CampaignCorrelator:
    def correlate(self, model: CyberWorldModel) -> list[Campaign]:
        event_nodes = [node for node in model.nodes.values() if node.kind == "FLOW"]
        if not event_nodes:
            return []
        connected = self._connected_events(model, {node.node_id for node in event_nodes})
        campaigns = []
        for group in connected:
            nodes = [model.nodes[node_id] for node_id in group]
            event_ids = tuple(sorted(node.label for node in nodes if node.kind == "FLOW"))
            if not event_ids:
                continue
            related_ids = set(group)
            for node_id in tuple(group):
                related_ids.update(model.neighbors(node_id))
            related = [model.nodes[node_id] for node_id in related_ids if node_id in model.nodes]
            techniques = tuple(sorted(node.label for node in related if node.kind == "TECHNIQUE"))
            hosts = tuple(sorted(node.label for node in related if node.kind == "HOST"))
            sources = tuple(sorted(node.label for node in related if node.kind == "SOURCE"))
            timestamps = [
                edge.observed_at for edge in model.edges.values()
                if edge.source in related_ids or edge.target in related_ids
            ]
            campaign_id = stable_id("CAMPAIGN", "|".join(event_ids))
            confidence = min(.98, .48 + len(event_ids) * .04 + len(techniques) * .08)
            campaigns.append(Campaign(
                campaign_id,
                f"Campaign {campaign_id[-6:].upper()}",
                event_ids,
                tuple(sorted(related_ids)),
                techniques,
                hosts,
                sources,
                min(timestamps, default=""),
                max(timestamps, default=""),
                confidence,
                f"{len(event_ids)} events across {len(hosts)} hosts are probably one campaign.",
            ))
        return sorted(campaigns, key=lambda campaign: (-len(campaign.event_ids), campaign.campaign_id))

    @staticmethod
    def _connected_events(model: CyberWorldModel, event_ids: set[str]) -> list[set[str]]:
        remaining = set(event_ids)
        groups = []
        while remaining:
            root = remaining.pop()
            group = {root}
            frontier = {root}
            while frontier:
                current = frontier.pop()
                neighbors = model.neighbors(current)
                event_neighbors = {node for node in neighbors if node in event_ids and node in remaining}
                if not event_neighbors:
                    bridge_nodes = neighbors
                    event_neighbors = {
                        event for bridge in bridge_nodes for event in model.neighbors(bridge)
                        if event in event_ids and event in remaining
                    }
                remaining -= event_neighbors
                group |= event_neighbors
                frontier |= event_neighbors
            groups.append(group)
        return groups


class CausalGraphBuilder:
    def build(self, model: CyberWorldModel) -> list[CausalLink]:
        links = []
        for edge in model.edges.values():
            if edge.relationship != "PRECEDED":
                continue
            source_neighbors = model.neighbors(edge.source)
            target_neighbors = model.neighbors(edge.target)
            network_support = .8 if source_neighbors & target_neighbors else .35
            identity_support = (
                .8
                if any(
                    model.nodes[node].kind == "ACCOUNT"
                    for node in source_neighbors & target_neighbors
                )
                else 0.0
            )
            confidence = min(.95, .45 * edge.confidence + .4 * network_support + .15 * identity_support)
            links.append(CausalLink(
                model.nodes[edge.source].label,
                model.nodes[edge.target].label,
                edge.confidence,
                network_support,
                identity_support,
                confidence,
                ("shared maintenance window", "independent automated activity"),
            ))
        return links
