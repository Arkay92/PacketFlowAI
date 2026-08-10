"""Canonical packet parsing and local inference service."""

import hashlib
from typing import Any

import torch

from .config import AppConfig
from .domain import FlowFeatures, LocalPrediction, ResponseDecision, ThreatAssessment
from .features import PacketFeatures, canonical_tcp_flags, packet_features_from_mapping
from .flows import TemporalFlowEncoder
from .hdc import HypervectorEncoder
from .mitre import DeterministicMitreMapper
from .policy import AlertOnlyPolicy, RiskTracker
from .taxonomy import ATTACK_TYPES


def _scapy_layers():
    try:
        from scapy.layers.inet import IP, TCP, UDP
    except ImportError as error:
        raise RuntimeError(
            "packet capture and replay require Scapy; install dependencies with 'pip install -r requirements.txt'"
        ) from error
    return IP, TCP, UDP


def packet_features_from_scapy(packet: Any) -> tuple[PacketFeatures, str, str]:
    IP, TCP, UDP = _scapy_layers()
    if not packet.haslayer(IP):
        raise ValueError("only IPv4 TCP/UDP packets are supported in the packet-level pipeline")
    ip_layer = packet[IP]
    values: dict[str, Any] = {
        "ip_version": ip_layer.version,
        "ip_len": ip_layer.len,
    }
    if packet.haslayer(TCP):
        tcp_layer = packet[TCP]
        values.update({
            "protocol": "TCP",
            "tcp_sport": tcp_layer.sport,
            "tcp_dport": tcp_layer.dport,
            "tcp_flags": canonical_tcp_flags(int(tcp_layer.flags)),
        })
    elif packet.haslayer(UDP):
        udp_layer = packet[UDP]
        values.update({
            "protocol": "UDP",
            "udp_sport": udp_layer.sport,
            "udp_dport": udp_layer.dport,
        })
    else:
        raise ValueError("only TCP and UDP packets are supported")
    packet_bytes = bytes(packet)
    return packet_features_from_mapping(values), hashlib.sha256(packet_bytes).hexdigest(), ip_layer.src


class PacketInferenceService:
    def __init__(self, config: AppConfig, encoder: HypervectorEncoder, model: torch.nn.Module,
                 device: torch.device, model_id: str, model_version: str,
                 risk_tracker: RiskTracker | None = None,
                 response_policy: AlertOnlyPolicy | None = None):
        self.config = config
        self.encoder = encoder
        self.model = model
        self.device = device
        self.model_id = model_id
        self.model_version = model_version
        self.risk_tracker = risk_tracker or RiskTracker(
            half_life_seconds=config.runtime.risk_half_life_seconds,
            allowlist=config.runtime.allowlist,
        )
        self.response_policy = response_policy or AlertOnlyPolicy()
        self.model.eval()

    def predict(self, features: PacketFeatures) -> LocalPrediction:
        vector = self.encoder.encode_packet(features)
        tensor = torch.tensor(vector, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probabilities = torch.softmax(self.model(tensor), dim=1).squeeze(0).cpu()
        label_index = int(torch.argmax(probabilities).item())
        return LocalPrediction(
            label=ATTACK_TYPES[label_index],
            label_index=label_index,
            confidence=float(probabilities[label_index].item()),
            scores=tuple(float(value) for value in probabilities.tolist()),
            model_id=self.model_id,
            model_version=self.model_version,
        )

    def process(self, packet: Any) -> tuple[ThreatAssessment, tuple[ResponseDecision, ...]]:
        features, packet_id, source_ip = packet_features_from_scapy(packet)
        prediction = self.predict(features)
        risk_score = self.risk_tracker.update_after_classification(source_ip, prediction.label)
        assessment = ThreatAssessment(
            event_id=packet_id,
            packet_features=features,
            local_prediction=prediction,
            risk_score=risk_score,
            source_ip=source_ip,
        )
        event = {
            "packet_id": packet_id,
            "source_ip": source_ip,
            "label": prediction.label,
            "risk_score": risk_score,
        }
        decisions = tuple(
            ResponseDecision(
                action=result["action"],
                reason=f"local classification: {prediction.label}",
                policy_level=0,
                reversible=True,
                requires_confirmation=False,
                executed=bool(result["executed"]),
            )
            for result in self.response_policy.respond(event)
        )
        return assessment, decisions


class FlowInferenceService:
    def __init__(self, config: AppConfig, encoder: HypervectorEncoder, model: torch.nn.Module,
                 device: torch.device, model_id: str, model_version: str):
        self.config = config
        self.encoder = encoder
        self.temporal_encoder = TemporalFlowEncoder(encoder)
        self.model = model
        self.device = device
        self.model_id = model_id
        self.model_version = model_version
        self.mitre = DeterministicMitreMapper()
        self.model.eval()

    def predict(self, flow: FlowFeatures, event_tokens: tuple[str, ...] = ()) -> LocalPrediction:
        vector = self.temporal_encoder.encode(flow, event_tokens)
        tensor = torch.tensor(vector, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probabilities = torch.softmax(self.model(tensor), dim=1).squeeze(0).cpu()
        index = int(probabilities.argmax().item())
        label = ATTACK_TYPES[index]
        mapping = self.mitre.map(label, flow.protocol_metadata)
        return LocalPrediction(
            label=label,
            label_index=index,
            confidence=float(probabilities[index]),
            scores=tuple(float(value) for value in probabilities.tolist()),
            model_id=self.model_id,
            model_version=self.model_version,
            mitre_techniques=mapping.techniques,
        )
