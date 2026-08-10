"""Bidirectional flow tracking, host context, and temporal HDC."""

import hashlib
import math
import statistics
from collections import Counter, deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .domain import FlowFeatures, PacketObservation
from .hdc import HypervectorEncoder


@dataclass(frozen=True, order=True)
class FlowKey:
    endpoint_a_ip: str
    endpoint_a_port: int
    endpoint_b_ip: str
    endpoint_b_port: int
    protocol: str

    @classmethod
    def from_observation(cls, packet: PacketObservation) -> tuple["FlowKey", bool]:
        source = (packet.source_ip, packet.source_port or 0)
        destination = (packet.destination_ip, packet.destination_port or 0)
        forward = source <= destination
        first, second = (source, destination) if forward else (destination, source)
        return cls(first[0], first[1], second[0], second[1], packet.protocol.upper()), forward

    @property
    def flow_id(self) -> str:
        identity = (
            f"{self.endpoint_a_ip}:{self.endpoint_a_port}|"
            f"{self.endpoint_b_ip}:{self.endpoint_b_port}|{self.protocol}"
        )
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]


@dataclass
class _HostEvent:
    timestamp: float
    destination_ip: str
    destination_port: int
    protocol: str
    outbound: bool
    failed: bool


class HostBehaviorTracker:
    def __init__(self, window_seconds: float = 300.0, max_events_per_host: int = 10_000):
        self.window_seconds = window_seconds
        self.max_events_per_host = max_events_per_host
        self._events: dict[str, deque[_HostEvent]] = {}

    def update(self, packet: PacketObservation, outbound: bool = True) -> None:
        failed = bool(packet.tcp_flags & 0x04)
        source_events = self._events.setdefault(packet.source_ip, deque(maxlen=self.max_events_per_host))
        source_events.append(_HostEvent(
            packet.timestamp,
            packet.destination_ip,
            packet.destination_port or 0,
            packet.protocol,
            True,
            failed,
        ))
        destination_events = self._events.setdefault(packet.destination_ip, deque(maxlen=self.max_events_per_host))
        destination_events.append(_HostEvent(
            packet.timestamp,
            packet.source_ip,
            packet.source_port or 0,
            packet.protocol,
            False,
            failed,
        ))
        self._prune(packet.source_ip, packet.timestamp)
        self._prune(packet.destination_ip, packet.timestamp)

    def _prune(self, host: str, now: float) -> None:
        events = self._events.get(host)
        if not events:
            return
        cutoff = now - self.window_seconds
        while events and events[0].timestamp < cutoff:
            events.popleft()
        if not events:
            self._events.pop(host, None)

    def snapshot(self, host: str, now: float) -> dict[str, float | int]:
        self._prune(host, now)
        events = self._events.get(host, ())
        if not events:
            return {
                "unique_destination_hosts": 0,
                "unique_destination_ports": 0,
                "host_connection_rate": 0.0,
                "host_failure_rate": 0.0,
                "protocol_entropy": 0.0,
                "outbound_ratio": 0.0,
            }
        counts = Counter(event.protocol for event in events)
        total = len(events)
        entropy = -sum((count / total) * math.log2(count / total) for count in counts.values())
        return {
            "unique_destination_hosts": len({event.destination_ip for event in events}),
            "unique_destination_ports": len({event.destination_port for event in events}),
            "host_connection_rate": total / self.window_seconds,
            "host_failure_rate": sum(event.failed for event in events) / total,
            "protocol_entropy": entropy,
            "outbound_ratio": sum(event.outbound for event in events) / total,
        }


@dataclass
class FlowState:
    key: FlowKey
    started_at: float
    last_seen_at: float
    initiator_ip: str
    packet_lengths: list[int] = field(default_factory=list)
    timestamps: list[float] = field(default_factory=list)
    forward_packets: int = 0
    reverse_packets: int = 0
    forward_bytes: int = 0
    reverse_bytes: int = 0
    syn_count: int = 0
    ack_count: int = 0
    fin_count: int = 0
    rst_count: int = 0
    retransmission_count: int = 0
    seen_sequences: set[tuple[bool, int]] = field(default_factory=set)
    state: str = "NEW"
    protocol_metadata: dict[str, Any] = field(default_factory=dict)
    event_tokens: list[str] = field(default_factory=list)

    def update(self, packet: PacketObservation, forward: bool) -> None:
        self.last_seen_at = max(self.last_seen_at, packet.timestamp)
        self.packet_lengths.append(packet.length)
        self.timestamps.append(packet.timestamp)
        if forward:
            self.forward_packets += 1
            self.forward_bytes += packet.length
        else:
            self.reverse_packets += 1
            self.reverse_bytes += packet.length
        flags = packet.tcp_flags
        self.syn_count += bool(flags & 0x02)
        self.ack_count += bool(flags & 0x10)
        self.fin_count += bool(flags & 0x01)
        self.rst_count += bool(flags & 0x04)
        if packet.tcp_sequence is not None:
            sequence_key = (forward, packet.tcp_sequence)
            if sequence_key in self.seen_sequences:
                self.retransmission_count += 1
            self.seen_sequences.add(sequence_key)
        if flags & 0x04:
            self.state = "RESET"
        elif self.fin_count >= 2:
            self.state = "CLOSED"
        elif flags & 0x01:
            self.state = "CLOSING"
        elif self.syn_count and self.ack_count:
            self.state = "ESTABLISHED"
        elif self.syn_count:
            self.state = "SYN_SEEN"
        else:
            self.state = "ACTIVE"
        direction = "F" if forward else "R"
        self.event_tokens.append(f"{direction}:{packet.protocol}:{flags}:{min(packet.length // 128, 15)}")
        for key, value in packet.metadata.items():
            if value not in {None, ""}:
                self.protocol_metadata[key] = value

    def close(self, reason: str = "CLOSED") -> None:
        self.state = reason


class FlowEngine:
    def __init__(self, idle_timeout_seconds: float = 60.0, max_flows: int = 100_000,
                 host_tracker: HostBehaviorTracker | None = None):
        self.idle_timeout_seconds = idle_timeout_seconds
        self.max_flows = max_flows
        self.host_tracker = host_tracker or HostBehaviorTracker()
        self._flows: dict[FlowKey, FlowState] = {}

    def update(self, packet: PacketObservation) -> FlowState:
        key, canonical_forward = FlowKey.from_observation(packet)
        state = self._flows.get(key)
        if state is None:
            if len(self._flows) >= self.max_flows:
                oldest = min(self._flows, key=lambda candidate: self._flows[candidate].last_seen_at)
                self._flows.pop(oldest)
            state = FlowState(key, packet.timestamp, packet.timestamp, packet.source_ip)
            self._flows[key] = state
        forward = packet.source_ip == state.initiator_ip
        state.update(packet, forward if state.forward_packets + state.reverse_packets else True)
        self.host_tracker.update(packet, outbound=True)
        return state

    def close(self, key: FlowKey, reason: str = "CLOSED") -> FlowFeatures | None:
        state = self._flows.pop(key, None)
        if state is None:
            return None
        state.close(reason)
        return self.features(state)

    def expire(self, now: float) -> list[FlowFeatures]:
        expired = [key for key, state in self._flows.items() if now - state.last_seen_at >= self.idle_timeout_seconds]
        return [result for key in sorted(expired) if (result := self.close(key, "TIMED_OUT")) is not None]

    def active(self) -> tuple[FlowState, ...]:
        return tuple(self._flows[key] for key in sorted(self._flows))

    def features(self, state: FlowState) -> FlowFeatures:
        lengths = state.packet_lengths
        arrivals = [
            later - earlier
            for earlier, later in zip(state.timestamps, state.timestamps[1:], strict=False)
        ]
        duration = max(0.0, state.last_seen_at - state.started_at)
        effective_duration = max(duration, 1e-9)
        mean_arrival = statistics.fmean(arrivals) if arrivals else 0.0
        std_arrival = statistics.pstdev(arrivals) if len(arrivals) > 1 else 0.0
        burstiness = std_arrival / mean_arrival if mean_arrival > 0 else 0.0
        host = self.host_tracker.snapshot(state.initiator_ip, state.last_seen_at)
        initiator_is_a = state.initiator_ip == state.key.endpoint_a_ip
        return FlowFeatures(
            flow_id=state.key.flow_id,
            source_ip=state.initiator_ip,
            destination_ip=state.key.endpoint_b_ip if initiator_is_a else state.key.endpoint_a_ip,
            source_port=state.key.endpoint_a_port if initiator_is_a else state.key.endpoint_b_port,
            destination_port=state.key.endpoint_b_port if initiator_is_a else state.key.endpoint_a_port,
            protocol=state.key.protocol,
            packet_count=len(lengths),
            byte_count=sum(lengths),
            duration_seconds=duration,
            packets_per_second=len(lengths) / effective_duration,
            bytes_per_second=sum(lengths) / effective_duration,
            forward_packets=state.forward_packets,
            reverse_packets=state.reverse_packets,
            forward_bytes=state.forward_bytes,
            reverse_bytes=state.reverse_bytes,
            packet_length_mean=statistics.fmean(lengths) if lengths else 0.0,
            packet_length_std=statistics.pstdev(lengths) if len(lengths) > 1 else 0.0,
            packet_length_min=min(lengths, default=0),
            packet_length_max=max(lengths, default=0),
            inter_arrival_mean=mean_arrival,
            inter_arrival_std=std_arrival,
            syn_count=state.syn_count,
            ack_count=state.ack_count,
            fin_count=state.fin_count,
            rst_count=state.rst_count,
            retransmission_count=state.retransmission_count,
            burstiness=burstiness,
            state=state.state,
            unique_destination_hosts=int(host["unique_destination_hosts"]),
            unique_destination_ports=int(host["unique_destination_ports"]),
            host_connection_rate=float(host["host_connection_rate"]),
            host_failure_rate=float(host["host_failure_rate"]),
            protocol_entropy=float(host["protocol_entropy"]),
            outbound_ratio=float(host["outbound_ratio"]),
            protocol_metadata=dict(state.protocol_metadata),
        )


class TemporalFlowEncoder:
    def __init__(self, encoder: HypervectorEncoder):
        self.encoder = encoder

    def encode(self, flow: FlowFeatures, event_tokens: Iterable[str] = ()) -> np.ndarray:
        scalars = {
            "duration": (flow.duration_seconds, 0, 3600),
            "packet_count": (flow.packet_count, 0, 10_000),
            "byte_count": (flow.byte_count, 0, 10_000_000),
            "forward_packets": (flow.forward_packets, 0, 10_000),
            "reverse_packets": (flow.reverse_packets, 0, 10_000),
            "syn_count": (flow.syn_count, 0, 1_000),
            "rst_count": (flow.rst_count, 0, 1_000),
            "unique_destination_ports": (flow.unique_destination_ports, 0, 65_535),
        }
        vectors = [self.encoder.encode_categorical("flow_protocol", flow.protocol)]
        vectors.extend(
            self.encoder.encode_numerical(name, value, minimum, maximum)
            for name, (value, minimum, maximum) in scalars.items()
        )
        tokens = [self.encoder.encode_categorical("flow_event", token) for token in event_tokens]
        if tokens:
            vectors.append(self.encoder.encode_sequence(tokens))
        return self.encoder.bundle(vectors)
