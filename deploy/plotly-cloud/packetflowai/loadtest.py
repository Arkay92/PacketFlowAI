"""Synthetic deterministic flow-engine load harness."""

import time
from dataclasses import dataclass

from .domain import PacketObservation
from .flows import FlowEngine


@dataclass(frozen=True)
class LoadTestReport:
    flows: int
    packets: int
    elapsed_seconds: float
    packets_per_second: float
    flows_per_second: float


def run_load_test(flow_count: int = 10_000, packets_per_flow: int = 4) -> LoadTestReport:
    if flow_count <= 0 or packets_per_flow <= 0:
        raise ValueError("load-test dimensions must be positive")
    engine = FlowEngine(max_flows=flow_count + 1)
    started = time.perf_counter()
    packet_count = 0
    for flow_index in range(flow_count):
        source = f"10.{(flow_index >> 16) & 255}.{(flow_index >> 8) & 255}.{flow_index & 255}"
        for packet_index in range(packets_per_flow):
            engine.update(PacketObservation(
                timestamp=float(packet_index),
                source_ip=source,
                destination_ip="198.51.100.1",
                source_port=1024 + flow_index % 50_000,
                destination_port=443,
                protocol="TCP",
                length=64 + packet_index,
                tcp_flags=0x02 if packet_index == 0 else 0x10,
                tcp_sequence=packet_index,
            ))
            packet_count += 1
    elapsed = time.perf_counter() - started
    return LoadTestReport(flow_count, packet_count, elapsed, packet_count / elapsed, flow_count / elapsed)
