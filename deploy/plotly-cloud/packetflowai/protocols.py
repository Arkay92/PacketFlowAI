"""Scapy-to-observation parsing with optional payload-derived metadata."""

from typing import Any

from .domain import PacketObservation


def observation_from_scapy(packet: Any, include_payload_metadata: bool = False) -> PacketObservation:
    try:
        from scapy.layers.dns import DNS, DNSQR
        from scapy.layers.inet import IP, TCP, UDP
        from scapy.layers.inet6 import IPv6
        from scapy.packet import Raw
    except ImportError as error:
        raise RuntimeError("Scapy is required for packet parsing") from error
    network = packet[IP] if packet.haslayer(IP) else packet[IPv6] if packet.haslayer(IPv6) else None
    if network is None:
        raise ValueError("packet is not IPv4 or IPv6")
    protocol = "TCP" if packet.haslayer(TCP) else "UDP" if packet.haslayer(UDP) else None
    if protocol is None:
        raise ValueError("packet is not TCP or UDP")
    transport = packet[TCP] if protocol == "TCP" else packet[UDP]
    metadata: dict[str, Any] = {}
    if include_payload_metadata:
        if packet.haslayer(DNS) and packet.haslayer(DNSQR):
            query = packet[DNSQR].qname
            metadata["dns_query"] = (
                query.decode("utf-8", errors="replace").rstrip(".")
                if isinstance(query, bytes)
                else str(query)
            )
        if packet.haslayer(Raw):
            payload = bytes(packet[Raw].load)[:1024]
            text = payload.decode("utf-8", errors="ignore")
            first_line = text.splitlines()[0] if text.splitlines() else ""
            if first_line.startswith(("GET ", "POST ", "PUT ", "DELETE ", "HEAD ", "OPTIONS ")):
                metadata["http_request_line"] = first_line[:512]
            if payload.startswith(b"\x16\x03"):
                metadata["tls_record_version"] = payload[1:3].hex()
    timestamp = float(getattr(packet, "time", 0.0))
    length = int(getattr(network, "len", 0) or getattr(network, "plen", 0) or len(bytes(packet)))
    return PacketObservation(
        timestamp=timestamp,
        source_ip=str(network.src),
        destination_ip=str(network.dst),
        source_port=int(transport.sport),
        destination_port=int(transport.dport),
        protocol=protocol,
        length=length,
        tcp_flags=int(transport.flags) if protocol == "TCP" else 0,
        tcp_sequence=int(transport.seq) if protocol == "TCP" else None,
        metadata=metadata,
    )
