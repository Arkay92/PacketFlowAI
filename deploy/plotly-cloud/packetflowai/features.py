"""Canonical packet feature schema and normalization helpers."""

import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

FEATURE_SCHEMA_VERSION = "packet-features-v1"

TCP_FLAG_BITS = {
    "F": 0x01,
    "S": 0x02,
    "R": 0x04,
    "P": 0x08,
    "A": 0x10,
    "U": 0x20,
    "E": 0x40,
    "C": 0x80,
}

SERVICE_PORTS = {
    "ftp": 21,
    "ssh": 22,
    "telnet": 23,
    "smtp": 25,
    "domain": 53,
    "dns": 53,
    "http": 80,
    "pop3": 110,
    "imap": 143,
    "https": 443,
}


def canonical_tcp_flags(value: Any) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            value = int(stripped)
        else:
            unknown = set(stripped.upper()) - set(TCP_FLAG_BITS)
            if unknown:
                raise ValueError(f"unknown TCP flag characters: {sorted(unknown)}")
            return sum(TCP_FLAG_BITS[flag] for flag in set(stripped.upper()))
    numeric = int(value)
    if not 0 <= numeric <= 255:
        raise ValueError(f"TCP flags outside 8-bit range: {numeric}")
    return numeric


def parse_port(value: Any) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in SERVICE_PORTS:
            return SERVICE_PORTS[normalized]
        if not normalized.isdigit():
            return None
        value = normalized
    port = int(value)
    if not 0 <= port <= 65535:
        raise ValueError(f"port outside valid range: {port}")
    return port


@dataclass(frozen=True)
class PacketFeatures:
    ip_version: int | None
    ip_len: int | None
    protocol: str | None
    tcp_sport: int | None = None
    tcp_dport: int | None = None
    tcp_flags: int | None = None
    udp_sport: int | None = None
    udp_dport: int | None = None

    def __post_init__(self) -> None:
        protocol = self.protocol.upper() if self.protocol else None
        object.__setattr__(self, "protocol", protocol)
        if protocol not in {None, "TCP", "UDP"}:
            raise ValueError(f"unsupported protocol: {protocol}")
        if self.ip_version not in {None, 4, 6}:
            raise ValueError(f"unsupported IP version: {self.ip_version}")
        if self.ip_len is not None and not 0 <= self.ip_len <= 65535:
            raise ValueError(f"IP length outside valid range: {self.ip_len}")
        for name in ("tcp_sport", "tcp_dport", "udp_sport", "udp_dport"):
            value = getattr(self, name)
            if value is not None and not 0 <= value <= 65535:
                raise ValueError(f"{name} outside valid range: {value}")

    def as_mapping(self) -> dict[str, Any]:
        return asdict(self)


_FIELD_PATTERNS = {
    "ip_version": re.compile(r"\bIP\s+version\s*:\s*(\d+)\b", re.IGNORECASE),
    "ip_len": re.compile(r"\bIP\s+len(?:gth)?\s*:\s*(\d+)\b", re.IGNORECASE),
    "tcp_sport": re.compile(r"\bTCP\s+sport\s*:\s*([^,;\s]+)", re.IGNORECASE),
    "tcp_dport": re.compile(r"\bTCP\s+dport\s*:\s*([^,;\s]+)", re.IGNORECASE),
    "tcp_flags": re.compile(r"\bTCP\s+flags\s*:\s*([^,;\s]+)", re.IGNORECASE),
    "udp_sport": re.compile(r"\bUDP\s+sport\s*:\s*([^,;\s]+)", re.IGNORECASE),
    "udp_dport": re.compile(r"\bUDP\s+dport\s*:\s*([^,;\s]+)", re.IGNORECASE),
}


def _match(description: str, field: str) -> str | None:
    match = _FIELD_PATTERNS[field].search(description)
    return match.group(1) if match else None


def packet_features_from_description(description: str) -> PacketFeatures:
    if not isinstance(description, str) or not description.strip():
        raise ValueError("packet description must be a non-empty string")
    has_tcp = any(_match(description, name) is not None for name in ("tcp_sport", "tcp_dport", "tcp_flags"))
    has_udp = any(_match(description, name) is not None for name in ("udp_sport", "udp_dport"))
    if has_tcp and has_udp:
        raise ValueError("packet description mixes TCP and UDP fields")
    ip_version = _match(description, "ip_version")
    ip_length = _match(description, "ip_len")
    return PacketFeatures(
        ip_version=int(ip_version) if ip_version is not None else None,
        ip_len=int(ip_length) if ip_length is not None else None,
        protocol="TCP" if has_tcp else "UDP" if has_udp else None,
        tcp_sport=parse_port(_match(description, "tcp_sport")),
        tcp_dport=parse_port(_match(description, "tcp_dport")),
        tcp_flags=canonical_tcp_flags(_match(description, "tcp_flags")),
        udp_sport=parse_port(_match(description, "udp_sport")),
        udp_dport=parse_port(_match(description, "udp_dport")),
    )


def packet_features_from_mapping(values: Mapping[str, Any]) -> PacketFeatures:
    return PacketFeatures(
        ip_version=int(values["ip_version"]) if values.get("ip_version") is not None else None,
        ip_len=int(values["ip_len"]) if values.get("ip_len") is not None else None,
        protocol=values.get("protocol"),
        tcp_sport=parse_port(values.get("tcp_sport")),
        tcp_dport=parse_port(values.get("tcp_dport")),
        tcp_flags=canonical_tcp_flags(values.get("tcp_flags")),
        udp_sport=parse_port(values.get("udp_sport")),
        udp_dport=parse_port(values.get("udp_dport")),
    )
