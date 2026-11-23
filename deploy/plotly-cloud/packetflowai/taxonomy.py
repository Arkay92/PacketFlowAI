"""Versioned normalization for authoritative and explicitly weak labels."""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

TAXONOMY_VERSION = "attack-taxonomy-v1"
ATTACK_TYPES = ("benign", "DDoS", "port_scan", "malware", "phishing", "other")
LABEL_FIELDS = ("label", "Label", "attack_type", "Attack Type", "attack", "category", "class")

_ALIASES = {
    "benign": "benign",
    "normal": "benign",
    "ddos": "DDoS",
    "dos": "DDoS",
    "denial of service": "DDoS",
    "port scan": "port_scan",
    "port_scan": "port_scan",
    "scan": "port_scan",
    "reconnaissance": "port_scan",
    "fuzzers": "port_scan",
    "analysis": "port_scan",
    "backdoor": "malware",
    "backdoors": "malware",
    "bot": "malware",
    "bots": "malware",
    "shellcode": "malware",
    "worms": "malware",
    "exploits": "other",
    "generic": "other",
    "infiltration": "other",
    "web attack": "other",
    "brute force": "other",
    "malware": "malware",
    "phishing": "phishing",
    "other": "other",
}


class ThreatStage(StrEnum):
    BENIGN = "benign"
    MALICIOUS = "malicious"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class TaxonomyResult:
    stage: ThreatStage
    attack_family: str | None
    taxonomy_version: str = TAXONOMY_VERSION


def classify_stage(label: str, is_unknown: bool = False) -> TaxonomyResult:
    if is_unknown:
        return TaxonomyResult(ThreatStage.UNKNOWN, None)
    if label == "benign":
        return TaxonomyResult(ThreatStage.BENIGN, "benign")
    if label not in ATTACK_TYPES:
        raise ValueError(f"label is outside the internal taxonomy: {label!r}")
    return TaxonomyResult(ThreatStage.MALICIOUS, label)


@dataclass(frozen=True)
class NormalizedLabel:
    native_value: str
    normalized_value: str
    source_field: str
    provenance: str

    @property
    def index(self) -> int:
        return ATTACK_TYPES.index(self.normalized_value)


def normalize_label(value: Any) -> str:
    if isinstance(value, bool):
        return "other" if value else "benign"
    if isinstance(value, int) and value in {0, 1}:
        return "benign" if value == 0 else "other"
    normalized = str(value).strip().lower().replace("-", " ")
    if normalized in _ALIASES:
        return _ALIASES[normalized]
    if "ddos" in normalized or "dos" in normalized:
        return "DDoS"
    if "scan" in normalized or "recon" in normalized:
        return "port_scan"
    if any(token in normalized for token in ("bot", "malware", "worm", "shellcode", "backdoor")):
        return "malware"
    if "phish" in normalized:
        return "phishing"
    if normalized:
        return "other"
    raise ValueError(f"unmapped dataset label: {value!r}")


def extract_authoritative_label(row: Mapping[str, Any]) -> NormalizedLabel:
    for field in LABEL_FIELDS:
        if field in row and row[field] not in {None, ""}:
            native = row[field]
            return NormalizedLabel(str(native), normalize_label(native), field, "dataset-native")
    raise ValueError(
        "dataset row has no authoritative label field; weak labels require a separate, explicitly marked experiment"
    )
