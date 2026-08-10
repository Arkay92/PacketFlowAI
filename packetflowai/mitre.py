"""Deterministic ATT&CK mapping with explicit provenance."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

MITRE_MAPPING_VERSION = "mitre-map-v1"

_FAMILY_TECHNIQUES = {
    "DDoS": ("T1498", "T1499"),
    "port_scan": ("T1046",),
    "malware": ("T1105", "T1204"),
    "phishing": ("T1566",),
    "other": (),
    "benign": (),
}


@dataclass(frozen=True)
class MitreMapping:
    techniques: tuple[str, ...]
    mapping_version: str
    source: str
    evidence_rules: tuple[str, ...]


class DeterministicMitreMapper:
    def map(self, attack_family: str, evidence: Mapping[str, Any] | None = None) -> MitreMapping:
        if attack_family not in _FAMILY_TECHNIQUES:
            raise ValueError(f"unsupported attack family: {attack_family}")
        techniques = list(_FAMILY_TECHNIQUES[attack_family])
        rules = [f"family:{attack_family}"]
        evidence = evidence or {}
        if evidence.get("credential_access") and "T1110" not in techniques:
            techniques.append("T1110")
            rules.append("evidence:credential_access")
        return MitreMapping(tuple(techniques), MITRE_MAPPING_VERSION, "deterministic", tuple(rules))

    def map_techniques(self, attack_family: str, evidence: Mapping[str, Any]) -> tuple[str, ...]:
        return self.map(attack_family, evidence).techniques
