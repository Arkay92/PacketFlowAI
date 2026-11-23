"""Approved read-only tools for bounded reasoning context."""

from collections.abc import Callable, Mapping
from typing import Any

from .mitre import DeterministicMitreMapper
from .storage import EventStore


class HistoricalFlowLookup:
    name = "historical_flow_lookup"

    def __init__(self, store: EventStore):
        self.store = store

    def query(self, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        limit = min(int(arguments.get("limit", 20)), 100)
        return {"flows": self.store.list("flows", limit)}


class ThreatIntelligenceLookup:
    name = "threat_intelligence_lookup"

    def __init__(self, lookup: Callable[[str], Mapping[str, Any]]):
        self.lookup = lookup

    def query(self, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        indicator = str(arguments.get("indicator", ""))[:256]
        return dict(self.lookup(indicator))


class DeterministicMitreLookup:
    name = "deterministic_mitre_lookup"

    def __init__(self):
        self.mapper = DeterministicMitreMapper()

    def query(self, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        mapping = self.mapper.map(str(arguments["attack_family"]), arguments.get("evidence", {}))
        return {
            "techniques": mapping.techniques,
            "mapping_version": mapping.mapping_version,
            "source": mapping.source,
            "rules": mapping.evidence_rules,
        }


class StaticAssetContext:
    name = "asset_context"

    def __init__(self, assets: Mapping[str, Mapping[str, Any]]):
        self.assets = dict(assets)

    def query(self, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        return dict(self.assets.get(str(arguments.get("asset_id", "")), {}))
