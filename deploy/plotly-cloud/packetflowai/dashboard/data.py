"""Read models used by the Dash visual layer."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any

from ..assurance import AssuranceService
from ..config import AppConfig
from ..registry import FilesystemModelRegistry
from ..storage import EventStore
from ..telemetry import MetricsRegistry


def serializable(value: Any) -> Any:
    """Return a Dash/JSON-safe copy of an operational value."""
    return json.loads(json.dumps(value, default=str))


class DashboardData:
    """Stable, read-only boundary between runtime state and the visual layer."""

    def __init__(
        self,
        config: AppConfig,
        store: EventStore,
        metrics: MetricsRegistry,
        registry: FilesystemModelRegistry,
        telemetry_mode: str = "network",
    ) -> None:
        self.config = config
        self.store = store
        self.metrics = metrics
        self.registry = registry
        self.telemetry_mode = telemetry_mode
        self.started_at = datetime.now(UTC)

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "service": "packetflowai",
            "version": "5.0.0",
            "uptime_seconds": (datetime.now(UTC) - self.started_at).total_seconds(),
        }

    def status(self) -> dict[str, Any]:
        return {
            "nim_mode": self.config.nim.mode,
            "containment_enabled": False,
            "artifact_root": str(self.config.artifacts.root),
            "encoder_seed": self.config.model.encoder_seed,
            "model_version": self.config.model.model_version,
            "telemetry_mode": self.telemetry_mode,
        }

    def metrics_snapshot(self) -> dict[str, Any]:
        return {**self.store.overview()["counts"], **self.metrics.snapshot()}

    def snapshot(self) -> dict[str, Any]:
        """Build the bounded payload refreshed by Dash callbacks."""
        return serializable(
            {
                "health": self.health(),
                "overview": self.store.overview(),
                "metrics": self.metrics_snapshot(),
                "status": self.status(),
                "flows": self.store.list("flows", 80),
                "alerts": self.store.list("alerts", 30),
                "decisions": self.store.list("decisions", 100),
                "evidence": self.store.list("evidence", 40),
                "models": self.registry.list_models(),
                "assurance": AssuranceService(self.store).snapshot(),
                "refreshed_at": datetime.now(UTC).isoformat(),
            }
        )

    def config_snapshot(self) -> dict[str, Any]:
        return serializable(asdict(self.config))
