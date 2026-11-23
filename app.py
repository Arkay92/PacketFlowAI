"""WSGI and local entry point for the PacketFlowAI Dash console."""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

from packetflowai.api import register_api_routes
from packetflowai.config import AppConfig, ArtifactPaths, writable_artifact_root
from packetflowai.dashboard import create_dash_app
from packetflowai.registry import FilesystemModelRegistry
from packetflowai.storage import EventStore
from packetflowai.telemetry import MetricsRegistry

config = AppConfig.from_env()
using_temporary_artifacts = False
if "PACKETFLOWAI_ARTIFACT_DIR" not in os.environ:
    selected_root = writable_artifact_root(Path("artifacts"))
    using_temporary_artifacts = selected_root.resolve() != Path("artifacts").resolve()
    config = replace(config, artifacts=ArtifactPaths(selected_root))
config.artifacts.create()
database_override = os.getenv("PACKETFLOWAI_EVENT_DATABASE")
database_path = (
    Path(database_override)
    if database_override
    else Path(":memory:")
    if using_temporary_artifacts
    else config.artifacts.event_database
)
store = EventStore(database_path)
registry = FilesystemModelRegistry(config.artifacts.registry)
metrics = MetricsRegistry()
app, dashboard_data = create_dash_app(config, store, metrics, registry, telemetry_mode="http-live")
register_api_routes(app, dashboard_data)
server = app.server


if __name__ == "__main__":
    app.run(host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "8050")), debug=False)
