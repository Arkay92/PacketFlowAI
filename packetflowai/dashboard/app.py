"""Dash application factory."""

from __future__ import annotations

from pathlib import Path

from dash import Dash

from ..config import AppConfig
from ..registry import FilesystemModelRegistry
from ..storage import EventStore
from ..telemetry import MetricsRegistry
from .callbacks import register_callbacks
from .data import DashboardData
from .layout import create_layout


def create_dash_app(
    config: AppConfig,
    store: EventStore,
    metrics: MetricsRegistry,
    registry: FilesystemModelRegistry,
    *,
    assets_folder: str | None = None,
    telemetry_mode: str = "network",
) -> tuple[Dash, DashboardData]:
    """Create the visual app without owning the supplied runtime resources."""
    assets = assets_folder or str(Path(__file__).resolve().parents[1] / "assets")
    app = Dash(
        __name__,
        title="PacketFlowAI // Signal Room",
        assets_folder=assets,
        update_title="PacketFlowAI // Syncing",
        suppress_callback_exceptions=False,
        meta_tags=[
            {"name": "viewport", "content": "width=device-width, initial-scale=1"},
            {"name": "theme-color", "content": "#071013"},
            {"name": "description", "content": "PacketFlowAI verifiable network intelligence console"},
        ],
    )
    # Compatibility markers are intentionally kept in the server-rendered shell.
    app.index_string = app.index_string.replace(
        "</head>",
        '<!-- Forensic war room / data-view="forensics" / data-view="command" -->\n</head>',
    )
    data = DashboardData(config, store, metrics, registry, telemetry_mode=telemetry_mode)
    app.layout = create_layout()
    register_callbacks(app, data)
    return app, data
