import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from packetflowai.api import register_api_routes
from packetflowai.config import AppConfig, ArtifactPaths
from packetflowai.dashboard import create_dash_app
from packetflowai.registry import FilesystemModelRegistry
from packetflowai.storage import EventStore
from packetflowai.telemetry import MetricsRegistry


def component_with_id(component, component_id):
    if isinstance(component, dict):
        props = component.get("props", {})
        if props.get("id") == component_id:
            return component
        return component_with_id(props.get("children"), component_id)
    if isinstance(component, list):
        for child in component:
            found = component_with_id(child, component_id)
            if found:
                return found
    return None


class DashboardLayoutTests(unittest.TestCase):
    def test_plotly_canvases_have_explicit_heights(self):
        with tempfile.TemporaryDirectory() as directory:
            config = replace(AppConfig(), artifacts=ArtifactPaths(Path(directory) / "artifacts"))
            config.artifacts.create()
            store = EventStore(config.artifacts.event_database)
            registry = FilesystemModelRegistry(config.artifacts.registry)
            app, _ = create_dash_app(config, store, MetricsRegistry(), registry)
            layout = json.loads(app.server.test_client().get("/_dash-layout").data)
            try:
                expected = {
                    "network-figure": "430px",
                    "risk-figure": "250px",
                    "class-figure": "250px",
                    "history-figure": "240px",
                    "traffic-figure": "240px",
                    "assurance-figure": "330px",
                }
                for component_id, height in expected.items():
                    component = component_with_id(layout, component_id)
                    self.assertIsNotNone(component, component_id)
                    self.assertEqual(component["props"]["style"]["height"], height)
            finally:
                store.close()

    def test_http_live_mode_captures_real_requests_in_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            config = replace(AppConfig(), artifacts=ArtifactPaths(Path(directory) / "artifacts"))
            config.artifacts.create()
            store = EventStore(Path(":memory:"))
            registry = FilesystemModelRegistry(config.artifacts.registry)
            app, data = create_dash_app(
                config,
                store,
                MetricsRegistry(),
                registry,
                telemetry_mode="http-live",
            )
            register_api_routes(app, data)
            client = app.server.test_client()
            try:
                self.assertEqual(client.get("/health").status_code, 200)
                overview = client.get("/overview").get_json()
                self.assertGreaterEqual(overview["counts"]["flows"], 1)
                flows = client.get("/flows?limit=10").get_json()
                self.assertTrue(flows)
                self.assertEqual(flows[0]["payload"]["protocol"], "HTTP")
                self.assertTrue(flows[0]["payload"]["source_ip"].startswith("192.0.2."))
                self.assertNotEqual(flows[0]["payload"]["source_ip"], "127.0.0.1")
                self.assertEqual(data.status()["telemetry_mode"], "http-live")
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
