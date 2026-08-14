"""Read-only operational HTTP API and Signal Room dashboard."""

import json
from dataclasses import asdict
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .config import AppConfig
from .intelligence import V3IntelligenceService
from .registry import FilesystemModelRegistry
from .storage import EventStore
from .telemetry import MetricsRegistry

STATIC_ROOT = Path(__file__).with_name("static")
STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/static/app.css": ("app.css", "text/css; charset=utf-8"),
    "/static/app.js": ("app.js", "text/javascript; charset=utf-8"),
}


def _serializable(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


class APIServer:
    def __init__(self, config: AppConfig, store: EventStore, metrics: MetricsRegistry,
                 registry: FilesystemModelRegistry, host: str = "127.0.0.1", port: int = 8080):
        self.config = config
        self.store = store
        self.metrics = metrics
        self.registry = registry
        self.host = host
        self.port = port
        self.started_at = datetime.now(UTC)
        self.server: ThreadingHTTPServer | None = None

    def _handler(self):
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                return

            def _write(self, payload: Any, status: int = 200, content_type: str = "application/json") -> None:
                body = payload if isinstance(payload, bytes) else (
                    payload.encode("utf-8") if isinstance(payload, str) else
                    json.dumps(_serializable(payload), sort_keys=True).encode("utf-8")
                )
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                self.wfile.write(body)

            def _serve_static(self, route: str) -> bool:
                static = STATIC_FILES.get(route)
                if static is None:
                    return False
                filename, content_type = static
                path = STATIC_ROOT / filename
                if not path.is_file():
                    self._write({"error": f"dashboard asset missing: {filename}"}, status=500)
                    return True
                self._write(path.read_bytes(), content_type=content_type)
                return True

            def do_GET(self) -> None:
                parsed = urlparse(self.path)
                if self._serve_static(parsed.path):
                    return
                try:
                    limit = min(max(int(parse_qs(parsed.query).get("limit", ["100"])[0]), 1), 1000)
                    if parsed.path == "/health":
                        self._write({
                            "status": "ok",
                            "service": "packetflowai",
                            "version": "3.0.0",
                            "uptime_seconds": (datetime.now(UTC) - outer.started_at).total_seconds(),
                        })
                    elif parsed.path == "/overview":
                        self._write(outer.store.overview())
                    elif parsed.path == "/metrics":
                        self._write({**outer.store.overview()["counts"], **outer.metrics.snapshot()})
                    elif parsed.path == "/metrics/prometheus":
                        self._write(outer.metrics.prometheus(), content_type="text/plain; version=0.0.4")
                    elif parsed.path == "/flows":
                        self._write(outer.store.list("flows", limit))
                    elif parsed.path == "/alerts":
                        self._write(outer.store.list("alerts", limit))
                    elif parsed.path == "/decisions":
                        self._write(outer.store.list("decisions", limit))
                    elif parsed.path == "/evidence":
                        self._write(outer.store.list("evidence", limit))
                    elif parsed.path == "/nim":
                        self._write(outer.store.list("nim_assessments", limit))
                    elif parsed.path == "/feedback":
                        self._write(outer.store.list("feedback", limit))
                    elif parsed.path == "/models":
                        self._write(outer.registry.list_models())
                    elif parsed.path == "/status":
                        self._write({
                            "nim_mode": outer.config.nim.mode,
                            "containment_enabled": False,
                            "artifact_root": str(outer.config.artifacts.root),
                            "encoder_seed": outer.config.model.encoder_seed,
                            "model_version": outer.config.model.model_version,
                        })
                    elif parsed.path == "/config":
                        self._write(asdict(outer.config))
                    elif parsed.path.startswith("/v3/"):
                        snapshot = V3IntelligenceService(outer.store).snapshot()
                        resources = {
                            "/v3/overview": snapshot,
                            "/v3/world-model": snapshot["world_model"],
                            "/v3/campaigns": snapshot["campaigns"],
                            "/v3/next-move": snapshot["predictions"],
                            "/v3/simulations": snapshot["simulation"],
                            "/v3/time-machine": snapshot["time_machine"],
                            "/v3/integrity": snapshot["integrity"],
                            "/v3/authority": snapshot["authority"],
                            "/v3/disagreements": snapshot["disagreements"],
                            "/v3/narrative": snapshot["narrative"],
                            "/v3/playbook": snapshot["playbook"],
                            "/v3/capabilities": snapshot["capabilities"],
                        }
                        resource = resources.get(parsed.path)
                        if resource is None:
                            self._write({"error": "not found"}, status=404)
                        else:
                            self._write(resource)
                    else:
                        self._write({"error": "not found"}, status=404)
                except (ValueError, RuntimeError) as error:
                    self._write({"error": str(error)}, status=400)

        return Handler

    def serve_forever(self) -> None:
        self.server = ThreadingHTTPServer((self.host, self.port), self._handler())
        self.server.serve_forever()

    def stop(self) -> None:
        if self.server:
            self.server.shutdown()
            self.server.server_close()
