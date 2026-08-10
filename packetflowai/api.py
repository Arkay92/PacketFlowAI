"""Read-only operational HTTP API and minimal dashboard."""

# ruff: noqa: E501 - the embedded dashboard is intentionally kept as a single static asset.

import json
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from .config import AppConfig
from .registry import FilesystemModelRegistry
from .storage import EventStore
from .telemetry import MetricsRegistry

DASHBOARD_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PacketFlowAI Operations</title><style>
:root{--ink:#16231d;--paper:#f4f0e6;--signal:#d34b2f;--mint:#8dbda7}body{margin:0;background:radial-gradient(circle at 85% 10%,#d9e9de,transparent 35%),var(--paper);color:var(--ink);font:16px Georgia,serif}header{padding:3rem 6vw 2rem;border-bottom:3px solid var(--ink)}h1{font-size:clamp(2.5rem,7vw,6rem);margin:0;letter-spacing:-.06em}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:1rem;padding:2rem 6vw}.card{background:#fff9;border:1px solid #16231d33;padding:1.25rem;min-height:180px}.value{font:700 2rem ui-monospace,monospace;color:var(--signal)}pre{white-space:pre-wrap;font:12px ui-monospace,monospace}button{background:var(--ink);color:white;border:0;padding:.7rem 1rem}</style></head>
<body><header><p>LOCAL DETECTION / READ ONLY</p><h1>PacketFlowAI</h1><p>Flows, uncertainty, risk, agreement, and containment state.</p></header><main>
<section class="card"><h2>Health</h2><div id="health" class="value">...</div></section>
<section class="card"><h2>Metrics</h2><pre id="metrics">...</pre></section>
<section class="card"><h2>Recent alerts</h2><pre id="alerts">...</pre></section>
<section class="card"><h2>Active model</h2><pre id="models">...</pre></section></main>
<script>async function get(p){let r=await fetch(p);return r.json()}async function refresh(){
document.querySelector('#health').textContent=(await get('/health')).status;
document.querySelector('#metrics').textContent=JSON.stringify(await get('/metrics'),null,2);
document.querySelector('#alerts').textContent=JSON.stringify(await get('/alerts?limit=5'),null,2);
document.querySelector('#models').textContent=JSON.stringify(await get('/models'),null,2)}refresh();setInterval(refresh,5000)</script></body></html>"""


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
        self.server: ThreadingHTTPServer | None = None

    def _handler(self):
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                return

            def _write(self, payload: Any, status: int = 200, content_type: str = "application/json") -> None:
                body = payload.encode("utf-8") if isinstance(payload, str) else json.dumps(
                    _serializable(payload), sort_keys=True
                ).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                parsed = urlparse(self.path)
                limit = int(parse_qs(parsed.query).get("limit", ["100"])[0])
                try:
                    if parsed.path == "/":
                        self._write(DASHBOARD_HTML, content_type="text/html; charset=utf-8")
                    elif parsed.path == "/health":
                        self._write({"status": "ok"})
                    elif parsed.path == "/metrics":
                        self._write(outer.metrics.snapshot())
                    elif parsed.path == "/metrics/prometheus":
                        self._write(outer.metrics.prometheus(), content_type="text/plain; version=0.0.4")
                    elif parsed.path == "/flows":
                        self._write(outer.store.list("flows", limit))
                    elif parsed.path == "/alerts":
                        self._write(outer.store.list("alerts", limit))
                    elif parsed.path == "/feedback":
                        self._write(outer.store.list("feedback", limit))
                    elif parsed.path == "/models":
                        self._write(outer.registry.list_models())
                    elif parsed.path == "/status":
                        self._write({
                            "nim_mode": outer.config.nim.mode,
                            "artifact_root": str(outer.config.artifacts.root),
                        })
                    elif parsed.path == "/config":
                        self._write(asdict(outer.config))
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
