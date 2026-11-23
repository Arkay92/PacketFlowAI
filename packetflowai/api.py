"""Read-only operational API served beside the Dash Signal Room."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from flask import Response, jsonify, request, send_from_directory
from werkzeug.serving import BaseWSGIServer, make_server

from .assurance import AssuranceService
from .config import AppConfig
from .dashboard import create_dash_app
from .dashboard.data import DashboardData, serializable
from .intelligence import V3IntelligenceService
from .live_capture import HTTPFlowCapture
from .platform import PlatformIntelligenceService
from .registry import FilesystemModelRegistry
from .storage import EventStore
from .telemetry import MetricsRegistry

STATIC_ROOT = Path(__file__).with_name("static")


def _limit() -> int:
    return min(max(int(request.args.get("limit", "100")), 1), 1000)


def _json(payload: Any, status: int = 200):
    response = jsonify(serializable(payload))
    response.status_code = status
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


def register_api_routes(app, data: DashboardData) -> None:
    """Attach the stable operational API contract to a Dash Flask server."""
    server = app.server

    @server.errorhandler(ValueError)
    @server.errorhandler(RuntimeError)
    def operational_error(error):
        return _json({"error": str(error)}, 400)

    @server.route("/health", methods=["GET"])
    def health():
        return _json(data.health())

    @server.route("/overview", methods=["GET"])
    def overview():
        return _json(data.store.overview())

    @server.route("/metrics", methods=["GET"])
    def metrics():
        return _json(data.metrics_snapshot())

    @server.route("/metrics/prometheus", methods=["GET"])
    def prometheus():
        return Response(data.metrics.prometheus(), content_type="text/plain; version=0.0.4")

    def list_resource(table: str):
        return _json(data.store.list(table, _limit()))

    server.add_url_rule("/flows", "packetflow_flows", lambda: list_resource("flows"), methods=["GET"])
    server.add_url_rule("/alerts", "packetflow_alerts", lambda: list_resource("alerts"), methods=["GET"])
    server.add_url_rule("/decisions", "packetflow_decisions", lambda: list_resource("decisions"), methods=["GET"])
    server.add_url_rule("/evidence", "packetflow_evidence", lambda: list_resource("evidence"), methods=["GET"])
    server.add_url_rule("/nim", "packetflow_nim", lambda: list_resource("nim_assessments"), methods=["GET"])
    server.add_url_rule("/feedback", "packetflow_feedback", lambda: list_resource("feedback"), methods=["GET"])

    @server.route("/models", methods=["GET"])
    def models():
        return _json(data.registry.list_models())

    @server.route("/status", methods=["GET"])
    def status():
        return _json(data.status())

    @server.route("/config", methods=["GET"])
    def config():
        return _json(data.config_snapshot())

    @server.route("/v3/<path:resource>", methods=["GET"])
    def v3(resource: str):
        snapshot = V3IntelligenceService(data.store).snapshot()
        resources = {
            "overview": snapshot,
            "world-model": snapshot["world_model"],
            "campaigns": snapshot["campaigns"],
            "next-move": snapshot["predictions"],
            "simulations": snapshot["simulation"],
            "time-machine": snapshot["time_machine"],
            "integrity": snapshot["integrity"],
            "authority": snapshot["authority"],
            "disagreements": snapshot["disagreements"],
            "narrative": snapshot["narrative"],
            "playbook": snapshot["playbook"],
            "capabilities": snapshot["capabilities"],
        }
        return _json(resources.get(resource, {"error": "not found"}), 200 if resource in resources else 404)

    @server.route("/v4/<path:resource>", methods=["GET"])
    def v4(resource: str):
        snapshot = PlatformIntelligenceService(data.store).snapshot()
        resources = {
            "overview": snapshot,
            "predictions": snapshot["prediction_v2"],
            "causal": snapshot["causal_v2"],
            "intervention": snapshot["intervention"],
            "digital-twin": snapshot["digital_twin_v2"],
            "time-machine": snapshot["time_machine_v2"],
            "authority": snapshot["authority_v2"],
            "explainability": snapshot["explainability"],
            "runtime": snapshot["runtime_v2"],
            "domains": snapshot["platform_domains"],
        }
        return _json(resources.get(resource, {"error": "not found"}), 200 if resource in resources else 404)

    @server.route("/v5/<path:resource>", methods=["GET"])
    def v5(resource: str):
        snapshot = AssuranceService(data.store).snapshot()
        resources = {
            "overview": snapshot,
            "assurance": snapshot,
            "claims": snapshot["formal_claims"],
            "contract": snapshot["contract"],
            "continuity": snapshot["continuity_by_source"],
            "omissions": snapshot["omission_ledger"],
            "witnesses": snapshot["witness_reconciliation"],
            "observation-world": snapshot["observation_world"],
            "attack-lab": snapshot["attack_lab"],
            "assurance-report": {
                key: snapshot[key]
                for key in (
                    "integrity",
                    "inclusion",
                    "sequence_continuity",
                    "expected_sources",
                    "observed_sources",
                    "known_gaps",
                    "unknown_omission_risk",
                    "assurance_level",
                )
            },
        }
        return _json(resources.get(resource, {"error": "not found"}), 200 if resource in resources else 404)

    @server.route("/audit/v1/<path:resource>", methods=["GET"])
    def audit(resource: str):
        snapshot = AssuranceService(data.store).snapshot()
        resources = {
            "manifest": {"schema": "PFCASE-1.0", "algorithms": ["SHA-256", "PF-CANONICAL-JSON-1"]},
            "public-keys": {"active": ["assurance-policy-root"], "history": "published"},
            "schema": {"identifier": "https://packetflow.ai/spec/pfcase/1.0"},
            "inclusion-proof": {"status": "AVAILABLE_IN_PFCASE", "algorithm": "SHA-256"},
            "consistency-proof": snapshot["witness_reconciliation"],
            "checkpoint": snapshot["epoch_manifests"][0] if snapshot["epoch_manifests"] else {},
            "evidence-contract": snapshot["contract"],
        }
        return _json(resources.get(resource, {"error": "not found"}), 200 if resource in resources else 404)

    @server.route("/static/app.css", methods=["GET"])
    def legacy_css():
        return send_from_directory(str(STATIC_ROOT), "app.css", mimetype="text/css")

    @server.route("/static/app.js", methods=["GET"])
    def legacy_js():
        return send_from_directory(str(STATIC_ROOT), "app.js", mimetype="text/javascript")

    if data.telemetry_mode == "http-live":
        capture = HTTPFlowCapture(data.store, data.metrics)
        capture.register(server)
        server.extensions["packetflow_http_capture"] = capture


class APIServer:
    """Thread-friendly Dash/WSGI server retained for CLI and test compatibility."""

    def __init__(
        self,
        config: AppConfig,
        store: EventStore,
        metrics: MetricsRegistry,
        registry: FilesystemModelRegistry,
        host: str = "127.0.0.1",
        port: int = 8080,
        assets_folder: str | None = None,
    ) -> None:
        self.config = config
        self.store = store
        self.metrics = metrics
        self.registry = registry
        self.host = host
        self.port = port
        self.started_at = datetime.now(UTC)
        self.app, self.data = create_dash_app(config, store, metrics, registry, assets_folder=assets_folder)
        register_api_routes(self.app, self.data)
        self.server: BaseWSGIServer | None = None

    def serve_forever(self) -> None:
        self.server = make_server(self.host, self.port, self.app.server, threaded=True)
        self.server.serve_forever()

    def stop(self) -> None:
        if self.server:
            self.server.shutdown()
            self.server.server_close()
