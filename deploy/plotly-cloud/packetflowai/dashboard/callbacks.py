"""Dash callback registration for live operational views."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from dash import Input, Output, html

from .data import DashboardData
from .figures import (
    assurance_figure,
    classification_figure,
    network_figure,
    risk_gauge,
    risk_history_figure,
    traffic_figure,
)


def _number(value: Any) -> str:
    numeric = float(value or 0)
    for suffix, divisor in (("M", 1_000_000), ("K", 1_000)):
        if abs(numeric) >= divisor:
            return f"{numeric / divisor:.1f}{suffix}"
    return f"{numeric:,.0f}"


def _empty(title: str, detail: str):
    return html.Div([html.Strong(title), html.Small(detail)], className="empty-state")


def _flow_rows(flows: list[dict[str, Any]]):
    if not flows:
        return [_empty("LISTENING FOR FLOWS", "Capture or replay traffic to populate the ledger.")]
    rows = [
        html.Div(
            [html.Span("ROUTE"), html.Span("PROTOCOL"), html.Span("VOLUME"), html.Span("STATE")],
            className="ledger-head",
        )
    ]
    for record in flows[:22]:
        flow = record.get("payload", {})
        source = f"{flow.get('source_ip', '?')}:{flow.get('source_port', '*')}"
        destination = f"{flow.get('destination_ip', '?')}:{flow.get('destination_port', '*')}"
        route = f"{source} → {destination}"
        rows.append(
            html.Article(
                [
                    html.Div(
                        [html.Strong(route), html.Small(str(flow.get("flow_id") or record.get("flow_id") or "unknown"))]
                    ),
                    html.Span(str(flow.get("protocol") or "--"), className="protocol-pill"),
                    html.Span(
                        f"{int(flow.get('byte_count', 0) or 0):,} B / {int(flow.get('packet_count', 0) or 0):,} pkt"
                    ),
                    html.Span(str(flow.get("state") or "ACTIVE"), className="state-pill"),
                ],
                className="ledger-row",
            )
        )
    return rows


def _incidents(alerts: list[dict[str, Any]]):
    if not alerts:
        return [_empty("NO POLICY ALERTS", "Alert-only mode is standing by.")]
    rows = []
    for record in alerts[:16]:
        alert = record.get("payload", {})
        level = int(alert.get("policy_level", 0) or 0)
        rows.append(
            html.Article(
                [
                    html.I(className="severity-bar"),
                    html.Div(
                        [
                            html.Strong(f"{alert.get('action', 'ALERT')} / LEVEL {level}"),
                            html.P(alert.get("reason") or "Policy threshold crossed."),
                            html.Small(
                                f"{alert.get('target', 'unknown target')} · {record.get('event_id', 'unknown event')}"
                            ),
                        ]
                    ),
                    html.Time(str(record.get("created_at", ""))[:19].replace("T", " ")),
                ],
                className=f"incident-row {'critical' if level >= 4 else ''}",
            )
        )
    return rows


def _evidence(decisions: list[dict[str, Any]]):
    latest = decisions[0].get("payload", {}) if decisions else {}
    evidence = latest.get("evidence", {})
    channels = [
        (
            "LOCAL CLASSIFIER",
            evidence.get("calibrated_confidence", evidence.get("classifier_confidence")),
            "Calibrated confidence",
        ),
        ("HDC PROTOTYPE", evidence.get("prototype_similarity"), "Cosine similarity"),
        ("ANOMALY MODEL", evidence.get("anomaly_score"), "Deviation score"),
        ("NIM REASONING", evidence.get("nim_reasoning_strength"), "Separate channel"),
    ]
    cards = []
    for title, value, note in channels:
        display = (
            "--" if value is None else f"{float(value) * 100:.1f}%" if 0 <= float(value) <= 1 else f"{float(value):.2f}"
        )
        cards.append(html.Article([html.Span(title), html.Strong(display), html.Small(note)]))
    if latest.get("explanation"):
        cards.append(
            html.Article(
                [html.Span("DECISION PROVENANCE"), html.P(latest["explanation"])], className="explanation-card"
            )
        )
    return cards


def _models(models: list[dict[str, Any]]):
    if not models:
        return [_empty("NO REGISTERED MODEL", "Train, evaluate, shadow, then promote.")]

    def status(model: dict[str, Any]) -> str:
        evaluated = "yes" if model.get("evaluated") else "no"
        shadow = "passed" if model.get("shadow_validated") else "pending"
        return f"Evaluated {evaluated} · Shadow {shadow}"

    return [
        html.Article(
            [
                html.Div(
                    [
                        html.Strong(f"{model.get('model_id')} / {model.get('version')}"),
                        html.Small(status(model)),
                    ]
                ),
                html.Span(str(model.get("state", "candidate")).upper(), className="model-state"),
            ]
        )
        for model in models
    ]


def _runtime(metrics: dict[str, Any]):
    values = [
        ("PACKETS / SEC", metrics.get("packets_per_second", 0), "pps"),
        ("FLOWS / SEC", metrics.get("flows_per_second", 0), "fps"),
        ("INFERENCE P95", metrics.get("inference_latency_p95_ms", 0), "ms"),
        ("QUEUE DEPTH", metrics.get("queue_depth", 0), "items"),
        ("DROPPED", metrics.get("dropped_packets", 0), "packets"),
        ("MEMORY", float(metrics.get("traced_memory_bytes", 0) or 0) / 1024**2, "MB"),
    ]
    return [
        html.Article([html.Span(label), html.Strong(f"{float(value or 0):,.1f}"), html.Small(unit)])
        for label, value, unit in values
    ]


def _assurance_summary(assurance: dict[str, Any]):
    contract = assurance.get("contract", {})
    return [
        html.Article([html.Span("ASSURANCE LEVEL"), html.Strong(str(assurance.get("assurance_level", "A0")))]),
        html.Article(
            [
                html.Span("UNKNOWN OMISSION RISK"),
                html.Strong(str(assurance.get("unknown_omission_risk", "NOT ELIMINATED")).replace("_", " ")),
            ]
        ),
        html.Article(
            [
                html.Span("SOURCE COVERAGE"),
                html.Strong(f"{assurance.get('observed_sources', 0)} / {assurance.get('expected_sources', 0)}"),
            ]
        ),
        html.Article([html.Span("EVIDENCE CONTRACT"), html.Strong(str(contract.get("contract_id", "UNBOUND")))]),
    ]


def _claims(assurance: dict[str, Any]):
    claims = assurance.get("formal_claims", [])
    if not claims:
        return [_empty("NO FORMAL CLAIMS", "Assurance claims will appear when evidence is sealed.")]
    return [
        html.Article(
            [
                html.Span(str(claim.get("id", "CLAIM"))),
                html.P(str(claim.get("statement", ""))),
                html.Strong(str(claim.get("status", "UNKNOWN"))),
            ]
        )
        for claim in claims
    ]


def register_callbacks(app, data: DashboardData) -> None:
    @app.callback(Output("signal-snapshot", "data"), Input("signal-interval", "n_intervals"))
    def refresh_snapshot(_: int):
        return data.snapshot()

    @app.callback(
        Output("system-state", "children"),
        Output("last-sync", "children"),
        Output("live-dot", "className"),
        Output("observatory-mode", "children"),
        Output("metric-flows", "children"),
        Output("metric-alerts", "children"),
        Output("metric-evidence", "children"),
        Output("metric-assurance", "children"),
        Output("network-figure", "figure"),
        Output("risk-figure", "figure"),
        Output("class-figure", "figure"),
        Output("history-figure", "figure"),
        Output("traffic-figure", "figure"),
        Output("flow-ledger", "children"),
        Output("incident-list", "children"),
        Output("evidence-grid", "children"),
        Output("model-stack", "children"),
        Output("runtime-grid", "children"),
        Output("assurance-figure", "figure"),
        Output("assurance-summary", "children"),
        Output("claim-ledger", "children"),
        Input("signal-snapshot", "data"),
    )
    def render(snapshot: dict[str, Any] | None):
        snapshot = snapshot or {}
        overview = snapshot.get("overview", {})
        counts = overview.get("counts", {})
        flows = snapshot.get("flows", [])
        alerts = snapshot.get("alerts", [])
        decisions = snapshot.get("decisions", [])
        assurance = snapshot.get("assurance", {})
        telemetry_mode = snapshot.get("status", {}).get("telemetry_mode", "network")
        refreshed = snapshot.get("refreshed_at")
        try:
            sync_label = f"SYNCED {datetime.fromisoformat(refreshed).astimezone().strftime('%H:%M:%S')}"
        except (TypeError, ValueError):
            sync_label = "AWAITING TELEMETRY"
        return (
            "HTTP LIVE"
            if telemetry_mode == "http-live"
            else "SYSTEM LIVE"
            if snapshot.get("health", {}).get("status") == "ok"
            else "INITIALISING",
            sync_label,
            "live-dot http-live" if telemetry_mode == "http-live" else "live-dot",
            "LIVE APPLICATION TRAFFIC" if telemetry_mode == "http-live" else "NETWORK OBSERVATORY",
            _number(counts.get("flows", 0)),
            _number(counts.get("alerts", 0)),
            _number(counts.get("evidence", 0)),
            str(assurance.get("assurance_level", "A0")),
            network_figure(flows, decisions),
            risk_gauge(overview.get("risk", {})),
            classification_figure(overview.get("classifications", {})),
            risk_history_figure(decisions),
            traffic_figure(flows),
            _flow_rows(flows),
            _incidents(alerts),
            _evidence(decisions),
            _models(snapshot.get("models", [])),
            _runtime(snapshot.get("metrics", {})),
            assurance_figure(assurance),
            _assurance_summary(assurance),
            _claims(assurance),
        )
