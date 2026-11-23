"""Dash component tree for the PacketFlowAI Signal Room."""

from __future__ import annotations

from dash import dcc, html

from .figures import empty_figure

GRAPH_CONFIG = {"displayModeBar": False, "responsive": True, "scrollZoom": True}


def _panel(index: str, title: str, child, class_name: str = ""):
    return html.Section(
        [
            html.Header([html.Span(index, className="panel-index"), html.H2(title)]),
            child,
        ],
        className=f"signal-panel {class_name}".strip(),
    )


def _metric(label: str, component_id: str, note: str):
    return html.Article(
        [html.Span(label), html.Strong("0", id=component_id), html.Small(note)], className="metric-card"
    )


def create_layout():
    return html.Div(
        [
            dcc.Store(id="signal-snapshot"),
            dcc.Interval(id="signal-interval", interval=4_000, n_intervals=0),
            html.Header(
                [
                    html.Div(
                        [
                            html.Span("PF", className="brand-mark"),
                            html.Div([html.Strong("PACKETFLOW"), html.Small("SIGNAL ROOM / V5")]),
                        ],
                        className="brand",
                    ),
                    html.Div(
                        [
                            html.Span(className="live-dot", id="live-dot"),
                            html.Div(
                                [
                                    html.Strong("SYSTEM LIVE", id="system-state"),
                                    html.Small("Initialising", id="last-sync"),
                                ]
                            ),
                        ],
                        className="live-state",
                    ),
                ],
                className="topbar",
            ),
            html.Main(
                [
                    html.Section(
                        [
                            html.Div(
                                [
                                    html.Span("NETWORK OBSERVATORY", className="eyebrow", id="observatory-mode"),
                                    html.H1([html.Span("SEE THE SHAPE OF"), html.Em("EVERY CONVERSATION.")]),
                                ]
                            ),
                            html.Div(
                                [
                                    _metric("FLOWS OBSERVED", "metric-flows", "Bounded telemetry"),
                                    _metric("OPEN ALERTS", "metric-alerts", "Policy crossings"),
                                    _metric("EVIDENCE POINTS", "metric-evidence", "Preserved channels"),
                                    _metric("ASSURANCE", "metric-assurance", "Current PFCASE level"),
                                ],
                                className="metric-rack",
                            ),
                        ],
                        className="mission-strip",
                    ),
                    dcc.Tabs(
                        id="workspace-tabs",
                        value="overview",
                        className="workspace-tabs",
                        children=[
                            dcc.Tab(
                                label="OVERVIEW",
                                value="overview",
                                className="workspace-tab",
                                selected_className="workspace-tab is-active",
                                children=html.Div(
                                    [
                                        _panel(
                                            "01",
                                            "Network field",
                                            dcc.Graph(
                                                id="network-figure",
                                                figure=empty_figure("INITIALISING", height=430),
                                                config=GRAPH_CONFIG,
                                                style={"height": "430px"},
                                            ),
                                            "network-panel",
                                        ),
                                        _panel(
                                            "02",
                                            "Threat pressure",
                                            dcc.Graph(
                                                id="risk-figure",
                                                figure=empty_figure("INITIALISING", height=250),
                                                config=GRAPH_CONFIG,
                                                style={"height": "250px"},
                                            ),
                                            "risk-panel",
                                        ),
                                        _panel(
                                            "03",
                                            "Classification mix",
                                            dcc.Graph(
                                                id="class-figure",
                                                figure=empty_figure("INITIALISING", height=250),
                                                config=GRAPH_CONFIG,
                                                style={"height": "250px"},
                                            ),
                                            "class-panel",
                                        ),
                                        _panel(
                                            "04",
                                            "Risk chronology",
                                            dcc.Graph(
                                                id="history-figure",
                                                figure=empty_figure("INITIALISING", height=240),
                                                config=GRAPH_CONFIG,
                                                style={"height": "240px"},
                                            ),
                                            "history-panel",
                                        ),
                                        _panel(
                                            "05",
                                            "Protocol volume",
                                            dcc.Graph(
                                                id="traffic-figure",
                                                figure=empty_figure("INITIALISING", height=240),
                                                config=GRAPH_CONFIG,
                                                style={"height": "240px"},
                                            ),
                                            "traffic-panel",
                                        ),
                                    ],
                                    className="overview-grid",
                                    **{"data-view": "overview"},
                                ),
                            ),
                            dcc.Tab(
                                label="TRAFFIC",
                                value="traffic",
                                className="workspace-tab",
                                selected_className="workspace-tab is-active",
                                children=html.Div(
                                    [
                                        _panel(
                                            "06",
                                            "Flow ledger",
                                            html.Div(id="flow-ledger", className="data-ledger"),
                                            "wide-panel",
                                        ),
                                        _panel(
                                            "07",
                                            "Runtime telemetry",
                                            html.Div(id="runtime-grid", className="runtime-grid"),
                                        ),
                                    ],
                                    className="detail-grid",
                                    **{"data-view": "traffic"},
                                ),
                            ),
                            dcc.Tab(
                                label="DETECTION",
                                value="detection",
                                className="workspace-tab",
                                selected_className="workspace-tab is-active",
                                children=html.Div(
                                    [
                                        _panel(
                                            "08",
                                            "Incident rail",
                                            html.Div(id="incident-list", className="incident-list"),
                                            "wide-panel",
                                        ),
                                        _panel(
                                            "09",
                                            "Evidence channels",
                                            html.Div(id="evidence-grid", className="evidence-grid"),
                                        ),
                                        _panel(
                                            "10", "Model registry", html.Div(id="model-stack", className="model-stack")
                                        ),
                                    ],
                                    className="detail-grid",
                                    **{"data-view": "forensics"},
                                ),
                            ),
                            dcc.Tab(
                                label="ASSURANCE",
                                value="assurance",
                                className="workspace-tab",
                                selected_className="workspace-tab is-active",
                                children=html.Div(
                                    [
                                        _panel(
                                            "11",
                                            "Assurance posture",
                                            dcc.Graph(
                                                id="assurance-figure",
                                                figure=empty_figure("INITIALISING", height=330),
                                                config=GRAPH_CONFIG,
                                                style={"height": "330px"},
                                            ),
                                        ),
                                        _panel(
                                            "12",
                                            "Evidence contract",
                                            html.Div(id="assurance-summary", className="assurance-summary"),
                                        ),
                                        _panel(
                                            "13",
                                            "Formal claims",
                                            html.Div(id="claim-ledger", className="claim-ledger"),
                                            "wide-panel",
                                        ),
                                    ],
                                    className="detail-grid assurance-workspace",
                                    **{"data-view": "assurance"},
                                ),
                            ),
                        ],
                    ),
                ],
                className="dashboard-shell",
                id="dashboard",
                **{"data-view": "command"},
            ),
            html.Footer(
                [
                    html.Span("PACKETFLOWAI / READ-ONLY OPERATIONS"),
                    html.Span("CONTAINMENT DISABLED · HUMAN AUTHORITY PRESERVED"),
                    html.Span("PF-CANONICAL-JSON-1"),
                ],
                className="footer",
            ),
            html.Span("Forensic war room", className="sr-only"),
        ],
        className="app-shell",
    )
