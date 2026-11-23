"""Plotly figure builders for the Signal Room."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from typing import Any

import plotly.graph_objects as go

BG = "rgba(0,0,0,0)"
GRID = "rgba(120, 151, 165, .12)"
TEXT = "#9fb1b8"
CYAN = "#50d8d7"
MINT = "#7de2b8"
AMBER = "#f2bd6d"
RED = "#ff6b6b"
SLATE = "#25333a"


def _base(figure: go.Figure, *, height: int = 300) -> go.Figure:
    figure.update_layout(
        height=height,
        margin=dict(l=18, r=18, t=18, b=18),
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(family="IBM Plex Mono, monospace", color=TEXT, size=11),
        hoverlabel=dict(bgcolor="#111a1f", bordercolor=CYAN, font_color="#e7f1f2"),
        showlegend=False,
        uirevision="packetflow",
    )
    return figure


def empty_figure(message: str, *, height: int = 300) -> go.Figure:
    figure = go.Figure()
    figure.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(color=TEXT, size=12),
    )
    figure.update_xaxes(visible=False)
    figure.update_yaxes(visible=False)
    return _base(figure, height=height)


def _position(value: str, index: int, total: int) -> tuple[float, float]:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    angle = (int.from_bytes(digest[:4], "big") / 2**32) * math.tau
    ring = 0.44 + (digest[4] / 255) * 0.42
    angle += index / max(total, 1) * 0.28
    return math.cos(angle) * ring, math.sin(angle) * ring


def network_figure(flows: list[dict[str, Any]], decisions: list[dict[str, Any]]) -> go.Figure:
    if not flows:
        return empty_figure("AWAITING FLOW TELEMETRY", height=430)

    risk_by_event = {
        str(item.get("event_id", "")): float(item.get("payload", {}).get("risk_score", 0) or 0) for item in decisions
    }
    nodes: dict[str, tuple[float, float]] = {}
    endpoints: list[str] = []
    edges: list[tuple[str, str, float, dict[str, Any]]] = []
    for record in reversed(flows[:55]):
        flow = record.get("payload", {})
        source = str(flow.get("source_ip") or "unknown-source")
        target = str(flow.get("destination_ip") or "unknown-target")
        if source not in endpoints:
            endpoints.append(source)
        if target not in endpoints:
            endpoints.append(target)
        event_id = str(record.get("flow_id") or flow.get("flow_id") or "")
        edges.append((source, target, risk_by_event.get(event_id, 0), flow))
    for index, endpoint in enumerate(endpoints):
        nodes[endpoint] = _position(endpoint, index, len(endpoints))

    figure = go.Figure()
    for source, target, risk, flow in edges:
        x0, y0 = nodes[source]
        x1, y1 = nodes[target]
        color = RED if risk >= 65 else AMBER if risk >= 35 else "rgba(80,216,215,.26)"
        protocol = flow.get("protocol", "--")
        byte_count = int(flow.get("byte_count", 0) or 0)
        hover = f"{source} → {target}<br>{protocol} · {byte_count:,} B<br>Risk {risk:.1f}"
        figure.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line=dict(color=color, width=1 + min(risk / 55, 1.8)),
                hoverinfo="text",
                text=hover,
            )
        )

    degrees: defaultdict[str, int] = defaultdict(int)
    for source, target, _, _ in edges:
        degrees[source] += 1
        degrees[target] += 1
    figure.add_trace(
        go.Scatter(
            x=[nodes[node][0] for node in endpoints],
            y=[nodes[node][1] for node in endpoints],
            mode="markers+text",
            text=[node if degrees[node] >= 2 else "" for node in endpoints],
            textposition="top center",
            textfont=dict(size=9, color="#bed0d5"),
            marker=dict(
                size=[8 + min(degrees[node] * 2, 16) for node in endpoints],
                color=[CYAN if index % 3 else MINT for index, _ in enumerate(endpoints)],
                line=dict(color="#071013", width=2),
            ),
            customdata=[[node, degrees[node]] for node in endpoints],
            hovertemplate="%{customdata[0]}<br>%{customdata[1]} linked flows<extra></extra>",
        )
    )
    figure.update_xaxes(visible=False, range=[-1.08, 1.08])
    figure.update_yaxes(visible=False, range=[-1.08, 1.08], scaleanchor="x", scaleratio=1)
    return _base(figure, height=430)


def risk_gauge(risk: dict[str, Any]) -> go.Figure:
    current = max(0, min(100, float(risk.get("current", 0) or 0)))
    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=current,
            number=dict(font=dict(color="#ecf6f5", size=44), suffix=""),
            title=dict(text="FUSED RISK / 100", font=dict(color=TEXT, size=11)),
            gauge=dict(
                axis=dict(range=[0, 100], tickwidth=0, tickcolor=BG, tickfont=dict(color=TEXT, size=9)),
                bar=dict(color=RED if current >= 65 else AMBER if current >= 35 else MINT, thickness=0.22),
                bgcolor=SLATE,
                borderwidth=0,
                steps=[
                    dict(range=[0, 35], color="rgba(125,226,184,.08)"),
                    dict(range=[35, 65], color="rgba(242,189,109,.08)"),
                    dict(range=[65, 100], color="rgba(255,107,107,.08)"),
                ],
                threshold=dict(line=dict(color="#eef7f7", width=2), thickness=0.65, value=current),
            ),
        )
    )
    _base(figure, height=250)
    figure.update_layout(margin=dict(l=18, r=18, t=42, b=8))
    return figure


def classification_figure(classifications: dict[str, int]) -> go.Figure:
    if not classifications:
        return empty_figure("NO CLASSIFICATIONS", height=250)
    labels, values = zip(*sorted(classifications.items(), key=lambda item: item[1], reverse=True), strict=True)
    colors = [MINT if label.lower() == "benign" else AMBER if label.lower() == "unknown" else RED for label in labels]
    figure = go.Figure(
        go.Pie(
            labels=[label.replace("_", " ").upper() for label in labels],
            values=values,
            hole=0.68,
            marker=dict(colors=colors, line=dict(color="#0a1115", width=3)),
            textinfo="none",
            hovertemplate="%{label}<br>%{value:,} decisions · %{percent}<extra></extra>",
        )
    )
    figure.add_annotation(
        text=f"{sum(values):,}<br><span style='font-size:9px'>DECISIONS</span>",
        showarrow=False,
        font=dict(color="#e7f1f2", size=18),
    )
    return _base(figure, height=250)


def risk_history_figure(decisions: list[dict[str, Any]]) -> go.Figure:
    items = list(reversed(decisions[:60]))
    if not items:
        return empty_figure("NO DECISION HISTORY", height=240)
    x = [item.get("created_at", "") for item in items]
    y = [float(item.get("payload", {}).get("risk_score", 0) or 0) for item in items]
    figure = go.Figure(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            line=dict(color=CYAN, width=2, shape="spline"),
            fill="tozeroy",
            fillcolor="rgba(80,216,215,.08)",
            hovertemplate="%{x}<br>Risk %{y:.1f}<extra></extra>",
        )
    )
    figure.add_hrect(y0=65, y1=100, fillcolor="rgba(255,107,107,.06)", line_width=0)
    figure.update_xaxes(showgrid=False, zeroline=False, tickfont=dict(size=9), title=None)
    figure.update_yaxes(range=[0, 100], gridcolor=GRID, zeroline=False, tickfont=dict(size=9))
    return _base(figure, height=240)


def traffic_figure(flows: list[dict[str, Any]]) -> go.Figure:
    if not flows:
        return empty_figure("NO TRAFFIC VOLUME", height=240)
    by_protocol: defaultdict[str, int] = defaultdict(int)
    for record in flows:
        flow = record.get("payload", {})
        by_protocol[str(flow.get("protocol") or "OTHER").upper()] += int(flow.get("byte_count", 0) or 0)
    ordered = sorted(by_protocol.items(), key=lambda item: item[1])[-8:]
    figure = go.Figure(
        go.Bar(
            x=[value for _, value in ordered],
            y=[label for label, _ in ordered],
            orientation="h",
            marker=dict(color=[CYAN if index == len(ordered) - 1 else "#35515a" for index in range(len(ordered))]),
            hovertemplate="%{y}<br>%{x:,} bytes<extra></extra>",
        )
    )
    figure.update_xaxes(gridcolor=GRID, zeroline=False, tickfont=dict(size=9))
    figure.update_yaxes(showgrid=False, tickfont=dict(size=9))
    return _base(figure, height=240)


def assurance_figure(assurance: dict[str, Any]) -> go.Figure:
    values = [
        1 if assurance.get("integrity") == "VERIFIED" else 0.25,
        1 if assurance.get("inclusion") == "VERIFIED" else 0.25,
        1 if assurance.get("sequence_continuity") == "VERIFIED" else 0.25,
        min(
            1,
            float(assurance.get("observed_sources", 0) or 0) / max(float(assurance.get("expected_sources", 1) or 1), 1),
        ),
        1 if assurance.get("witness_reconciliation", {}).get("status") in {"CONSISTENT", "VERIFIED"} else 0.25,
    ]
    labels = ["INTEGRITY", "INCLUSION", "CONTINUITY", "COVERAGE", "WITNESSES"]
    figure = go.Figure(
        go.Scatterpolar(
            r=values + values[:1],
            theta=labels + labels[:1],
            fill="toself",
            line=dict(color=CYAN, width=2),
            fillcolor="rgba(80,216,215,.12)",
            marker=dict(color=MINT, size=5),
            hovertemplate="%{theta}<br>%{r:.0%}<extra></extra>",
        )
    )
    figure.update_layout(
        polar=dict(
            bgcolor=BG,
            radialaxis=dict(visible=True, range=[0, 1], showticklabels=False, gridcolor=GRID, linecolor=GRID),
            angularaxis=dict(gridcolor=GRID, linecolor=GRID, tickfont=dict(size=9, color=TEXT)),
        )
    )
    return _base(figure, height=330)
