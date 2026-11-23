"""Privacy-preserving capture of real HTTP traffic received by the Dash app."""

from __future__ import annotations

import hashlib
import secrets
import time
from datetime import UTC, datetime
from threading import Lock
from typing import Any

from flask import Flask, Request, Response, request

from .domain import FlowFeatures
from .storage import EventStore
from .telemetry import MetricsRegistry


class HTTPFlowCapture:
    """Record real application requests without retaining visitor IP addresses."""

    def __init__(self, store: EventStore, metrics: MetricsRegistry, maximum_records: int = 1_000) -> None:
        self.store = store
        self.metrics = metrics
        self.maximum_records = maximum_records
        self.started_at = time.monotonic()
        self.request_count = 0
        self._salt = secrets.token_bytes(32)
        self._lock = Lock()

    def register(self, server: Flask) -> None:
        @server.after_request
        def capture_http_flow(response: Response) -> Response:
            try:
                self.observe(request, response)
            except Exception:
                # Telemetry must never make an otherwise healthy request fail.
                server.logger.exception("HTTP flow capture failed")
            return response

    def _pseudonymous_source(self, incoming: Request) -> str:
        forwarded = incoming.headers.get("X-Forwarded-For", "").split(",", 1)[0].strip()
        address = forwarded or incoming.remote_addr or "unknown"
        digest = hashlib.sha256(self._salt + address.encode("utf-8", "replace")).digest()
        # RFC 5737 documentation space prevents the UI from exposing visitor IPs.
        return f"192.0.2.{1 + digest[0] % 254}"

    @staticmethod
    def _classification(status: int) -> tuple[str, float, int]:
        if status >= 500:
            return "service_error", 82.0, 4
        if status >= 400:
            return "unexpected_request", 42.0, 2
        return "benign", 8.0, 1

    def observe(self, incoming: Request, response: Response) -> None:
        now = datetime.now(UTC)
        with self._lock:
            self.request_count += 1
            sequence = self.request_count
        source = self._pseudonymous_source(incoming)
        status = int(response.status_code)
        label, risk, policy_level = self._classification(status)
        destination_port = 443 if incoming.scheme == "https" else 80
        request_bytes = int(incoming.content_length or 0)
        response_bytes = int(response.content_length or 0)
        byte_count = request_bytes + response_bytes
        event_id = hashlib.sha256(
            f"{now.isoformat()}|{sequence}|{incoming.method}|{incoming.path}|{source}".encode()
        ).hexdigest()[:24]
        created_at = now.isoformat()
        flow = FlowFeatures(
            flow_id=event_id,
            source_ip=source,
            destination_ip="198.51.100.10",
            source_port=40_000 + sequence % 20_000,
            destination_port=destination_port,
            protocol="HTTP",
            packet_count=2,
            byte_count=byte_count,
            duration_seconds=0.0,
            packets_per_second=2.0,
            bytes_per_second=float(byte_count),
            forward_packets=1,
            reverse_packets=1,
            forward_bytes=request_bytes,
            reverse_bytes=response_bytes,
            state="CLOSED",
            protocol_metadata={
                "capture": "real_http",
                "method": incoming.method,
                "path": incoming.path[:160],
                "status": status,
                "user_agent_family": self._user_agent_family(incoming.headers.get("User-Agent", "")),
            },
        )
        evidence: dict[str, Any] = {
            "classifier_label": label,
            "classifier_confidence": 1.0,
            "calibrated_confidence": 1.0,
            "prototype_similarity": None,
            "anomaly_score": risk / 100,
            "nim_reasoning_strength": None,
            "capture_channel": "http_request",
        }
        decision = {
            "event_id": event_id,
            "action": "alert" if status >= 500 else "observe",
            "policy_level": policy_level,
            "risk_score": risk,
            "explanation": f"Observed real {incoming.method} {incoming.path[:120]} response {status}.",
            "evidence": evidence,
        }
        self.store.add_flow(flow, created_at)
        self.store.add_decision(f"http-{event_id}", event_id, decision, created_at)
        self.store.add_evidence(event_id, "http_capture", evidence, created_at)
        if status >= 500:
            self.store.add_alert(
                f"http-alert-{event_id}",
                event_id,
                {
                    "event_id": event_id,
                    "policy_level": policy_level,
                    "action": "alert",
                    "target": incoming.path[:120],
                    "reason": f"Application request returned HTTP {status}.",
                },
                created_at,
            )
        elapsed = max(time.monotonic() - self.started_at, 1.0)
        self.metrics.set("flows_per_second", self.request_count / elapsed)
        self.metrics.set("packets_per_second", self.request_count * 2 / elapsed)
        self.metrics.set("queue_depth", 0)
        self.metrics.set("dropped_packets", 0)
        if sequence % 25 == 0:
            for table in ("flows", "decisions", "evidence", "alerts"):
                self.store.prune(table, self.maximum_records)

    @staticmethod
    def _user_agent_family(user_agent: str) -> str:
        normalized = user_agent.lower()
        for family in ("chrome", "firefox", "safari", "edge", "curl", "python"):
            if family in normalized:
                return family
        return "other"
