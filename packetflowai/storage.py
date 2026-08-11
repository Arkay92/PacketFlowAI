"""SQLite evidence and feedback store."""

import json
import sqlite3
from dataclasses import asdict, is_dataclass
from pathlib import Path
from threading import Lock
from typing import Any, cast

from .domain import FeedbackRecord

SCHEMA = """
CREATE TABLE IF NOT EXISTS flows (
    flow_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS alerts (
    alert_id TEXT PRIMARY KEY, event_id TEXT NOT NULL, created_at TEXT NOT NULL, payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS decisions (
    decision_id TEXT PRIMARY KEY, event_id TEXT NOT NULL, created_at TEXT NOT NULL, payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS evidence (
    evidence_id INTEGER PRIMARY KEY AUTOINCREMENT, event_id TEXT NOT NULL, channel TEXT NOT NULL,
    created_at TEXT NOT NULL, payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS nim_assessments (
    event_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS feedback (
    event_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, adjudicated INTEGER NOT NULL,
    analyst_label TEXT, payload TEXT NOT NULL
);
"""


def _json(value: Any) -> str:
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(cast(Any, value))
    return json.dumps(value, sort_keys=True, default=str)


class EventStore:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._connection = sqlite3.connect(path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.executescript(SCHEMA)
        self._connection.commit()
        self._lock = Lock()

    def close(self) -> None:
        self._connection.close()

    def _upsert(self, table: str, key_column: str, key: str, payload: Any,
                created_at: str, extra_columns: dict[str, Any] | None = None) -> None:
        columns = {key_column: key, "created_at": created_at, "payload": _json(payload), **(extra_columns or {})}
        names = ", ".join(columns)
        placeholders = ", ".join("?" for _ in columns)
        updates = ", ".join(f"{name}=excluded.{name}" for name in columns if name != key_column)
        with self._lock:
            self._connection.execute(
                f"INSERT INTO {table} ({names}) VALUES ({placeholders}) "
                f"ON CONFLICT({key_column}) DO UPDATE SET {updates}",
                tuple(columns.values()),
            )
            self._connection.commit()

    def add_flow(self, flow: Any, created_at: str) -> None:
        self._upsert("flows", "flow_id", flow.flow_id, flow, created_at)

    def add_alert(self, alert_id: str, event_id: str, payload: Any, created_at: str) -> None:
        self._upsert("alerts", "alert_id", alert_id, payload, created_at, {"event_id": event_id})

    def add_decision(self, decision_id: str, event_id: str, payload: Any, created_at: str) -> None:
        self._upsert("decisions", "decision_id", decision_id, payload, created_at, {"event_id": event_id})

    def add_evidence(self, event_id: str, channel: str, payload: Any, created_at: str) -> None:
        with self._lock:
            self._connection.execute(
                "INSERT INTO evidence(event_id, channel, created_at, payload) VALUES (?, ?, ?, ?)",
                (event_id, channel, created_at, _json(payload)),
            )
            self._connection.commit()

    def add_nim_assessment(self, event_id: str, assessment: Any, created_at: str) -> None:
        self._upsert("nim_assessments", "event_id", event_id, assessment, created_at)

    def add_feedback(self, record: FeedbackRecord) -> None:
        if record.analyst_label and not record.adjudicated:
            raise ValueError("analyst labels must be adjudicated before storage")
        self._upsert(
            "feedback", "event_id", record.event_id, record, record.created_at,
            {"adjudicated": int(record.adjudicated), "analyst_label": record.analyst_label},
        )

    def supervised_feedback(self) -> list[dict[str, Any]]:
        rows = self._connection.execute(
            "SELECT payload FROM feedback WHERE adjudicated=1 AND analyst_label IS NOT NULL ORDER BY created_at"
        ).fetchall()
        return [json.loads(row["payload"]) for row in rows]

    def list(self, table: str, limit: int = 100) -> list[dict[str, Any]]:
        if table not in {"flows", "alerts", "decisions", "evidence", "nim_assessments", "feedback"}:
            raise ValueError("unsupported table")
        rows = self._connection.execute(
            f"SELECT * FROM {table} ORDER BY created_at DESC LIMIT ?", (min(max(limit, 1), 1000),)
        ).fetchall()
        return [{**dict(row), "payload": json.loads(row["payload"])} for row in rows]

    def overview(self) -> dict[str, Any]:
        tables = ("flows", "alerts", "decisions", "evidence", "nim_assessments", "feedback")
        counts = {
            table: int(self._connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in tables
        }
        decision_rows = self._connection.execute(
            "SELECT payload FROM decisions ORDER BY created_at DESC LIMIT 1000"
        ).fetchall()
        decisions = [json.loads(row["payload"]) for row in decision_rows]
        classifications: dict[str, int] = {}
        policy_levels: dict[str, int] = {}
        risks = []
        for decision in decisions:
            evidence = decision.get("evidence", {})
            label = str(evidence.get("classifier_label", "unknown"))
            classifications[label] = classifications.get(label, 0) + 1
            level = str(decision.get("policy_level", "UNKNOWN"))
            policy_levels[level] = policy_levels.get(level, 0) + 1
            risks.append(float(decision.get("risk_score", 0.0)))
        latest_flow = self.list("flows", 1)
        latest_decision = self.list("decisions", 1)
        return {
            "counts": counts,
            "classifications": classifications,
            "policy_levels": policy_levels,
            "risk": {
                "current": risks[0] if risks else 0.0,
                "average": sum(risks) / len(risks) if risks else 0.0,
                "peak": max(risks, default=0.0),
            },
            "latest_flow": latest_flow[0] if latest_flow else None,
            "latest_decision": latest_decision[0] if latest_decision else None,
        }
