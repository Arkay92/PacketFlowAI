"""Response adapter interfaces and reversible action requests."""

import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from .fusion import PolicyLevel


@dataclass(frozen=True)
class ActionRequest:
    action_id: str
    event_id: str
    action: str
    target: str | None
    policy_level: PolicyLevel
    reason: str
    reversible: bool
    requires_confirmation: bool
    ttl_seconds: int | None
    confirmed: bool = False
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    @property
    def expires_at(self) -> str | None:
        if self.ttl_seconds is None:
            return None
        created = datetime.fromisoformat(self.created_at)
        return (created + timedelta(seconds=self.ttl_seconds)).isoformat()


@dataclass(frozen=True)
class ActionResult:
    action_id: str
    status: str
    adapter: str
    details: dict[str, Any] = field(default_factory=dict)


class ResponseAdapter(ABC):
    @abstractmethod
    def execute(self, request: ActionRequest) -> ActionResult: ...

    def _validate(self, request: ActionRequest) -> None:
        if request.requires_confirmation and not request.confirmed:
            raise PermissionError(f"action {request.action} requires confirmation")
        if request.action in {"rate_limit", "temporary_block", "quarantine"}:
            if not request.reversible or not request.ttl_seconds:
                raise ValueError("containment actions must be reversible and TTL-based")


class AlertAdapter(ResponseAdapter):
    def execute(self, request: ActionRequest) -> ActionResult:
        logging.warning("Alert %s event=%s reason=%s", request.action_id, request.event_id, request.reason)
        return ActionResult(request.action_id, "executed", "alert")


class JSONFileAdapter(ResponseAdapter):
    def __init__(self, path: Path):
        self.path = path

    def execute(self, request: ActionRequest) -> ActionResult:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as output:
            output.write(json.dumps({**asdict(request), "policy_level": request.policy_level.name}) + "\n")
        return ActionResult(request.action_id, "executed", "json-file", {"path": str(self.path)})


class WebhookAdapter(ResponseAdapter):
    def __init__(self, url: str, timeout_seconds: float = 5.0):
        self.url = url
        self.timeout_seconds = timeout_seconds

    def execute(self, request: ActionRequest) -> ActionResult:
        payload = json.dumps({**asdict(request), "policy_level": request.policy_level.name}).encode("utf-8")
        http_request = Request(self.url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
        with urlopen(http_request, timeout=self.timeout_seconds) as response:
            status = response.status
        return ActionResult(request.action_id, "executed", "webhook", {"http_status": status})


class SIEMAdapter(JSONFileAdapter):
    """CEF-compatible JSON export; transport can be replaced without policy changes."""

    def execute(self, request: ActionRequest) -> ActionResult:
        result = super().execute(request)
        return ActionResult(result.action_id, result.status, "siem-json", result.details)


class CEFAdapter(ResponseAdapter):
    def __init__(self, path: Path):
        self.path = path

    def execute(self, request: ActionRequest) -> ActionResult:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        severity = min(10, int(request.policy_level) * 2)
        extension = f"src={request.target or ''} cs1={request.event_id} cs1Label=eventId"
        line = f"CEF:0|PacketFlowAI|PacketFlowAI|2.3|{request.action}|{request.reason[:128]}|{severity}|{extension}\n"
        with self.path.open("a", encoding="utf-8") as output:
            output.write(line)
        return ActionResult(request.action_id, "executed", "cef", {"path": str(self.path)})


class EnforcementAdapter(ResponseAdapter):
    def __init__(self, name: str, apply: Callable[[ActionRequest], Any], rollback: Callable[[ActionRequest], Any]):
        self.name = name
        self.apply = apply
        self.rollback = rollback

    def execute(self, request: ActionRequest) -> ActionResult:
        self._validate(request)
        details = self.apply(request)
        return ActionResult(
            request.action_id,
            "executed",
            self.name,
            {"result": details, "expires_at": request.expires_at},
        )

    def reverse(self, request: ActionRequest) -> ActionResult:
        self._validate(request)
        details = self.rollback(request)
        return ActionResult(request.action_id, "reversed", self.name, {"result": details})


class MirrorAdapter(EnforcementAdapter):
    pass


class RateLimitAdapter(EnforcementAdapter):
    pass


class TemporaryBlockAdapter(EnforcementAdapter):
    pass


class QuarantineAdapter(EnforcementAdapter):
    pass


class ReversibleActionExecutor:
    """Tracks containment expiry and invokes adapter rollback deterministically."""

    def __init__(self):
        self._active: dict[str, tuple[ActionRequest, EnforcementAdapter]] = {}

    def execute(self, request: ActionRequest, adapter: EnforcementAdapter) -> ActionResult:
        result = adapter.execute(request)
        if request.reversible and request.expires_at:
            self._active[request.action_id] = (request, adapter)
        return result

    def expire_due(self, now: datetime | None = None) -> tuple[ActionResult, ...]:
        current = now or datetime.now(UTC)
        expired = []
        for action_id, (request, adapter) in list(self._active.items()):
            expires_at = datetime.fromisoformat(request.expires_at) if request.expires_at else None
            if expires_at is not None and expires_at <= current:
                expired.append(adapter.reverse(request))
                self._active.pop(action_id)
        return tuple(expired)


class ResponsePolicyEngine:
    def __init__(self, containment_enabled: bool = False, default_ttl_seconds: int = 300):
        self.containment_enabled = containment_enabled
        self.default_ttl_seconds = default_ttl_seconds

    def decide(self, event_id: str, source_ip: str, level: PolicyLevel, reason: str) -> tuple[ActionRequest, ...]:
        if level == PolicyLevel.NORMAL:
            return ()
        alert = ActionRequest(
            action_id=f"{event_id}:alert",
            event_id=event_id,
            action="alert",
            target=source_ip,
            policy_level=level,
            reason=reason,
            reversible=True,
            requires_confirmation=False,
            ttl_seconds=None,
            confirmed=True,
        )
        if level != PolicyLevel.CONTAIN or not self.containment_enabled:
            return (alert,)
        containment = ActionRequest(
            action_id=f"{event_id}:temporary-block",
            event_id=event_id,
            action="temporary_block",
            target=source_ip,
            policy_level=level,
            reason=reason,
            reversible=True,
            requires_confirmation=True,
            ttl_seconds=self.default_ttl_seconds,
        )
        return alert, containment
