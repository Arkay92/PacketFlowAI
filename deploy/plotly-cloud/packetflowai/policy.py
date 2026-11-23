"""Conservative response actions and post-classification risk tracking."""

import ipaddress
import math
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class ActionCapabilities:
    reversible: bool
    requires_confirmation: bool
    default_ttl: int | None
    minimum_policy_level: int


class ResponseAction:
    name = "response"
    capabilities = ActionCapabilities(True, True, None, 1)

    def execute(self, event: dict) -> dict:
        return {"action": self.name, "event": event, "executed": True}


class AlertAction(ResponseAction):
    name = "alert"
    capabilities = ActionCapabilities(True, False, None, 0)


class WebhookAction(ResponseAction):
    name = "webhook"


class SIEMAction(ResponseAction):
    name = "siem"


class MirrorAction(ResponseAction):
    name = "mirror"


class RateLimitAction(ResponseAction):
    name = "rate_limit"
    capabilities = ActionCapabilities(True, True, 300, 2)


class TemporaryBlockAction(ResponseAction):
    name = "temporary_block"
    capabilities = ActionCapabilities(True, True, 300, 3)


class QuarantineAction(ResponseAction):
    name = "quarantine"
    capabilities = ActionCapabilities(True, True, 300, 4)


class AlertOnlyPolicy:
    def __init__(self, action: AlertAction | None = None):
        self.action = action or AlertAction()

    def respond(self, event: dict) -> list[dict]:
        return [self.action.execute(event)] if event.get("label") != "benign" else []


class RiskTracker:
    def __init__(self, half_life_seconds: float = 300.0, allowlist: tuple[str, ...] = ()):
        if half_life_seconds <= 0:
            raise ValueError("half_life_seconds must be positive")
        self.half_life_seconds = half_life_seconds
        self.allowlist = tuple(ipaddress.ip_network(value, strict=False) for value in allowlist)
        self._scores: dict[str, tuple[float, float]] = {}

    def is_allowlisted(self, source_ip: str) -> bool:
        address = ipaddress.ip_address(source_ip)
        return any(address in network for network in self.allowlist)

    def score(self, source_ip: str, now: float | None = None) -> float:
        score, updated_at = self._scores.get(source_ip, (0.0, now if now is not None else time.time()))
        current_time = time.time() if now is None else now
        return score * math.pow(0.5, max(0.0, current_time - updated_at) / self.half_life_seconds)

    def update_after_classification(self, source_ip: str, label: str, severity: float = 1.0,
                                    now: float | None = None) -> float:
        current_time = time.time() if now is None else now
        current = self.score(source_ip, current_time)
        if label == "benign" or self.is_allowlisted(source_ip):
            self._scores[source_ip] = (current, current_time)
            return current
        updated = current + max(0.0, severity)
        self._scores[source_ip] = (updated, current_time)
        return updated
