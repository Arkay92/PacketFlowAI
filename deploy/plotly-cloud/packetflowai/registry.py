"""Filesystem model registry with atomic promotion and rollback."""

import json
import os
import shutil
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RegistryEntry:
    model_id: str
    version: str
    artifact: str
    state: str
    evaluated: bool
    shadow_validated: bool
    evaluation_report: str | None
    created_at: str


class FilesystemModelRegistry:
    def __init__(self, root: Path):
        self.root = root
        self.models = root / "models"
        self.state_path = root / "state.json"
        self.models.mkdir(parents=True, exist_ok=True)
        if not self.state_path.exists():
            self._write_state({"entries": {}, "active": None, "previous": None})

    def _read_state(self) -> dict[str, Any]:
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def _write_state(self, state: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, self.state_path)

    def register_candidate(self, model_id: str, version: str, artifact: Path,
                           evaluation_report: Path | None = None,
                           shadow_validated: bool = False) -> RegistryEntry:
        if not artifact.is_file():
            raise FileNotFoundError(artifact)
        key = f"{model_id}:{version}"
        target = self.models / f"{model_id}-{version}{artifact.suffix}"
        shutil.copy2(artifact, target)
        entry = RegistryEntry(
            model_id, version, str(target), "candidate", bool(evaluation_report), shadow_validated,
            str(evaluation_report) if evaluation_report else None, datetime.now(UTC).isoformat(),
        )
        state = self._read_state()
        state["entries"][key] = asdict(entry)
        self._write_state(state)
        return entry

    def list_models(self) -> list[dict[str, Any]]:
        return list(self._read_state()["entries"].values())

    def active_model(self) -> dict[str, Any]:
        state = self._read_state()
        if not state["active"]:
            raise RuntimeError("registry has no active model")
        return state["entries"][state["active"]]

    def promote(self, key: str, require_shadow: bool = True) -> dict[str, Any]:
        state = self._read_state()
        if key not in state["entries"]:
            raise KeyError(key)
        candidate = state["entries"][key]
        if not candidate["evaluated"]:
            raise ValueError("candidate requires offline evaluation before promotion")
        if require_shadow and not candidate["shadow_validated"]:
            raise ValueError("candidate requires shadow validation before promotion")
        previous = state["active"]
        if previous:
            state["entries"][previous]["state"] = "previous"
        candidate["state"] = "active"
        state["previous"] = previous
        state["active"] = key
        self._write_state(state)
        return candidate

    def mark_evaluated(self, key: str, report: Path, shadow_validated: bool = False) -> dict[str, Any]:
        if not report.is_file():
            raise FileNotFoundError(report)
        state = self._read_state()
        if key not in state["entries"]:
            raise KeyError(key)
        entry = state["entries"][key]
        entry["evaluated"] = True
        entry["evaluation_report"] = str(report)
        entry["shadow_validated"] = bool(shadow_validated)
        self._write_state(state)
        return entry

    def rollback(self) -> dict[str, Any]:
        state = self._read_state()
        previous = state["previous"]
        if not previous:
            raise RuntimeError("registry has no previous model")
        current = state["active"]
        state["entries"][current]["state"] = "candidate"
        state["entries"][previous]["state"] = "active"
        state["active"], state["previous"] = previous, current
        self._write_state(state)
        return state["entries"][previous]
