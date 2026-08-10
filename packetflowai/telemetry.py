"""Logging setup kept outside runtime services."""

import json
import logging
from threading import Lock

from .config import AppConfig


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class MetricsRegistry:
    def __init__(self):
        self._values: dict[str, float] = {}
        self._lock = Lock()

    def set(self, name: str, value: float) -> None:
        with self._lock:
            self._values[name] = float(value)

    def increment(self, name: str, value: float = 1.0) -> None:
        with self._lock:
            self._values[name] = self._values.get(name, 0.0) + value

    def snapshot(self) -> dict[str, float]:
        with self._lock:
            return dict(self._values)

    def prometheus(self) -> str:
        lines = []
        for name, value in sorted(self.snapshot().items()):
            normalized = "".join(
                character if character.isalnum() or character == "_" else "_"
                for character in name
            )
            safe_name = "packetflowai_" + normalized
            lines.append(f"# TYPE {safe_name} gauge")
            lines.append(f"{safe_name} {value}")
        return "\n".join(lines) + "\n"


def configure_logging(config: AppConfig, verbose: bool = False, structured: bool = True) -> None:
    config.artifacts.create()
    level = logging.DEBUG if verbose else logging.INFO
    formatter = JSONFormatter() if structured else logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S"
    )
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(formatter)
    error_file = logging.FileHandler(config.artifacts.error_log)
    error_file.setLevel(logging.ERROR)
    error_file.setFormatter(formatter)
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)
    root.addHandler(console)
    root.addHandler(error_file)
