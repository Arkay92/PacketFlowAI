"""Portable evidence bundles, witnesses, anchoring, and forensic diffing."""

from __future__ import annotations

import hashlib
import hmac
import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .governance import EvidenceLedger, canonical_json


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _merkle(leaves: list[str]) -> str:
    layer = list(leaves)
    if not layer:
        return _digest(b"")
    while len(layer) > 1:
        if len(layer) % 2:
            layer.append(layer[-1])
        layer = [_digest((layer[index] + layer[index + 1]).encode()) for index in range(0, len(layer), 2)]
    return layer[0]


class FileTransparencyLog:
    """Local append-only anchor backend; remote systems can implement the same interface."""

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def anchor(self, root: str) -> dict[str, Any]:
        entries = self.entries()
        previous = entries[-1]["entry_hash"] if entries else "0" * 64
        body = {
            "sequence": len(entries),
            "root": root,
            "previous": previous,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        entry = {**body, "entry_hash": _digest(canonical_json(body))}
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry, sort_keys=True) + "\n")
        return entry

    def entries(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line]


class EvidenceBundleExporter:
    def export(
        self,
        output: Path,
        case_id: str,
        records: dict[str, list[dict[str, Any]]],
        ledger: EvidenceLedger,
        reproducibility: dict[str, Any],
        witness_keys: dict[str, str] | None = None,
        anchor: FileTransparencyLog | None = None,
    ) -> dict[str, Any]:
        files: dict[str, bytes] = {}
        mapping = {
            "events": "events.jsonl",
            "evidence": "evidence/records.jsonl",
            "decisions": "decisions/records.jsonl",
            "models": "models/artifacts.json",
            "policies": "policies/versions.json",
            "authority": "authority/records.jsonl",
        }
        for key, name in mapping.items():
            values = records.get(key, [])
            if name.endswith(".jsonl"):
                files[name] = b"\n".join(canonical_json(item) for item in values) + (b"\n" if values else b"")
            else:
                files[name] = canonical_json(values)
        files["models/reproducibility.json"] = canonical_json(reproducibility)
        files["hashes/chain.json"] = canonical_json(
            [{key: value for key, value in event.__dict__.items()} for event in ledger.events]
        )
        hashes = {name: _digest(data) for name, data in files.items()}
        root = _merkle([hashes[name] for name in sorted(hashes)])
        witnesses = {
            name: hmac.new(secret.encode(), root.encode(), hashlib.sha256).hexdigest()
            for name, secret in (witness_keys or {}).items()
        }
        receipt = anchor.anchor(root) if anchor else None
        manifest = {
            "bundle_version": "1.0",
            "case_id": case_id,
            "created_at": datetime.now(UTC).isoformat(),
            "files": hashes,
            "merkle_root": root,
            "witnesses": witnesses,
            "external_anchor": receipt,
            "algorithms": {"digest": "SHA-256", "witness": "HMAC-SHA256", "canonical": "RFC8785-like JSON"},
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
            for name, data in files.items():
                archive.writestr(name, data)
            archive.writestr("manifest.json", canonical_json(manifest))
        return manifest


class EvidenceDiffer:
    def compare(self, expected: list[dict[str, Any]], current: list[dict[str, Any]], key: str) -> dict[str, Any]:
        left = {str(item[key]): item for item in expected}
        right = {str(item[key]): item for item in current}
        return {
            "missing": sorted(set(left) - set(right)),
            "unexpected": sorted(set(right) - set(left)),
            "modified": sorted(
                item for item in left.keys() & right.keys() if canonical_json(left[item]) != canonical_json(right[item])
            ),
        }
