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
        assurance: dict[str, Any] | None = None,
        evidence_contract: dict[str, Any] | None = None,
        trust_material: dict[str, Any] | None = None,
        signer: tuple[str, str] | None = None,
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
        events = records.get("events", [])
        files["case.json"] = canonical_json({"case_id": case_id, "record_count": len(events)})
        files["evidence/events.jsonl"] = files["events.jsonl"]
        files["evidence/observations.jsonl"] = b"\n".join(
            canonical_json(item) for item in records.get("observations", events)
        ) + (b"\n" if records.get("observations", events) else b"")
        files["evidence/sources.json"] = canonical_json(records.get("sources", []))
        files["evidence/receipts.jsonl"] = b"\n".join(canonical_json(item) for item in records.get("receipts", [])) + (
            b"\n" if records.get("receipts") else b""
        )
        files["commitments/epochs.json"] = canonical_json(records.get("epochs", []))
        files["commitments/sequence-roots.json"] = canonical_json(records.get("sequence_roots", []))
        files["authority/records.jsonl"] = files["authority/records.jsonl"]
        files["attestations/collectors.json"] = canonical_json(records.get("attestations", []))
        files["attestations/reasoning-receipts.json"] = canonical_json(records.get("reasoning_receipts", []))
        files["anchors/checkpoints.json"] = canonical_json(records.get("checkpoints", []))
        files["omissions/ledger.jsonl"] = b"\n".join(canonical_json(item) for item in records.get("omissions", [])) + (
            b"\n" if records.get("omissions") else b""
        )
        files["redactions/leaves.json"] = canonical_json(records.get("redactions", []))
        files["decisions/capsules.json"] = canonical_json(records.get("capsules", []))
        files["evidence/transformations.jsonl"] = b"\n".join(
            canonical_json(item) for item in records.get("transformations", [])
        ) + (b"\n" if records.get("transformations") else b"")
        files["omissions/destruction-receipts.json"] = canonical_json(records.get("destruction_receipts", []))
        files["signatures/trust-material.json"] = canonical_json(trust_material or {})
        files["contracts/evidence-contract.json"] = canonical_json(evidence_contract or {})
        files["verification.json"] = canonical_json(
            assurance
            or {
                "integrity": "UNKNOWN",
                "inclusion": "UNKNOWN",
                "sequence_continuity": "UNKNOWN",
                "unknown_omission_risk": "NOT_ELIMINATED",
            }
        )
        files["models/reproducibility.json"] = canonical_json(reproducibility)
        files["hashes/chain.json"] = canonical_json(
            [{key: value for key, value in event.__dict__.items()} for event in ledger.events]
        )
        hashes = {name: _digest(data) for name, data in files.items()}
        root = _merkle([hashes[name] for name in sorted(hashes)])
        files["commitments/merkle-roots.json"] = canonical_json(
            {
                "case_root": root,
                "algorithm": "SHA-256",
                "leaf_order": sorted(hashes),
            }
        )
        hashes["commitments/merkle-roots.json"] = _digest(files["commitments/merkle-roots.json"])
        root = _merkle([hashes[name] for name in sorted(hashes)])
        witnesses = {
            name: hmac.new(secret.encode(), root.encode(), hashlib.sha256).hexdigest()
            for name, secret in (witness_keys or {}).items()
        }
        receipt = anchor.anchor(root) if anchor else None
        manifest: dict[str, Any] = {
            "bundle_version": "PFCASE-1.0",
            "schema": "https://packetflow.ai/spec/pfcase/1.0",
            "case_id": case_id,
            "created_at": datetime.now(UTC).isoformat(),
            "files": hashes,
            "merkle_root": root,
            "witnesses": witnesses,
            "external_anchor": receipt,
            "assurance_claims": (assurance or {}).get("formal_claims", []),
            "unknown_omission_risk": (assurance or {}).get("unknown_omission_risk", "NOT_ELIMINATED"),
            "algorithms": {"digest": "SHA-256", "witness": "HMAC-SHA256", "canonical": "PF-CANONICAL-JSON-1"},
        }
        if signer:
            key_id, secret = signer
            unsigned = dict(manifest)
            manifest["manifest_signature"] = {
                "key_id": key_id,
                "algorithm": "HMAC-SHA256",
                "value": hmac.new(secret.encode(), canonical_json(unsigned), hashlib.sha256).hexdigest(),
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
