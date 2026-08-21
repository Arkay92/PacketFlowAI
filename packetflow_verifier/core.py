"""Verification primitives intentionally independent from the PacketFlowAI package."""

from __future__ import annotations

import hashlib
import hmac
import json
import zipfile
from pathlib import Path
from typing import Any


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def merkle_root(leaves: list[str]) -> str:
    layer = list(leaves)
    if not layer:
        return digest(b"")
    while len(layer) > 1:
        if len(layer) % 2:
            layer.append(layer[-1])
        layer = [digest((layer[index] + layer[index + 1]).encode()) for index in range(0, len(layer), 2)]
    return layer[0]


class BundleVerifier:
    def verify(self, bundle: Path, witness_keys: dict[str, str] | None = None) -> dict[str, Any]:
        with zipfile.ZipFile(bundle) as archive:
            names = set(archive.namelist())
            manifest = json.loads(archive.read("manifest.json"))
            expected = manifest["files"]
            missing = sorted(set(expected) - names)
            mismatches = sorted(
                name
                for name, expected_hash in expected.items()
                if name in names and digest(archive.read(name)) != expected_hash
            )
            unexpected = sorted(names - set(expected) - {"manifest.json"})
            leaves = [expected[name] for name in sorted(expected)]
            root_valid = merkle_root(leaves) == manifest["merkle_root"]
            witnesses = self._verify_witnesses(manifest, witness_keys or {})
            chain = self._verify_chain(archive) if "hashes/chain.json" in names else False
            return {
                "verified": not missing and not mismatches and root_valid and chain,
                "bundle_version": manifest.get("bundle_version"),
                "case_id": manifest.get("case_id"),
                "files": len(expected),
                "missing": missing,
                "hash_mismatches": mismatches,
                "unexpected": unexpected,
                "merkle_root": manifest["merkle_root"],
                "merkle_verified": root_valid,
                "hash_chain_verified": chain,
                "witnesses": witnesses,
                "external_anchor": manifest.get("external_anchor"),
            }

    def replay_decision(self, bundle: Path, decision_id: str) -> dict[str, Any]:
        with zipfile.ZipFile(bundle) as archive:
            records = [json.loads(line) for line in archive.read("decisions/records.jsonl").splitlines()]
            decision = next((item for item in records if item.get("decision_id") == decision_id), None)
            if decision is None:
                raise KeyError(f"decision not found: {decision_id}")
            replay = decision.get("replay", {})
            input_valid = digest(canonical(replay.get("input", {}))) == replay.get("input_digest")
            output_valid = digest(canonical(decision.get("payload", {}))) == replay.get("output_digest")
            return {
                "decision_id": decision_id,
                "reproducible": input_valid and output_valid,
                "input_digest_verified": input_valid,
                "recorded_output_verified": output_valid,
                "model": decision.get("reproducibility", {}),
            }

    @staticmethod
    def _verify_chain(archive: zipfile.ZipFile) -> bool:
        records = json.loads(archive.read("hashes/chain.json"))
        previous = "0" * 64
        for record in records:
            body = dict(record)
            record_hash = body.pop("record_hash")
            if body.get("previous_hash") != previous or digest(canonical(body)) != record_hash:
                return False
            previous = record_hash
        return True

    @staticmethod
    def _verify_witnesses(manifest: dict[str, Any], keys: dict[str, str]) -> dict[str, Any]:
        signatures = manifest.get("witnesses", {})
        valid: list[str] = []
        invalid: list[str] = []
        for witness, signature in signatures.items():
            key = keys.get(witness)
            expected = (
                hmac.new(key.encode(), manifest["merkle_root"].encode(), hashlib.sha256).hexdigest() if key else None
            )
            (valid if expected and hmac.compare_digest(expected, signature) else invalid).append(witness)
        return {"valid": valid, "invalid_or_untrusted": invalid, "count": len(valid)}
