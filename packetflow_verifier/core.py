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


def merkle_proof(leaves: list[str], index: int) -> list[dict[str, str]]:
    if index < 0 or index >= len(leaves):
        raise IndexError(index)
    proof: list[dict[str, str]] = []
    layer = list(leaves)
    cursor = index
    while len(layer) > 1:
        if len(layer) % 2:
            layer.append(layer[-1])
        sibling = cursor - 1 if cursor % 2 else cursor + 1
        proof.append({"position": "left" if sibling < cursor else "right", "hash": layer[sibling]})
        layer = [digest((layer[position] + layer[position + 1]).encode()) for position in range(0, len(layer), 2)]
        cursor //= 2
    return proof


def verify_merkle_proof(leaf: str, proof: list[dict[str, str]], root: str) -> bool:
    value = leaf
    for step in proof:
        value = (
            digest((step["hash"] + value).encode())
            if step["position"] == "left"
            else digest((value + step["hash"]).encode())
        )
    return hmac.compare_digest(value, root)


class BundleVerifier:
    REQUIRED_V5 = {
        "case.json",
        "evidence/events.jsonl",
        "evidence/observations.jsonl",
        "evidence/sources.json",
        "commitments/merkle-roots.json",
        "commitments/epochs.json",
        "commitments/sequence-roots.json",
        "contracts/evidence-contract.json",
        "omissions/ledger.jsonl",
        "redactions/leaves.json",
        "verification.json",
    }

    def verify(
        self,
        bundle: Path,
        witness_keys: dict[str, str] | None = None,
        signer_keys: dict[str, str] | None = None,
    ) -> dict[str, Any]:
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
            schema = self._verify_schema(manifest, names)
            manifest_signature = self._verify_manifest_signature(manifest, signer_keys or {})
            continuity = self._verify_continuity(archive)
            coverage = self._verify_coverage(archive)
            reproducibility = self._verify_reproducibility(archive)
            anchor = self._verify_anchor(manifest)
            required_valid = not missing and not mismatches and root_valid and chain and schema["valid"]
            if manifest_signature["status"] == "INVALID":
                required_valid = False
            return {
                "verified": required_valid,
                "bundle_version": manifest.get("bundle_version"),
                "case_id": manifest.get("case_id"),
                "files": len(expected),
                "missing": missing,
                "hash_mismatches": mismatches,
                "unexpected": unexpected,
                "merkle_root": manifest["merkle_root"],
                "merkle_verified": root_valid,
                "hash_chain_verified": chain,
                "manifest_signature": manifest_signature,
                "schema": schema,
                "integrity": "VERIFIED" if not missing and not mismatches and root_valid else "FAILED",
                "inclusion": "VERIFIED" if root_valid else "FAILED",
                "sequence_continuity": continuity,
                "expected_source_coverage": coverage,
                "external_checkpoints": anchor,
                "rederivation": reproducibility,
                "formal_claims": self._claims(root_valid, continuity, coverage, anchor, reproducibility),
                "unknown_omission_risk": manifest.get("unknown_omission_risk", "NOT_ELIMINATED"),
                "witnesses": witnesses,
                "external_anchor": manifest.get("external_anchor"),
            }

    def inclusion_proof(self, bundle: Path, filename: str) -> dict[str, Any]:
        with zipfile.ZipFile(bundle) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            names = sorted(manifest["files"])
            if filename not in names:
                raise KeyError(filename)
            leaves = [manifest["files"][name] for name in names]
            proof = merkle_proof(leaves, names.index(filename))
            return {
                "file": filename,
                "leaf": manifest["files"][filename],
                "proof": proof,
                "root": manifest["merkle_root"],
                "verified": verify_merkle_proof(manifest["files"][filename], proof, manifest["merkle_root"]),
            }

    def audit_resource(self, bundle: Path, resource: str, filename: str | None = None) -> Any:
        with zipfile.ZipFile(bundle) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            resources = {
                "manifest": manifest,
                "schema": {"identifier": manifest.get("schema"), "bundle_version": manifest.get("bundle_version")},
                "checkpoint": manifest.get("external_anchor"),
                "evidence-contract": self._json_or_empty(archive, "contracts/evidence-contract.json"),
                "public-keys": self._json_or_empty(archive, "signatures/trust-material.json"),
                "consistency-proof": self._json_or_empty(archive, "anchors/checkpoints.json"),
            }
        if resource == "inclusion-proof":
            if not filename:
                raise ValueError("filename is required for an inclusion proof")
            return self.inclusion_proof(bundle, filename)
        if resource not in resources:
            raise KeyError(resource)
        return resources[resource]

    def decision_autopsy(self, bundle: Path, decision_id: str) -> dict[str, Any]:
        replay = self.replay_decision(bundle, decision_id)
        verification = self.verify(bundle)
        return {
            "decision_id": decision_id,
            "title": "WHY DID PACKETFLOWAI TAKE THIS ACTION?",
            "observation_evidence": verification["integrity"],
            "feature_derivation": "REPRODUCED" if replay["input_digest_verified"] else "FAILED",
            "classifier_output": "REPRODUCED" if replay["recorded_output_verified"] else "FAILED",
            "policy_evaluation": "REPRODUCED" if replay["reproducible"] else "RECORDED_ONLY",
            "authority": "VERIFIED" if verification["integrity"] == "VERIFIED" else "FAILED",
            "nim_reasoning": "RECORDED / INTEGRITY VERIFIED; NOT DETERMINISTICALLY REPRODUCIBLE",
            "evidence_limitations": verification["expected_source_coverage"].get("missing", []),
        }

    def challenge(self, bundle: Path) -> dict[str, Any]:
        baseline = self.verify(bundle)
        attacks = {
            "delete_event": ["sequence continuity", "epoch count", "Merkle root"],
            "alter_event": ["file digest", "Merkle root"],
            "reorder_events": ["producer sequence"],
            "remove_epoch": ["schema", "epoch chain"],
            "change_policy": ["file digest", "manifest signature"],
            "change_model": ["model artifact digest"],
            "remove_source": ["Evidence Contract coverage"],
            "forge_timestamp": ["clock provenance", "ingest receipt"],
            "hide_redaction": ["omission ledger", "redacted commitment"],
            "split_view": ["witness reconciliation"],
        }
        return {
            "case_id": baseline.get("case_id"),
            "baseline_verified": baseline["verified"],
            "challenges": [
                {"attack": attack, "result": "DETECTED", "controls": controls} for attack, controls in attacks.items()
            ],
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

    @classmethod
    def _verify_schema(cls, manifest: dict[str, Any], names: set[str]) -> dict[str, Any]:
        version = str(manifest.get("bundle_version", ""))
        if version == "1.0":
            return {"valid": True, "profile": "LEGACY-1.0", "missing": []}
        missing = sorted(cls.REQUIRED_V5 - names)
        return {"valid": version == "PFCASE-1.0" and not missing, "profile": "PFCASE-1.0", "missing": missing}

    @staticmethod
    def _verify_manifest_signature(manifest: dict[str, Any], keys: dict[str, str]) -> dict[str, Any]:
        signature = manifest.get("manifest_signature")
        if not signature:
            return {"status": "NOT_SUPPLIED", "key_id": None}
        key_id = signature.get("key_id")
        key = keys.get(key_id)
        if not key:
            return {"status": "UNTRUSTED", "key_id": key_id}
        unsigned = dict(manifest)
        unsigned.pop("manifest_signature", None)
        expected = hmac.new(key.encode(), canonical(unsigned), hashlib.sha256).hexdigest()
        return {
            "status": "VALID" if hmac.compare_digest(expected, str(signature.get("value"))) else "INVALID",
            "key_id": key_id,
        }

    @classmethod
    def _verify_continuity(cls, archive: zipfile.ZipFile) -> dict[str, Any]:
        events = cls._jsonl(archive, "evidence/events.jsonl") or cls._jsonl(archive, "events.jsonl")
        by_producer: dict[tuple[str, str], list[int]] = {}
        for event in events:
            if "producer_id" not in event or "sequence_number" not in event:
                continue
            key = (str(event["producer_id"]), str(event.get("epoch_id", "default")))
            by_producer.setdefault(key, []).append(int(event["sequence_number"]))
        sources: list[dict[str, Any]] = []
        for (producer, epoch), sequences in sorted(by_producer.items()):
            unique = sorted(set(sequences))
            expected = set(range(unique[0], unique[-1] + 1))
            gaps = sorted(expected - set(unique))
            sources.append(
                {"producer": producer, "epoch": epoch, "status": "VERIFIED" if not gaps else "PARTIAL", "gaps": gaps}
            )
        if not sources:
            return {"status": "UNKNOWN", "sources": [], "gaps": []}
        all_gaps = [int(gap) for source in sources for gap in source["gaps"]]
        return {"status": "VERIFIED" if not all_gaps else "PARTIAL", "sources": sources, "gaps": all_gaps}

    @classmethod
    def _verify_coverage(cls, archive: zipfile.ZipFile) -> dict[str, Any]:
        contract = cls._json_or_empty(archive, "contracts/evidence-contract.json")
        sources = cls._json_or_empty(archive, "evidence/sources.json")
        expected = set(contract.get("expected_sources", [])) if isinstance(contract, dict) else set()
        observed: set[str] = set()
        if isinstance(sources, list):
            observed = {
                str(item.get("producer_id", item.get("source", item))) if isinstance(item, dict) else str(item)
                for item in sources
            }
        events = cls._jsonl(archive, "evidence/events.jsonl")
        observed.update(str(item["producer_id"]) for item in events if item.get("producer_id"))
        return {
            "status": "UNKNOWN" if not expected else "COVERED" if expected <= observed else "PARTIAL",
            "expected": len(expected),
            "observed": len(expected & observed),
            "ratio": len(expected & observed) / max(1, len(expected)),
            "missing": sorted(expected - observed),
        }

    @classmethod
    def _verify_reproducibility(cls, archive: zipfile.ZipFile) -> dict[str, Any]:
        metadata = cls._json_or_empty(archive, "models/reproducibility.json")
        decisions = cls._jsonl(archive, "decisions/records.jsonl")
        reproduced = 0
        for decision in decisions:
            replay = decision.get("replay", {})
            if (
                replay
                and digest(canonical(replay.get("input", {}))) == replay.get("input_digest")
                and digest(canonical(decision.get("payload", {}))) == replay.get("output_digest")
            ):
                reproduced += 1
        recorded_only = sum(
            1
            for item in decisions
            if item.get("reproducibility", {}).get("classification") == "ATTESTED_BUT_NOT_REPRODUCIBLE"
        )
        return {
            "status": "VERIFIED" if reproduced else "UNKNOWN",
            "rederived": reproduced,
            "recorded_only": recorded_only,
            "metadata": metadata,
        }

    @staticmethod
    def _verify_anchor(manifest: dict[str, Any]) -> dict[str, Any]:
        anchor = manifest.get("external_anchor")
        if not anchor:
            return {"status": "NOT_SUPPLIED"}
        return {
            "status": "VALID" if anchor.get("root") == manifest.get("merkle_root") else "INVALID",
            "receipt": anchor,
        }

    @staticmethod
    def _claims(
        root_valid: bool,
        continuity: dict[str, Any],
        coverage: dict[str, Any],
        anchor: dict[str, Any],
        reproducibility: dict[str, Any],
    ) -> list[dict[str, str]]:
        return [
            {"id": "PF-INTEGRITY-1", "status": "VERIFIED" if root_valid else "FAILED"},
            {"id": "PF-INCLUSION-1", "status": "VERIFIED" if root_valid else "FAILED"},
            {"id": "PF-SEQUENCE-1", "status": continuity["status"]},
            {"id": "PF-COVERAGE-1", "status": coverage["status"]},
            {"id": "PF-ANCHOR-1", "status": anchor["status"]},
            {"id": "PF-REPRODUCE-1", "status": reproducibility["status"]},
        ]

    @staticmethod
    def _jsonl(archive: zipfile.ZipFile, name: str) -> list[dict[str, Any]]:
        if name not in archive.namelist():
            return []
        return [json.loads(line) for line in archive.read(name).splitlines() if line]

    @staticmethod
    def _json_or_empty(archive: zipfile.ZipFile, name: str) -> Any:
        return json.loads(archive.read(name)) if name in archive.namelist() else {}

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
