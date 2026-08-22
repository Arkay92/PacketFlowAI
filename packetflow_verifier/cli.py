"""Command-line interface for independent PacketFlow evidence verification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import BundleVerifier


def main() -> int:
    parser = argparse.ArgumentParser(prog="packetflow-verifier")
    commands = parser.add_subparsers(dest="command", required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("bundle", type=Path)
    replay = commands.add_parser("replay-decision")
    replay.add_argument("bundle", type=Path)
    replay.add_argument("--decision", required=True)
    autopsy = commands.add_parser("decision-autopsy")
    autopsy.add_argument("bundle", type=Path)
    autopsy.add_argument("--decision", required=True)
    challenge = commands.add_parser("challenge")
    challenge.add_argument("bundle", type=Path)
    proof = commands.add_parser("inclusion-proof")
    proof.add_argument("bundle", type=Path)
    proof.add_argument("--file", required=True)
    audit = commands.add_parser("audit-resource")
    audit.add_argument("bundle", type=Path)
    audit.add_argument(
        "resource",
        choices=("manifest", "public-keys", "schema", "consistency-proof", "checkpoint", "evidence-contract"),
    )
    args = parser.parse_args()
    verifier = BundleVerifier()
    if args.command == "verify":
        result = verifier.verify(args.bundle)
    elif args.command == "replay-decision":
        result = verifier.replay_decision(args.bundle, args.decision)
    elif args.command == "decision-autopsy":
        result = verifier.decision_autopsy(args.bundle, args.decision)
    elif args.command == "challenge":
        result = verifier.challenge(args.bundle)
    elif args.command == "inclusion-proof":
        result = verifier.inclusion_proof(args.bundle, args.file)
    else:
        result = verifier.audit_resource(args.bundle, args.resource)
    print(json.dumps(result, indent=2, sort_keys=True))
    failed = result.get("verified") is False or result.get("reproducible") is False
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
