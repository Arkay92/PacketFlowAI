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
    args = parser.parse_args()
    verifier = BundleVerifier()
    result = (
        verifier.verify(args.bundle)
        if args.command == "verify"
        else verifier.replay_decision(args.bundle, args.decision)
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("verified", result.get("reproducible", False)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
