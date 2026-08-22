"""Record the PacketFlowAI v5 Verifiable Assurance walkthrough."""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from threading import Thread

from playwright.sync_api import Page, sync_playwright

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from create_forensics_video import (  # noqa: E402
    guided_click,
    hold,
    install_guided_cursor,
    move_to,
    smooth_scroll,
    tour_card,
)
from create_ui_video import MINIMUM_HOLD_SECONDS, install_card_style, seed_presentation  # noqa: E402

from packetflowai.api import APIServer  # noqa: E402
from packetflowai.config import AppConfig, ArtifactPaths  # noqa: E402
from packetflowai.registry import FilesystemModelRegistry  # noqa: E402
from packetflowai.storage import EventStore  # noqa: E402
from packetflowai.telemetry import MetricsRegistry  # noqa: E402

DEFAULT_OUTPUT = Path("artifacts/presentation/packetflowai-v5-verifiable-assurance-tour.webm")
DEFAULT_HOLD_SECONDS = 5.0


def card(
    page: Page,
    index: str,
    eyebrow: str,
    title: str,
    description: str,
    points: list[tuple[str, str]],
    hold_seconds: float,
    dismiss: bool = True,
) -> None:
    tour_card(page, index, eyebrow, title, description, points, hold_seconds, dismiss=dismiss)


def record_v5_tour(page: Page, base_url: str, hold_seconds: float) -> None:
    page.add_init_script(
        """() => {
          const nativeSetInterval = window.setInterval.bind(window);
          window.setInterval = (callback, delay, ...args) =>
            delay === 4000 ? 0 : nativeSetInterval(callback, delay, ...args);
        }""",
    )
    page.goto(base_url, wait_until="domcontentloaded")
    page.wait_for_function("document.querySelector('#system-state')?.textContent === 'System live'")
    page.evaluate(
        """() => {
          state.paused = true;
          const lockSnapshotStatus = () => {
            const systemState = document.querySelector('#system-state');
            const lastSync = document.querySelector('#last-sync');
            const liveDot = document.querySelector('#live-dot');
            const pauseButton = document.querySelector('#pause-button');
            if (systemState && systemState.textContent !== 'Presentation snapshot') {
              systemState.textContent = 'Presentation snapshot';
            }
            if (lastSync && lastSync.textContent !== 'Seeded case / feed paused') {
              lastSync.textContent = 'Seeded case / feed paused';
            }
            if (liveDot) liveDot.className = 'live-dot is-live';
            if (pauseButton) {
              pauseButton.textContent = 'Resume feed';
              pauseButton.setAttribute('aria-pressed', 'true');
            }
          };
          lockSnapshotStatus();
          window.setInterval(lockSnapshotStatus, 250);
        }""",
    )
    install_card_style(page)
    install_guided_cursor(page)

    card(
        page,
        "00 / 07",
        "PacketFlowAI v5",
        "Know what happened. Know what is missing.",
        "Verifiable Assurance separates a dangerous-looking incident from the strength of the "
        "record supporting it. This tour follows both, without pretending uncertainty disappeared.",
        [
            ("Threat", "Measure what the activity may mean."),
            ("Assurance", "Measure what the evidence earned."),
            ("Boundary", "State what still cannot be proved."),
        ],
        hold_seconds,
    )
    move_to(page, page.locator(".risk-panel"), "THREAT RISK")
    hold(page, hold_seconds)
    smooth_scroll(page, page.locator(".topbar"), offset=0)
    guided_click(page, page.locator("button[data-view='assurance']"), "OPEN ASSURANCE")
    page.wait_for_function("document.querySelector('#assurance-level')?.textContent === 'A4'")
    page.evaluate("window.scrollTo(0, 0)")
    hold(page, hold_seconds)

    card(
        page,
        "01 / 07",
        "Evidence assurance vector",
        "Integrity is not completeness.",
        "Instead of one reassuring percentage, v5 reports integrity, inclusion, continuity, "
        "contract coverage, liveness, attestation, anchoring, re-derivation, and known gaps separately.",
        [
            ("Verified", "Cryptographic or deterministic claims reproduced."),
            ("Partial", "A scoped property has known limitations."),
            ("Not eliminated", "Unknown omission risk remains explicit."),
        ],
        hold_seconds,
    )
    move_to(page, page.locator(".assurance-level"), "ASSURANCE LEVEL A4")
    hold(page, hold_seconds)
    move_to(page, page.locator(".risk-duality"), "THREAT VS ASSURANCE")
    hold(page, hold_seconds)
    smooth_scroll(page, page.locator(".vector-panel"), offset=85)
    move_to(page, page.locator("#assurance-vector"), "EARNED TRUST VECTOR")
    hold(page, hold_seconds)
    move_to(page, page.locator(".omission-warning"), "UNKNOWN OMISSION RISK")
    hold(page, hold_seconds)

    card(
        page,
        "02 / 07",
        "Proof explorer",
        "Follow verification, step by step.",
        "Every supplied event can be traced from its hash through a Merkle path, epoch checkpoint, "
        "independent witness receipt, and external anchor.",
        [
            ("Inclusion", "Prove a supplied record was committed."),
            ("Witnessed", "Move trust outside one keeper."),
            ("Portable", "Verify the same path from a .pfcase export."),
        ],
        hold_seconds,
    )
    smooth_scroll(page, page.locator(".proof-panel"), offset=85)
    guided_click(page, page.get_by_role("button", name="Merkle path"), "TRACE MERKLE PATH")
    hold(page, hold_seconds)
    guided_click(page, page.get_by_role("button", name="Signed checkpoint"), "SIGNED CHECKPOINT")
    hold(page, hold_seconds)
    guided_click(page, page.get_by_role("button", name="Witness receipt"), "INDEPENDENT WITNESS")
    hold(page, hold_seconds)
    guided_click(page, page.get_by_role("button", name="External anchor"), "EXTERNAL ANCHOR")
    hold(page, hold_seconds)

    card(
        page,
        "03 / 07",
        "Observation plane",
        "Investigate the machinery observing reality.",
        "Signed heartbeats distinguish no observed events from a sensor that was not observing. "
        "Dark periods, clock skew, transport loss, and cross-source asymmetry become forensic signals.",
        [
            ("Liveness", "See exactly when a source went dark."),
            ("Path health", "Trace produced, transported, normalized, and committed counts."),
            ("World model", "Connect hosts, sensors, ingest, witnesses, and anchors."),
        ],
        hold_seconds,
    )
    smooth_scroll(page, page.locator(".heatmap-panel"), offset=85)
    move_to(page, page.locator("#assurance-heatmap"), "OBSERVATION HEATMAP")
    hold(page, hold_seconds)
    guided_click(page, page.locator(".heat-cell.dark").first, "INSPECT DARK PERIOD")
    hold(page, hold_seconds)
    smooth_scroll(page, page.locator(".observation-panel"), offset=85)
    move_to(page, page.locator(".observation-stage"), "ASSURANCE WORLD MODEL")
    hold(page, hold_seconds)
    smooth_scroll(page, page.locator(".path-panel"), offset=85)
    move_to(page, page.locator("#recording-path"), "TRACE RECORDING LOSS")
    hold(page, hold_seconds)

    card(
        page,
        "04 / 07",
        "Formal claims",
        "Say exactly what was proved.",
        "PF-INTEGRITY, PF-INCLUSION, PF-SEQUENCE, PF-COVERAGE, PF-ANCHOR, and PF-REPRODUCE "
        "replace vague green states with narrowly scoped statements and limitations.",
        [
            ("Known", "Show conclusions supported by the record."),
            ("Missing", "Reason-code unavailable expected sources."),
            ("Unknown", "Never convert absence of observation into observed absence."),
        ],
        hold_seconds,
    )
    smooth_scroll(page, page.locator(".claims-panel"), offset=85)
    move_to(page, page.locator("#formal-claims"), "FORMAL ASSURANCE CLAIMS")
    hold(page, hold_seconds)
    smooth_scroll(page, page.locator(".know-panel"), offset=85)
    move_to(page, page.locator("#what-we-know"), "WHAT WE KNOW")
    hold(page, hold_seconds)
    move_to(page, page.locator("#what-we-cannot-prove"), "WHAT WE CANNOT PROVE")
    hold(page, hold_seconds)

    card(
        page,
        "05 / 07",
        "Re-derivation boundary",
        "Reproduce deterministic work. Attest the rest.",
        "Flow features, HDC encoding, anomaly scoring, and policy can be re-derived. NIM reasoning "
        "is preserved with integrity, but v5 never promises identical regenerated prose.",
        [
            ("Capsule", "Package evidence, model, policy, authority, and result."),
            ("Receipt", "Hash the exact reasoning request and response used."),
            ("Authority", "Require humans when necessary evidence is absent."),
        ],
        hold_seconds,
    )
    smooth_scroll(page, page.locator(".rederive-panel"), offset=85)
    move_to(page, page.locator("#rederivation-status"), "RE-DERIVATION STATUS")
    hold(page, hold_seconds)
    move_to(page, page.locator("#assurance-authority"), "ASSURANCE-AWARE AUTHORITY")
    hold(page, hold_seconds)
    move_to(page, page.locator("#assurance-debt"), "OPEN ASSURANCE DEBT")
    hold(page, hold_seconds)

    card(
        page,
        "06 / 07",
        "Independent challenge",
        "Attack the attack record.",
        "Challenge mode tests deletion, modification, reordered sequences, missing epochs, changed "
        "models and policies, forged timestamps, hidden redactions, and transparency split views.",
        [
            ("Detect", "Map every mutation to the control that catches it."),
            ("Witness", "Reconcile checkpoints across independent services."),
            ("Contract", "Commit to expected evidence before the incident."),
        ],
        hold_seconds,
    )
    smooth_scroll(page, page.locator(".attack-record-panel"), offset=85)
    move_to(page, page.locator("#assurance-attack-lab"), "EVIDENCE CHAOS LAB")
    hold(page, hold_seconds)
    move_to(page, page.locator("#evidence-contract"), "SIGNED EVIDENCE CONTRACT")
    hold(page, hold_seconds)
    move_to(page, page.locator("#witness-status"), "WITNESS RECONCILIATION")
    hold(page, hold_seconds)

    smooth_scroll(page, page.locator(".topbar"), offset=0)
    guided_click(page, page.locator("button[data-view='forensics']"), "OPEN FORENSICS")
    page.wait_for_function("Number(document.querySelector('#forensic-case-count')?.textContent) > 0")
    hold(page, hold_seconds)
    guided_click(page, page.locator("#forensic-next"), "FOLLOW FLAGGED EVIDENCE")
    hold(page, hold_seconds)
    guided_click(page, page.locator("button[data-view='assurance']"), "RETURN TO ASSURANCE")
    page.wait_for_function("document.querySelector('#assurance-level')?.textContent === 'A4'")
    page.evaluate("window.scrollTo(0, 0)")
    hold(page, hold_seconds)

    card(
        page,
        "07 / 07",
        "PacketFlowAI v5",
        "Prove what you can. Show what you cannot.",
        "Evidence Contracts, witnessed commitments, visible omissions, reproducible decisions, and "
        "independent PFCASE verification let the evidence outlive the system that created it.",
        [
            ("Know", "Understand what the supplied record supports."),
            ("Missing", "Make controlled absence and blind spots visible."),
            ("Prove", "Verify without trusting PacketFlowAI."),
        ],
        hold_seconds,
        dismiss=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--hold-seconds", type=float, default=DEFAULT_HOLD_SECONDS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.hold_seconds < MINIMUM_HOLD_SECONDS:
        raise SystemExit(f"--hold-seconds must be at least {MINIMUM_HOLD_SECONDS:g}")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="packetflowai-v5-tour-") as directory:
        root = Path(directory)
        config = replace(AppConfig(), artifacts=ArtifactPaths(root / "artifacts"))
        config = replace(config, nim=replace(config.nim, mode="shadow"))
        config.artifacts.create()
        store = EventStore(config.artifacts.event_database)
        metrics = MetricsRegistry()
        registry = FilesystemModelRegistry(config.artifacts.registry)
        seed_presentation(store, registry, metrics, root)
        server = APIServer(config, store, metrics, registry, port=0)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        for _ in range(100):
            if server.server:
                break
            time.sleep(0.02)
        if not server.server:
            raise RuntimeError("presentation server did not start")
        try:
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(headless=True)
                context = browser.new_context(
                    viewport={"width": 1600, "height": 900},
                    record_video_dir=root / "video",
                    record_video_size={"width": 1600, "height": 900},
                    device_scale_factor=1,
                )
                page = context.new_page()
                video = page.video
                record_v5_tour(page, f"http://127.0.0.1:{server.server.server_address[1]}", args.hold_seconds)
                context.close()
                if video is None:
                    raise RuntimeError("Playwright did not initialize video recording")
                video.save_as(output)
                browser.close()
        finally:
            server.stop()
            thread.join(3)
            store.close()
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError("video was not created")
    print(f"Created {output} ({output.stat().st_size / 1024 / 1024:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
