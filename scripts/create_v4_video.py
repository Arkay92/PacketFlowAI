"""Record the PacketFlowAI v4 provable collective-defence walkthrough."""

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

DEFAULT_OUTPUT = Path("artifacts/presentation/packetflowai-v4-collective-defence-tour.webm")
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


def select_graph_entity(page: Page) -> None:
    canvas = page.locator("#world-canvas")
    move_to(page, canvas, "SELECT GRAPH ENTITY")
    box = canvas.bounding_box()
    point = page.evaluate(
        "worldField.points.find(item => item.node.kind === 'TECHNIQUE') || worldField.points[0]",
    )
    if box and point:
        x = box["x"] + point["x"]
        y = box["y"] + point["y"]
        page.mouse.move(x, y, steps=32)
        page.mouse.click(x, y)


def record_v4_tour(page: Page, base_url: str, hold_seconds: float) -> None:
    page.add_init_script(
        """() => {
          const nativeSetInterval = window.setInterval.bind(window);
          window.setInterval = (callback, delay, ...args) =>
            delay === 4000 ? 0 : nativeSetInterval(callback, delay, ...args);
        }""",
    )
    page.goto(base_url, wait_until="domcontentloaded")
    page.wait_for_function("document.querySelector('#system-state')?.textContent === 'System live'")
    page.evaluate("state.paused = true")
    install_card_style(page)
    install_guided_cursor(page)

    card(
        page,
        "00 / 06",
        "PacketFlowAI v4",
        "Provable collective defence.",
        "A guided journey from live evidence to campaign intelligence, challenged decisions, "
        "collective learning, and independently verifiable forensics.",
        [
            ("Observe", "Unify evidence across the environment."),
            ("Challenge", "Test predictions and interventions."),
            ("Prove", "Export records that do not require trust."),
        ],
        hold_seconds,
    )
    move_to(page, page.locator(".network-panel"), "LIVE SENSOR FABRIC")
    hold(page, hold_seconds)
    smooth_scroll(page, page.locator(".incident-panel"), offset=90)
    move_to(page, page.locator(".incident").first, "CORRELATED SIGNAL")
    hold(page, hold_seconds)

    card(
        page,
        "01 / 06",
        "Campaign intelligence",
        "Stop investigating isolated alerts.",
        "Flows, hosts, identities, services, alerts, ATT&CK techniques, and cases become one "
        "temporal world model with causal alternatives and future branches.",
        [
            ("Campaign", "Connect the full attacker journey."),
            ("Causal", "Separate sequence from plausible enablement."),
            ("Predictive", "Compare 5-minute, 1-hour, and 24-hour horizons."),
        ],
        hold_seconds,
    )
    smooth_scroll(page, page.locator(".topbar"), offset=0)
    guided_click(page, page.locator("button[data-view='command']"), "OPEN COMMAND")
    page.wait_for_function("document.querySelector('#world-counts')?.textContent.includes('nodes')")
    page.evaluate("window.scrollTo(0, 0)")
    hold(page, hold_seconds)
    select_graph_entity(page)
    hold(page, hold_seconds)
    move_to(page, page.locator(".forecast-module"), "PREDICT NEXT MOVE")
    hold(page, hold_seconds)

    card(
        page,
        "02 / 06",
        "Counterfactual defence",
        "Before acting, change the future safely.",
        "The digital twin compares threat reduction, legitimate disruption, dependencies, "
        "blast radius, reversibility, and residual risk before authority is requested.",
        [
            ("Compare", "Block, limit, isolate, or observe."),
            ("Optimise", "Find the least disruptive safe action."),
            ("Reversible", "Prefer actions that can be rolled back."),
        ],
        hold_seconds,
    )
    smooth_scroll(page, page.locator(".simulation-module"), offset=85)
    move_to(page, page.locator(".simulation-options"), "COMPARE SCENARIOS")
    hold(page, hold_seconds)
    guided_click(page, page.locator(".simulation-card", has_text="RATE LIMIT"), "SIMULATE RATE LIMIT")
    hold(page, hold_seconds)
    guided_click(page, page.locator(".simulation-card", has_text="ISOLATE TARGET"), "SIMULATE ISOLATION")
    hold(page, hold_seconds)
    move_to(page, page.locator(".twin-orbit"), "REVIEW BLAST RADIUS")
    hold(page, hold_seconds)

    card(
        page,
        "03 / 06",
        "Provable forensics",
        "Do not trust the record. Verify it.",
        "The time machine preserves the knowledge boundary. Hash chains, Merkle roots, witnesses, "
        "external anchors, and portable .pfcase bundles allow independent verification.",
        [
            ("Known then", "Later evidence stays outside historical reasoning."),
            ("Sealed", "Missing or modified records are detectable."),
            ("Independent", "packetflow-verifier shares no application code."),
        ],
        hold_seconds,
    )
    smooth_scroll(page, page.locator(".time-module"), offset=85)
    guided_click(page, page.locator("#time-slider"), "REWIND EVIDENCE")
    page.keyboard.press("Home")
    hold(page, hold_seconds)
    page.keyboard.press("End")
    hold(page, hold_seconds)
    move_to(page, page.locator(".integrity-module"), "VERIFY INTEGRITY")
    hold(page, hold_seconds)
    smooth_scroll(page, page.locator(".authority-module"), offset=85)
    move_to(page, page.locator(".authority-ladder"), "TRACE AUTHORITY")
    hold(page, hold_seconds)

    card(
        page,
        "04 / 06",
        "Assurance deck",
        "Challenge the leading story.",
        "Causal links, earliest intervention, missed authority opportunities, minimum safe action, "
        "and missing sensor context make every consequential decision reviewable.",
        [
            ("Why", "Trace action back to policy and evidence."),
            ("Why not", "Expose authority and business constraints."),
            ("Devil's advocate", "Search for the strongest counterargument."),
        ],
        hold_seconds,
    )
    smooth_scroll(page, page.locator(".assurance-module"), offset=85)
    move_to(page, page.locator(".assurance-grid"), "DECISION ASSURANCE")
    hold(page, hold_seconds)
    smooth_scroll(page, page.locator(".platform-module"), offset=85)
    move_to(page, page.locator(".platform-domains"), "COLLECTIVE DEFENCE")
    hold(page, hold_seconds)

    card(
        page,
        "05 / 06",
        "Collective defence",
        "One environment learns. Every environment improves.",
        "Threat memory, poisoning-resistant federation, OCSF, STIX/TAXII, Sigma, adaptive sensing, "
        "Linux fast paths, and grounded reasoning operate through explicit trust boundaries.",
        [
            ("Private", "Share fingerprints, never raw traffic."),
            ("Interoperable", "Meet the SOC where it already works."),
            ("Adaptive", "Preserve uncertain and high-risk evidence under load."),
        ],
        hold_seconds,
    )
    move_to(page, page.locator(".runtime-posture"), "RUNTIME POSTURE")
    hold(page, hold_seconds)
    smooth_scroll(page, page.locator(".topbar"), offset=0)
    guided_click(page, page.locator("button[data-view='forensics']"), "OPEN FORENSICS")
    page.wait_for_function("Number(document.querySelector('#forensic-case-count')?.textContent) > 0")
    hold(page, hold_seconds)
    guided_click(page, page.locator("#forensic-next"), "FOLLOW SEALED EVIDENCE")
    hold(page, hold_seconds)

    card(
        page,
        "06 / 06",
        "PacketFlowAI v4",
        "Observe. Remember. Challenge. Prove.",
        "A neuro-symbolic cyber world model that predicts likely progression, simulates defence, "
        "acts under explicit authority, and produces evidence a hostile third party can verify.",
        [
            ("Campaign", "Understand what is happening."),
            ("Authority", "Control what may happen next."),
            ("Evidence", "Prove exactly why it happened."),
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
    with tempfile.TemporaryDirectory(prefix="packetflowai-v4-tour-") as directory:
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
                record_v4_tour(page, f"http://127.0.0.1:{server.server.server_address[1]}", args.hold_seconds)
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
