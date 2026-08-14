"""Record an interactive PacketFlowAI v3 predictive-defence walkthrough."""

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

DEFAULT_OUTPUT = Path("artifacts/presentation/packetflowai-v3-predictive-defence-tour.webm")
DEFAULT_HOLD_SECONDS = 5.0


def record_v3_tour(page: Page, base_url: str, hold_seconds: float) -> None:
    page.goto(base_url, wait_until="networkidle")
    page.wait_for_function("document.querySelector('#system-state')?.textContent === 'System live'")
    page.evaluate("state.paused = true")
    install_card_style(page)
    install_guided_cursor(page)

    tour_card(
        page, "00 / 07", "PacketFlowAI v3", "From incident to prediction.",
        "Follow a live signal into a temporal threat world model, test the response, "
        "prove authority, and return to the evidence.",
        [("Understand", "Correlate the whole campaign."), ("Predict", "Estimate plausible next moves."),
         ("Simulate", "Compare impact before action.")], hold_seconds,
    )
    move_to(page, page.locator(".network-panel"), "LIVE SIGNAL ROOM")
    hold(page, hold_seconds)
    smooth_scroll(page, page.locator(".incident-panel"), offset=90)
    move_to(page, page.locator(".incident").first, "CAMPAIGN SIGNAL")
    hold(page, hold_seconds)

    tour_card(
        page, "01 / 07", "The journey", "Signals become a world model.",
        "PacketFlowAI joins flows, hosts, identities, services, alerts, ATT&CK techniques, "
        "and cases instead of treating alerts as isolated rows.",
        [("Persistent", "Graph state is stored in SQLite."), ("Temporal", "PRECEDED edges retain sequence."),
         ("Correlated", "Events resolve into campaigns.")], hold_seconds,
    )
    smooth_scroll(page, page.locator(".topbar"), offset=0)
    guided_click(page, page.locator("button[data-view='command']"), "OPEN COMMAND")
    page.wait_for_function("document.querySelector('#world-counts')?.textContent.includes('nodes')")
    page.evaluate("window.scrollTo(0, 0)")
    hold(page, hold_seconds)

    move_to(page, page.locator("#world-canvas"), "THREAT WORLD MODEL")
    graph_box = page.locator("#world-canvas").bounding_box()
    graph_point = page.evaluate(
        "worldField.points.find(point => point.node.kind === 'TECHNIQUE') || worldField.points[0]",
    )
    if graph_box and graph_point:
        page.mouse.move(
            graph_box["x"] + graph_point["x"],
            graph_box["y"] + graph_point["y"],
            steps=30,
        )
        page.mouse.click(graph_box["x"] + graph_point["x"], graph_box["y"] + graph_point["y"])
    hold(page, hold_seconds)
    move_to(page, page.locator(".forecast-module"), "NEXT MOVE")
    hold(page, hold_seconds)

    tour_card(
        page, "02 / 07", "Prediction", "What is likely to happen next?",
        "Observed ATT&CK sequences produce bounded next-move probabilities with a time horizon, "
        "supporting event IDs, and explicit residual uncertainty.",
        [("Progression", "Sequence-aware technique transitions."), ("Evidence", "Every forecast points backward."),
         ("Uncertainty", "No false certainty is hidden.")], hold_seconds,
    )
    smooth_scroll(page, page.locator(".simulation-module"), offset=85)
    move_to(page, page.locator(".simulation-options"), "COMPARE RESPONSES")
    hold(page, hold_seconds)
    guided_click(page, page.locator(".simulation-card", has_text="RATE LIMIT"), "TEST RATE LIMIT")
    hold(page, hold_seconds)
    guided_click(page, page.locator(".simulation-card", has_text="ISOLATE TARGET"), "TEST ISOLATION")
    hold(page, hold_seconds)
    move_to(page, page.locator(".twin-orbit"), "DIGITAL TWIN")
    hold(page, hold_seconds)

    tour_card(
        page, "03 / 07", "Counterfactual defence", "Simulate before authority.",
        "The analyst compares risk reduction, business impact, disrupted flows, dependency exposure, "
        "blast radius, and evidence gain before choosing a response.",
        [("Alternatives", "Block, limit, isolate, or observe."), ("Impact", "Threat paths and legitimate traffic."),
         ("Control", "Recommendation is not execution.")], hold_seconds,
    )
    smooth_scroll(page, page.locator(".time-module"), offset=85)
    move_to(page, page.locator("#time-slider"), "EVIDENCE TIME MACHINE")
    guided_click(page, page.locator("#time-slider"), "REWIND KNOWLEDGE")
    page.keyboard.press("Home")
    hold(page, hold_seconds)
    page.keyboard.press("ArrowRight")
    page.keyboard.press("ArrowRight")
    hold(page, hold_seconds)
    page.keyboard.press("End")
    hold(page, hold_seconds)
    move_to(page, page.locator(".integrity-module"), "VERIFY RECORD")
    hold(page, hold_seconds)

    tour_card(
        page, "04 / 07", "Evidence accountability", "Reconstruct and verify.",
        "The time machine shows only what was known at that moment. A hash chain and Merkle root "
        "detect later modification of evidence, decisions, policy, or model provenance.",
        [("As-of", "Future evidence stays hidden."), ("Sealed", "Consequential records are hash chained."),
         ("Verified", "Integrity is visible in the room.")], hold_seconds,
    )
    smooth_scroll(page, page.locator(".authority-module"), offset=85)
    move_to(page, page.locator(".authority-ladder"), "AUTHORITY GRAPH")
    hold(page, hold_seconds)
    smooth_scroll(page, page.locator(".capability-module"), offset=85)
    move_to(page, page.locator(".capability-rail"), "DEFENCE FABRIC")
    page.mouse.wheel(0, 260)
    hold(page, hold_seconds)

    tour_card(
        page, "05 / 07", "Explicit authority", "Prove why action is permitted.",
        "Autonomous observation and alerting stop at clear policy boundaries. Higher-impact actions "
        "expose approver role, scope, expiry, rollback, and policy provenance.",
        [("Bounded", "Playbooks cannot cross authority."), ("Adaptive", "Sensor fidelity follows risk."),
         ("Extensible", "Federation and fast paths are explicit contracts.")], hold_seconds,
    )
    smooth_scroll(page, page.locator(".topbar"), offset=0)
    guided_click(page, page.locator("button[data-view='forensics']"), "OPEN FORENSICS")
    page.wait_for_function("Number(document.querySelector('#forensic-case-count')?.textContent) > 0")
    hold(page, hold_seconds)
    guided_click(page, page.locator("#forensic-next"), "FOLLOW EVIDENCE")
    hold(page, hold_seconds)

    tour_card(
        page, "06 / 07", "One operating picture", "Prediction meets forensics.",
        "The predictive command cell and Forensic War Room share the same preserved evidence, "
        "allowing an analyst to move from campaign outlook back to packet-level proof.",
        [("Campaign", "See connected behaviour."), ("Packet", "Inspect the original record."),
         ("Return", "Keep the wider context intact.")], hold_seconds,
    )
    smooth_scroll(page, page.locator(".topbar"), offset=0)
    guided_click(page, page.locator("button[data-view='command']"), "BACK TO COMMAND")
    page.evaluate("window.scrollTo(0, 0)")
    hold(page, hold_seconds)
    guided_click(page, page.locator("button[data-view='all']"), "RETURN TO OVERVIEW")
    page.evaluate("window.scrollTo(0, 0)")
    hold(page, hold_seconds)

    tour_card(
        page, "07 / 07", "PacketFlowAI v3", "Understand. Predict. Prove.",
        "A temporal cyber world model that preserves evidence, predicts plausible progression, "
        "simulates defensive choices, and acts only through explicit authority.",
        [("World model", "Understand the campaign."), ("Next move", "Prepare for progression."),
         ("Authority", "Control and explain every action.")], hold_seconds, dismiss=False,
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
    with tempfile.TemporaryDirectory(prefix="packetflowai-v3-tour-") as directory:
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
            time.sleep(.02)
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
                record_v3_tour(page, f"http://127.0.0.1:{server.server.server_address[1]}", args.hold_seconds)
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
