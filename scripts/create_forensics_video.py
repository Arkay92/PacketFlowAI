"""Record an interactive analyst journey through the Forensic War Room."""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from threading import Thread

from playwright.sync_api import Locator, Page, sync_playwright

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from create_ui_video import (  # noqa: E402
    MINIMUM_HOLD_SECONDS,
    install_card_style,
    seed_presentation,
    show_card,
)

from packetflowai.api import APIServer  # noqa: E402
from packetflowai.config import AppConfig, ArtifactPaths  # noqa: E402
from packetflowai.registry import FilesystemModelRegistry  # noqa: E402
from packetflowai.storage import EventStore  # noqa: E402
from packetflowai.telemetry import MetricsRegistry  # noqa: E402

DEFAULT_OUTPUT = Path("artifacts/presentation/packetflowai-forensics-war-room-tour.webm")
DEFAULT_HOLD_SECONDS = 5.0
SCROLL_DURATION_MS = 1800
CURSOR_CSS = """
#pf-guided-cursor {
  position: fixed;
  z-index: 2000;
  left: 0;
  top: 0;
  width: 22px;
  height: 22px;
  border: 1px solid #ffbf69;
  border-radius: 50%;
  pointer-events: none;
  transform: translate(-50%, -50%);
  box-shadow: 0 0 18px rgba(255,191,105,.45);
  transition: width 140ms, height 140ms, background 140ms;
}
#pf-guided-cursor::before {
  content: "";
  position: absolute;
  inset: 7px;
  border-radius: 50%;
  background: #ffbf69;
}
#pf-guided-cursor.is-clicking {
  width: 36px;
  height: 36px;
  background: rgba(255,191,105,.13);
}
#pf-cursor-label {
  position: absolute;
  left: 24px;
  top: -10px;
  width: max-content;
  padding: 5px 7px;
  border: 1px solid rgba(255,191,105,.3);
  color: #ffbf69;
  background: rgba(8,11,12,.92);
  font: 8px "Bahnschrift", monospace;
  letter-spacing: .09em;
  text-transform: uppercase;
}
"""


def install_guided_cursor(page: Page) -> None:
    page.add_style_tag(content=CURSOR_CSS)
    page.evaluate(
        """() => {
          const cursor = document.createElement('div');
          cursor.id = 'pf-guided-cursor';
          cursor.style.left = '120px';
          cursor.style.top = '120px';
          const label = document.createElement('span');
          label.id = 'pf-cursor-label';
          label.textContent = 'ANALYST';
          cursor.append(label);
          document.body.append(cursor);
          document.addEventListener('mousemove', event => {
            cursor.style.left = `${event.clientX}px`;
            cursor.style.top = `${event.clientY}px`;
          });
        }""",
    )


def move_to(page: Page, target: Locator, label: str = "ANALYST") -> tuple[float, float]:
    target.scroll_into_view_if_needed()
    box = target.bounding_box()
    if box is None:
        raise RuntimeError("target is not visible")
    x = box["x"] + box["width"] / 2
    y = box["y"] + box["height"] / 2
    page.locator("#pf-cursor-label").evaluate("(node, value) => node.textContent = value", label)
    page.mouse.move(x, y, steps=36)
    page.wait_for_timeout(650)
    return x, y


def guided_click(page: Page, target: Locator, label: str) -> None:
    x, y = move_to(page, target, label)
    page.locator("#pf-guided-cursor").evaluate("node => node.classList.add('is-clicking')")
    page.mouse.down()
    page.wait_for_timeout(240)
    page.mouse.up()
    page.wait_for_timeout(300)
    page.locator("#pf-guided-cursor").evaluate("node => node.classList.remove('is-clicking')")
    page.mouse.move(x + 16, y + 10, steps=10)


def smooth_scroll(page: Page, target: Locator, offset: int = 90) -> None:
    target.evaluate(
        """(node, options) => {
          const destination = node.getBoundingClientRect().top + window.scrollY - options.offset;
          const start = window.scrollY;
          const change = destination - start;
          const started = performance.now();
          const ease = value => value < .5 ? 2 * value * value : 1 - Math.pow(-2 * value + 2, 2) / 2;
          const animate = now => {
            const progress = Math.min(1, (now - started) / options.duration);
            window.scrollTo(0, start + change * ease(progress));
            if (progress < 1) requestAnimationFrame(animate);
          };
          requestAnimationFrame(animate);
        }""",
        {"duration": SCROLL_DURATION_MS, "offset": offset},
    )
    page.wait_for_timeout(SCROLL_DURATION_MS + 350)


def hold(page: Page, seconds: float) -> None:
    page.wait_for_timeout(seconds * 1000)


def tour_card(
    page: Page,
    index: str,
    eyebrow: str,
    title: str,
    description: str,
    points: list[tuple[str, str]],
    hold_seconds: float,
    dismiss: bool = True,
) -> None:
    page.locator("#pf-guided-cursor").evaluate("node => node.style.opacity = '0'")
    show_card(page, index, eyebrow, title, description, points, hold_seconds, dismiss=dismiss)
    if dismiss:
        page.locator("#pf-guided-cursor").evaluate("node => node.style.opacity = '1'")


def record_forensics_tour(page: Page, base_url: str, hold_seconds: float) -> None:
    page.goto(base_url, wait_until="networkidle")
    page.wait_for_function("document.querySelector('#system-state')?.textContent === 'System live'")
    page.evaluate("state.paused = true")
    install_card_style(page)
    install_guided_cursor(page)

    tour_card(
        page, "00 / 05", "Signal to evidence", "Follow the incident.",
        "An analyst journey from live network pressure into the new Forensic War Room, then safely back to command.",
        [("Detect", "Start from live operational signals."), ("Investigate", "Navigate real flagged cases."),
         ("Return", "Preserve the wider operating picture.")], hold_seconds,
    )

    move_to(page, page.locator(".risk-panel"), "THREAT PRESSURE")
    hold(page, hold_seconds)
    smooth_scroll(page, page.locator(".incident-panel"), offset=100)
    move_to(page, page.locator(".incident").first, "FLAGGED INCIDENT")
    hold(page, hold_seconds)

    tour_card(
        page, "01 / 05", "Escalation path", "From signal to case file.",
        "Orange and red conversations are correlated across flows, decisions, alerts, evidence, and NIM context.",
        [("Orange", "Suspicious activity for review."), ("Red", "High-confidence policy signals."),
         ("Correlated", "One event ID joins the record.")], hold_seconds,
    )

    smooth_scroll(page, page.locator(".topbar"), offset=0)
    guided_click(page, page.locator("button[data-view='forensics']"), "OPEN FORENSICS")
    page.wait_for_function("document.querySelector('#forensic-case-count').textContent === '5'")
    hold(page, hold_seconds)

    move_to(page, page.locator("#forensic-canvas"), "THREAT CONSTELLATION")
    hold(page, hold_seconds)
    guided_click(page, page.locator(".forensic-case-button.orange").first, "SELECT ORANGE")
    hold(page, hold_seconds)
    guided_click(page, page.locator("#forensic-next"), "NEXT CASE")
    hold(page, hold_seconds)
    guided_click(page, page.locator(".forensic-case-button.red").nth(1), "SELECT RED")
    hold(page, hold_seconds)

    tour_card(
        page, "02 / 05", "Active dossier", "Identity before interpretation.",
        "The analyst can move between cases while route, protocol, risk, policy, and response context update together.",
        [("Navigate", "Constellation, case rail, or arrow keys."), ("Compare", "Orange and red severity states."),
         ("Bounded", "Policy remains visible and reviewable.")], hold_seconds,
    )

    smooth_scroll(page, page.locator(".forensic-details"), offset=90)
    move_to(page, page.locator(".packet-balance"), "DIRECTIONAL VOLUME")
    hold(page, hold_seconds)
    smooth_scroll(page, page.locator(".forensic-evidence"), offset=90)
    move_to(page, page.locator(".forensic-evidence-card").nth(2), "ANOMALY CHANNEL")
    hold(page, hold_seconds)

    tour_card(
        page, "03 / 05", "Evidence depth", "Fused, not flattened.",
        "Packet anatomy, directional volume, local confidence, HDC similarity, anomaly score, and NIM stay distinct.",
        [("Anatomy", "Rate, volume, retransmits, spread."), ("Evidence", "Every channel retains meaning."),
         ("Provenance", "The raw joined record remains inspectable.")], hold_seconds,
    )

    smooth_scroll(page, page.locator(".forensic-raw"), offset=90)
    move_to(page, page.locator("#forensic-metadata"), "PROTOCOL METADATA")
    hold(page, hold_seconds)
    move_to(page, page.locator("#forensic-raw-json"), "READ-ONLY RECORD")
    page.mouse.wheel(0, 420)
    hold(page, hold_seconds)

    tour_card(
        page, "04 / 05", "Return to command", "Investigation stays connected.",
        "After reviewing the case, the analyst returns to Signal Room with the wider operational context intact.",
        [("No dead end", "Forensics is part of the live workflow."), ("No hidden state", "Selection is explicit."),
         ("No auto-action", "Humans retain authority.")], hold_seconds,
    )

    smooth_scroll(page, page.locator(".topbar"), offset=0)
    guided_click(page, page.locator("button[data-view='all']"), "BACK TO OVERVIEW")
    page.evaluate("window.scrollTo(0, 0)")
    hold(page, hold_seconds)

    tour_card(
        page, "05 / 05", "PacketFlowAI", "See it. Trace it. Understand it.",
        "The Forensic War Room turns every serious signal into a navigable, evidence-led analyst journey.",
        [("Live", "Built into Signal Room."), ("Deep", "Full correlated case context."),
         ("Controlled", "Operator-led from start to finish.")], hold_seconds, dismiss=False,
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

    with tempfile.TemporaryDirectory(prefix="packetflowai-forensics-tour-") as directory:
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
                record_forensics_tour(
                    page,
                    f"http://127.0.0.1:{server.server.server_address[1]}",
                    args.hold_seconds,
                )
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
