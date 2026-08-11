"""Record a guided Signal Room presentation with Playwright."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Thread

from playwright.sync_api import Page, sync_playwright

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from packetflowai.api import APIServer  # noqa: E402
from packetflowai.config import AppConfig, ArtifactPaths  # noqa: E402
from packetflowai.domain import FeedbackRecord, FlowFeatures  # noqa: E402
from packetflowai.registry import FilesystemModelRegistry  # noqa: E402
from packetflowai.storage import EventStore  # noqa: E402
from packetflowai.telemetry import MetricsRegistry  # noqa: E402

DEFAULT_OUTPUT = Path("artifacts/presentation/packetflowai-signal-room-tour.webm")
MINIMUM_HOLD_SECONDS = 5.0
DEFAULT_HOLD_SECONDS = 6.0
TRANSITION_MS = 900

CARD_CSS = """
#pf-tour-card {
  --tour-paper: #dce7df;
  --tour-muted: #82928a;
  --tour-green: #9df6ae;
  position: fixed;
  inset: 0;
  z-index: 1000;
  display: grid;
  place-items: center;
  overflow: hidden;
  color: var(--tour-paper);
  background:
    radial-gradient(circle at 78% 12%, rgba(113, 201, 255, .10), transparent 30rem),
    radial-gradient(circle at 15% 78%, rgba(157, 246, 174, .08), transparent 32rem),
    #0b0f10;
  font-family: "Bahnschrift", "DIN Alternate", monospace;
  opacity: 0;
  transition: opacity 900ms ease;
}
#pf-tour-card::before {
  content: "";
  position: absolute;
  inset: 0;
  background-image:
    linear-gradient(rgba(255,255,255,.022) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255,255,255,.022) 1px, transparent 1px);
  background-size: 42px 42px;
  mask-image: linear-gradient(135deg, black, transparent 78%);
}
#pf-tour-card.is-visible { opacity: 1; }
#pf-tour-card.is-leaving { opacity: 0; }
.pf-tour-frame {
  position: relative;
  width: min(1120px, calc(100vw - 120px));
  min-height: 570px;
  display: grid;
  grid-template-columns: 150px 1fr;
  border: 1px solid rgba(205, 226, 211, .22);
  background: linear-gradient(145deg, rgba(19,26,27,.95), rgba(11,15,16,.92));
  box-shadow: 0 35px 90px rgba(0,0,0,.36), inset 0 1px rgba(255,255,255,.025);
}
.pf-tour-rail {
  padding: 34px 28px;
  border-right: 1px solid rgba(205, 226, 211, .14);
  color: var(--tour-green);
  font-size: 13px;
  letter-spacing: .12em;
}
.pf-tour-mark {
  width: 54px;
  height: 54px;
  display: grid;
  place-items: center;
  margin-bottom: 54px;
  border: 1px solid var(--tour-green);
  font-size: 18px;
  font-weight: 700;
}
.pf-tour-index { display: block; margin-top: 10px; color: var(--tour-muted); font-size: 10px; }
.pf-tour-copy { align-self: center; padding: 72px 80px; }
.pf-tour-eyebrow {
  margin: 0 0 22px;
  color: var(--tour-green);
  font-size: 11px;
  letter-spacing: .22em;
  text-transform: uppercase;
}
.pf-tour-copy h1 {
  max-width: 810px;
  margin: 0;
  font: 700 72px/.92 "Arial Narrow", "Aptos Display", sans-serif;
  letter-spacing: -.055em;
  text-transform: uppercase;
}
.pf-tour-copy p {
  max-width: 720px;
  margin: 30px 0 0;
  color: #aebbb3;
  font-size: 17px;
  line-height: 1.65;
}
.pf-tour-points {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 1px;
  margin-top: 42px;
  border: 1px solid rgba(205, 226, 211, .14);
  background: rgba(205, 226, 211, .14);
}
.pf-tour-points span {
  min-height: 84px;
  padding: 18px;
  background: #0c1112;
  color: var(--tour-muted);
  font-size: 10px;
  line-height: 1.5;
  letter-spacing: .08em;
  text-transform: uppercase;
}
.pf-tour-points b { display: block; margin-bottom: 8px; color: var(--tour-paper); font-size: 12px; }
.pf-tour-footer {
  position: absolute;
  right: 30px;
  bottom: 24px;
  color: var(--tour-muted);
  font-size: 9px;
  letter-spacing: .14em;
  text-transform: uppercase;
}
"""


def _timestamp(offset_seconds: int) -> str:
    return (datetime.now(UTC) - timedelta(seconds=offset_seconds)).isoformat()


def seed_presentation(
    store: EventStore,
    registry: FilesystemModelRegistry,
    metrics: MetricsRegistry,
    root: Path,
) -> None:
    labels = [
        "port_scan", "benign", "benign", "benign", "credential_attack", "benign",
        "data_exfiltration", "benign", "ddos", "benign", "unknown", "benign",
    ]
    ports = [443, 53, 22, 3389, 445, 8080, 443, 8443, 80, 25, 1433, 443]
    for index, (label, port) in enumerate(zip(labels, ports, strict=True)):
        event_id = f"flow-{9821 + index:05d}"
        malicious = label not in {"benign", "unknown"}
        risk = 18.0 + index * 2.4 if not malicious else 58.0 + index * 3.1
        flow = FlowFeatures(
            flow_id=event_id,
            source_ip=f"10.24.{2 + index % 3}.{31 + index}",
            destination_ip=f"172.18.{4 + index % 4}.{70 + index}",
            source_port=49152 + index * 37,
            destination_port=port,
            protocol="UDP" if port == 53 else "TCP",
            packet_count=18 + index * 11,
            byte_count=4200 + index * 8350,
            duration_seconds=0.24 + index * 0.31,
            packets_per_second=32.0 + index * 4.7,
            bytes_per_second=18000.0 + index * 7400,
            forward_packets=12 + index * 7,
            reverse_packets=6 + index * 4,
            forward_bytes=3000 + index * 5900,
            reverse_bytes=1200 + index * 2450,
            state="ESTABLISHED" if index % 4 else "CLOSED",
            protocol_metadata={"service": str(port), "sensor": "edge-west-02"},
        )
        created_at = _timestamp(index * 8)
        store.add_flow(flow, created_at)
        confidence = 0.93 - index * 0.012 if label != "unknown" else 0.46
        decision = {
            "event_id": event_id,
            "action": "alert" if malicious else "observe",
            "policy_level": 4 if risk >= 70 else 3 if malicious else 1,
            "risk_score": min(risk, 96.0),
            "explanation": (
                f"Fused local evidence identifies {label.replace('_', ' ')} behavior. "
                "Policy remains bounded and requires local confirmation before containment."
            ),
            "evidence": {
                "classifier_label": label,
                "classifier_confidence": confidence,
                "calibrated_confidence": confidence - 0.025,
                "prototype_similarity": 0.89 - index * 0.014,
                "anomaly_score": 0.81 if malicious else 0.14 + index * 0.017,
                "nim_reasoning_strength": 0.76 if malicious else None,
            },
        }
        store.add_decision(f"decision-{index:03d}", event_id, decision, created_at)
        store.add_evidence(event_id, "local_classifier", decision["evidence"], created_at)
        store.add_evidence(event_id, "hdc_prototype", {"similarity": 0.89 - index * 0.014}, created_at)
        store.add_evidence(event_id, "anomaly", {"score": decision["evidence"]["anomaly_score"]}, created_at)
        if malicious:
            alert = {
                "event_id": event_id,
                "policy_level": decision["policy_level"],
                "action": "alert",
                "target": flow.source_ip,
                "reason": f"{label.replace('_', ' ').title()} evidence exceeded the calibrated threshold.",
            }
            store.add_alert(f"alert-{index:03d}", event_id, alert, created_at)
            store.add_nim_assessment(
                event_id,
                {
                    "mode": "shadow",
                    "verdict": label,
                    "assessment": "Corroborating context only; local policy remains authoritative.",
                    "self_reported_confidence": 0.76,
                },
                created_at,
            )

    store.add_feedback(FeedbackRecord(
        event_id="flow-09825",
        model_prediction="credential_attack",
        analyst_label="credential_attack",
        analyst_id="analyst-07",
        adjudicated=True,
        notes="Confirmed against authentication telemetry.",
    ))

    artifact = root / "packet-hv-mlp.bin"
    report = root / "evaluation.json"
    artifact.write_bytes(b"packetflowai presentation artifact")
    report.write_text(json.dumps({"macro_f1": 0.934, "ece": 0.027}), encoding="utf-8")
    registry.register_candidate("packet-hv-mlp", "2.3.0", artifact)
    registry.mark_evaluated("packet-hv-mlp:2.3.0", report, shadow_validated=True)
    registry.promote("packet-hv-mlp:2.3.0")
    registry.register_candidate("packet-hv-mlp", "2.4.0-rc1", artifact)

    for name, value in {
        "packets_per_second": 18472.0,
        "flows_per_second": 128.4,
        "inference_latency_p95_ms": 12.8,
        "queue_depth": 7.0,
        "dropped_packets": 0.0,
        "traced_memory_bytes": 184.6 * 1024 * 1024,
    }.items():
        metrics.set(name, value)


def install_card_style(page: Page) -> None:
    page.add_style_tag(content=CARD_CSS)


def show_card(
    page: Page,
    index: str,
    eyebrow: str,
    title: str,
    description: str,
    points: list[tuple[str, str]],
    hold_seconds: float,
    dismiss: bool = True,
) -> None:
    page.evaluate(
        """({ index, eyebrow, title, description, points, hold }) => {
          document.getElementById('pf-tour-card')?.remove();
          const card = document.createElement('section');
          card.id = 'pf-tour-card';
          const frame = document.createElement('div');
          frame.className = 'pf-tour-frame';
          const rail = document.createElement('aside');
          rail.className = 'pf-tour-rail';
          const mark = document.createElement('div');
          mark.className = 'pf-tour-mark';
          mark.textContent = 'PF';
          const section = document.createElement('strong');
          section.textContent = 'SECTION';
          const number = document.createElement('span');
          number.className = 'pf-tour-index';
          number.textContent = index;
          rail.append(mark, section, number);
          const copy = document.createElement('div');
          copy.className = 'pf-tour-copy';
          const label = document.createElement('div');
          label.className = 'pf-tour-eyebrow';
          label.textContent = eyebrow;
          const heading = document.createElement('h1');
          heading.textContent = title;
          const body = document.createElement('p');
          body.textContent = description;
          const pointGrid = document.createElement('div');
          pointGrid.className = 'pf-tour-points';
          points.forEach(([name, detail]) => {
            const point = document.createElement('span');
            const strong = document.createElement('b');
            strong.textContent = name;
            point.append(strong, detail);
            pointGrid.append(point);
          });
          const footer = document.createElement('div');
          footer.className = 'pf-tour-footer';
          footer.textContent = `PacketFlowAI // Signal Room // Hold ${hold.toFixed(0)} sec`;
          copy.append(label, heading, body, pointGrid);
          frame.append(rail, copy, footer);
          card.append(frame);
          document.body.append(card);
          requestAnimationFrame(() => requestAnimationFrame(() => card.classList.add('is-visible')));
        }""",
        {
            "index": index,
            "eyebrow": eyebrow,
            "title": title,
            "description": description,
            "points": points,
            "hold": hold_seconds,
        },
    )
    page.wait_for_timeout(hold_seconds * 1000)
    if not dismiss:
        return
    page.locator("#pf-tour-card").evaluate("card => card.classList.add('is-leaving')")
    page.wait_for_timeout(TRANSITION_MS)
    page.locator("#pf-tour-card").evaluate("card => card.remove()")


def hold_view(page: Page, hold_seconds: float) -> None:
    page.wait_for_timeout(hold_seconds * 1000)


def record_tour(page: Page, base_url: str, hold_seconds: float) -> None:
    page.goto(base_url, wait_until="networkidle")
    page.wait_for_function("document.querySelector('#system-state')?.textContent === 'System live'")
    page.evaluate("state.paused = true")
    install_card_style(page)

    show_card(
        page, "00 / 06", "Live network observatory", "PacketFlowAI, made visible.",
        "A guided tour of every flow, decision, evidence channel, and bounded response in one operating picture.",
        [("Observe", "See traffic become flows."), ("Explain", "Inspect fused evidence."),
         ("Respond", "Keep policy conservative.")], hold_seconds,
    )
    page.locator("button[data-view='all']").click()
    page.evaluate("window.scrollTo({top: 0, behavior: 'smooth'})")
    hold_view(page, hold_seconds)

    show_card(
        page, "01 / 06", "Traffic view", "Every conversation mapped.",
        "Bidirectional flows are resolved into a live topology and a searchable ledger without losing "
        "protocol context.",
        [("Topology", "Animated routes and endpoints."), ("Flow ledger", "Address, protocol, volume, state."),
         ("Shared path", "Capture and replay use one pipeline.")], hold_seconds,
    )
    page.locator("button[data-view='traffic']").click()
    page.locator(".network-panel").scroll_into_view_if_needed()
    hold_view(page, hold_seconds)

    show_card(
        page, "02 / 06", "Threat view", "Pressure, not panic.",
        "Risk is fused from calibrated local channels, then surfaced alongside policy alerts and classification mix.",
        [("Risk", "Current, mean, and session peak."), ("Alerts", "Threshold crossings with provenance."),
         ("Policy", "Alert-only defaults and bounded actions.")], hold_seconds,
    )
    page.locator("button[data-view='threats']").click()
    page.locator("[data-panel='threats']").first.scroll_into_view_if_needed()
    hold_view(page, hold_seconds)

    show_card(
        page, "03 / 06", "Reasoning view", "Evidence stays inspectable.",
        "Classifier confidence, HDC prototype similarity, anomaly scoring, and optional NIM context remain separate.",
        [("Local first", "Local detection remains authoritative."), ("NIM bounded", "Shadow context cannot enforce."),
         ("Traceable", "Every fused decision explains itself.")], hold_seconds,
    )
    page.locator("button[data-view='reasoning']").click()
    page.locator("[data-panel='reasoning']").first.scroll_into_view_if_needed()
    hold_view(page, hold_seconds)

    show_card(
        page, "04 / 06", "Operations", "Models and runtime in frame.",
        "The model registry and runtime telemetry expose what is active, what is being evaluated, and system health.",
        [("Registry", "Candidate, active, and previous states."), ("Telemetry", "Throughput, latency, queue, memory."),
         ("Control", "Pause the feed for incident review.")], hold_seconds,
    )
    page.locator("button[data-view='all']").click()
    page.locator(".runtime-panel").scroll_into_view_if_needed()
    hold_view(page, hold_seconds)

    show_card(
        page, "05 / 06", "PacketFlowAI", "Observe. Explain. Act conservatively.",
        "Signal Room turns the full detection pipeline into a clear, live, and reviewable operational surface.",
        [("Flow-centric", "Conversations over isolated packets."), ("Confidence-aware", "Calibrated evidence."),
         ("Operator-led", "Humans retain final authority.")], hold_seconds, dismiss=False,
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

    with tempfile.TemporaryDirectory(prefix="packetflowai-presentation-") as directory:
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

        raw_video_dir = root / "video"
        try:
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(headless=True)
                context = browser.new_context(
                    viewport={"width": 1600, "height": 900},
                    record_video_dir=raw_video_dir,
                    record_video_size={"width": 1600, "height": 900},
                    device_scale_factor=1,
                )
                page = context.new_page()
                video = page.video
                record_tour(page, f"http://127.0.0.1:{server.server.server_address[1]}", args.hold_seconds)
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
