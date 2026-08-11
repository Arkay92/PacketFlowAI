"use strict";

const state = {
  paused: false,
  flows: [],
  alerts: [],
  decisions: [],
  evidence: [],
  nim: [],
  models: [],
  metrics: {},
  overview: null,
  status: {},
  riskHistory: Array(30).fill(0),
  lastRefresh: null,
};

const $ = (id) => document.getElementById(id);
const policyNames = ["NORMAL", "OBSERVE", "SUSPICIOUS", "LIKELY MALICIOUS", "HIGH CONFIDENCE", "CONTAIN"];
const number = new Intl.NumberFormat("en", { notation: "compact", maximumFractionDigits: 1 });

function text(id, value) {
  const target = $(id);
  if (target) target.textContent = String(value);
}

function safeNumber(value, fallback = 0) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function percent(value) {
  if (value === null || value === undefined) return "--";
  const numeric = safeNumber(value);
  return `${(numeric <= 1 ? numeric * 100 : numeric).toFixed(1)}%`;
}

function bytes(value) {
  const numeric = safeNumber(value);
  if (numeric < 1024) return `${numeric} B`;
  if (numeric < 1024 ** 2) return `${(numeric / 1024).toFixed(1)} KB`;
  return `${(numeric / 1024 ** 2).toFixed(1)} MB`;
}

function ago(value) {
  if (!value) return "now";
  const seconds = Math.max(0, (Date.now() - new Date(value).getTime()) / 1000);
  if (seconds < 60) return `${Math.round(seconds)}s ago`;
  if (seconds < 3600) return `${Math.round(seconds / 60)}m ago`;
  return `${Math.round(seconds / 3600)}h ago`;
}

async function getJSON(path) {
  const response = await fetch(path, { cache: "no-store" });
  if (!response.ok) throw new Error(`${path}: ${response.status}`);
  return response.json();
}

function create(tag, className, content) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (content !== undefined) node.textContent = content;
  return node;
}

function empty(title, detail, compact = false) {
  const container = create("div", `empty-state${compact ? " compact" : ""}`);
  container.append(create("strong", "", title), create("small", "", detail));
  return container;
}

function policyName(value) {
  if (typeof value === "number") return policyNames[value] || "UNKNOWN";
  const parsed = Number(value);
  if (Number.isInteger(parsed)) return policyNames[parsed] || "UNKNOWN";
  return String(value || "NORMAL").replaceAll("_", " ");
}

function renderSummary() {
  const overview = state.overview || { counts: {}, risk: {}, classifications: {} };
  const counts = overview.counts || {};
  text("metric-flows", number.format(counts.flows || 0));
  text("metric-alerts", number.format(counts.alerts || 0));
  text("metric-evidence", number.format(counts.evidence || 0));
  text("metric-nim", number.format(counts.nim_assessments || 0));
  text("metric-alert-note", counts.alerts ? "Review required" : "No active alerts");
  text("nim-mode-summary", `Mode ${state.status.nim_mode || "disabled"}`);
  text("nim-mode", `NIM ${(state.status.nim_mode || "disabled").toUpperCase()}`);
  text("footer-mode", state.status.containment_enabled ? "Containment enabled" : "Containment disabled");
  text("metric-flow-rate", `${safeNumber(state.metrics.flows_per_second).toFixed(1)} / sec`);

  const risk = overview.risk || {};
  const current = Math.max(0, Math.min(100, safeNumber(risk.current)));
  text("risk-current", Math.round(current));
  text("risk-peak", safeNumber(risk.peak).toFixed(1));
  text("risk-average", safeNumber(risk.average).toFixed(1));
  const instrument = $("risk-instrument");
  instrument.style.setProperty("--risk", current.toFixed(1));
  instrument.classList.toggle("is-warn", current >= 35 && current < 65);
  instrument.classList.toggle("is-danger", current >= 65);
  const latest = overview.latest_decision?.payload;
  text("policy-level", policyName(latest?.policy_level));
  state.riskHistory.push(current);
  state.riskHistory = state.riskHistory.slice(-30);
  drawSparkline();
}

function renderFlows() {
  const ledger = $("flow-ledger");
  text("flow-count-label", `Latest ${state.flows.length}`);
  if (!state.flows.length) {
    ledger.replaceChildren(empty("Listening for flows", "Capture or replay traffic to populate the ledger."));
    text("field-status", "Waiting for flow telemetry");
    return;
  }
  const fragment = document.createDocumentFragment();
  state.flows.slice(0, 12).forEach((record, index) => {
    const flow = record.payload || {};
    const row = create("article", "flow-row");
    row.style.animationDelay = `${index * 25}ms`;
    const route = create("div", "flow-route");
    route.append(
      create("strong", "", `${flow.source_ip || "?"}:${flow.source_port ?? "*"} → ${flow.destination_ip || "?"}:${flow.destination_port ?? "*"}`),
      create("small", "", `${flow.flow_id || record.flow_id || "unknown"} · ${safeNumber(flow.duration_seconds).toFixed(3)} sec`),
    );
    const protocol = create("span", "protocol-pill", flow.protocol || "--");
    const volume = create("span", "flow-volume", `${bytes(flow.byte_count)} / ${flow.packet_count || 0} pkt`);
    const status = create("span", `state-dot ${(flow.state || "").toLowerCase()}`, flow.state || "ACTIVE");
    row.append(route, protocol, volume, status);
    fragment.append(row);
  });
  ledger.replaceChildren(fragment);
  const latest = state.flows[0]?.payload || {};
  text("field-protocol", `${latest.protocol || "FLOW"} / ${latest.state || "ACTIVE"}`);
  text("field-status", `${state.flows.length} recent flows mapped`);
}

function renderClasses() {
  const target = $("classification-bars");
  const classes = state.overview?.classifications || {};
  const entries = Object.entries(classes).sort((a, b) => b[1] - a[1]);
  if (!entries.length) {
    target.replaceChildren(empty("No classifications", "Decisions will appear after flow inference.", true));
    return;
  }
  const total = entries.reduce((sum, [, count]) => sum + safeNumber(count), 0) || 1;
  target.replaceChildren(...entries.map(([label, count]) => {
    const row = create("div", `class-row ${label === "benign" ? "" : label === "unknown" ? "unknown" : "threat"}`);
    const track = create("div", "class-track");
    const fill = create("div", "class-fill");
    fill.style.setProperty("--width", `${(safeNumber(count) / total) * 100}%`);
    track.append(fill);
    row.append(create("span", "", label), track, create("span", "class-value", count));
    return row;
  }));
}

function renderIncidents() {
  const target = $("incident-list");
  if (!state.alerts.length) {
    target.replaceChildren(empty("No policy alerts", "Alert-only mode is standing by.", true));
    return;
  }
  target.replaceChildren(...state.alerts.slice(0, 10).map((record) => {
    const alert = record.payload || {};
    const level = safeNumber(alert.policy_level);
    const row = create("article", `incident ${level >= 4 ? "high" : ""}`);
    const copy = create("div", "incident-copy");
    copy.append(
      create("strong", "", `${policyName(level)} · ${alert.action || "alert"}`),
      create("p", "", alert.reason || "Policy threshold crossed."),
      create("small", "", `${alert.target || "unknown target"} · ${alert.event_id || record.event_id}`),
    );
    row.append(create("span", "incident-bar"), copy, create("span", "incident-time", ago(record.created_at)));
    return row;
  }));
}

function renderEvidence() {
  const decision = state.decisions[0]?.payload;
  const evidence = decision?.evidence || {};
  text("local-confidence", percent(evidence.calibrated_confidence ?? evidence.classifier_confidence));
  text("prototype-similarity", percent(evidence.prototype_similarity));
  text("anomaly-score", evidence.anomaly_score === null || evidence.anomaly_score === undefined ? "--" : safeNumber(evidence.anomaly_score).toFixed(2));
  text("nim-strength", percent(evidence.nim_reasoning_strength));
  const trace = $("reasoning-trace");
  const body = trace.querySelector("p");
  body.textContent = decision?.explanation || "No fused decision has been recorded.";
}

function renderModels() {
  const target = $("model-stack");
  text("model-count", `${state.models.length} artifact${state.models.length === 1 ? "" : "s"}`);
  if (!state.models.length) {
    target.replaceChildren(empty("No registered model", "Train, evaluate, shadow, then promote.", true));
    return;
  }
  target.replaceChildren(...state.models.map((model) => {
    const card = create("article", "model-card");
    const copy = create("div");
    copy.append(
      create("strong", "", `${model.model_id} / ${model.version}`),
      create("small", "", `Evaluated ${model.evaluated ? "yes" : "no"} · Shadow ${model.shadow_validated ? "passed" : "pending"}`),
    );
    card.append(copy, create("span", `model-state ${model.state}`, model.state));
    return card;
  }));
}

function renderRuntime() {
  const metrics = state.metrics || {};
  const values = [
    ["runtime-pps", safeNumber(metrics.packets_per_second).toFixed(1), Math.min(100, safeNumber(metrics.packets_per_second) / 100)],
    ["runtime-fps", safeNumber(metrics.flows_per_second).toFixed(1), Math.min(100, safeNumber(metrics.flows_per_second))],
    ["runtime-p95", `${safeNumber(metrics.inference_latency_p95_ms).toFixed(1)} ms`, Math.min(100, safeNumber(metrics.inference_latency_p95_ms))],
    ["runtime-queue", number.format(metrics.queue_depth || 0), Math.min(100, safeNumber(metrics.queue_depth) / 10)],
    ["runtime-dropped", number.format(metrics.dropped_packets || 0), Math.min(100, safeNumber(metrics.dropped_packets))],
    ["runtime-memory", `${(safeNumber(metrics.traced_memory_bytes) / 1024 ** 2).toFixed(1)} MB`, Math.min(100, safeNumber(metrics.traced_memory_bytes) / 1024 ** 2)],
  ];
  values.forEach(([id, value, fill]) => {
    text(id, value);
    $(id).nextElementSibling.style.setProperty("--fill", `${fill}%`);
  });
}

function drawSparkline() {
  const canvas = $("risk-sparkline");
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth || 300;
  const height = canvas.clientHeight || 72;
  canvas.width = width * ratio;
  canvas.height = height * ratio;
  const context = canvas.getContext("2d");
  context.scale(ratio, ratio);
  context.clearRect(0, 0, width, height);
  context.strokeStyle = "rgba(220,231,223,.08)";
  context.beginPath(); context.moveTo(0, height - 1); context.lineTo(width, height - 1); context.stroke();
  context.strokeStyle = "#9df6ae";
  context.lineWidth = 1.5;
  context.beginPath();
  state.riskHistory.forEach((value, index) => {
    const x = index / (state.riskHistory.length - 1) * width;
    const y = height - 4 - value / 100 * (height - 8);
    if (index === 0) context.moveTo(x, y); else context.lineTo(x, y);
  });
  context.stroke();
}

class NetworkField {
  constructor(canvas) {
    this.canvas = canvas;
    this.context = canvas.getContext("2d");
    this.phase = 0;
    this.resize = this.resize.bind(this);
    this.draw = this.draw.bind(this);
    new ResizeObserver(this.resize).observe(canvas);
    requestAnimationFrame(this.draw);
  }

  resize() {
    const ratio = window.devicePixelRatio || 1;
    this.width = this.canvas.clientWidth;
    this.height = this.canvas.clientHeight;
    this.canvas.width = this.width * ratio;
    this.canvas.height = this.height * ratio;
    this.context.setTransform(ratio, 0, 0, ratio, 0, 0);
  }

  hash(value) {
    let result = 2166136261;
    for (const char of String(value)) result = Math.imul(result ^ char.charCodeAt(0), 16777619);
    return (result >>> 0) / 4294967295;
  }

  draw() {
    if (!this.width) this.resize();
    const ctx = this.context;
    const cx = this.width / 2;
    const cy = this.height / 2;
    const radius = Math.min(this.width, this.height) * 0.39;
    ctx.clearRect(0, 0, this.width, this.height);
    ctx.lineWidth = 1;
    [0.28, 0.55, 0.82, 1].forEach((scale, index) => {
      ctx.strokeStyle = `rgba(157,246,174,${0.10 - index * 0.012})`;
      ctx.beginPath(); ctx.arc(cx, cy, radius * scale, 0, Math.PI * 2); ctx.stroke();
    });
    ctx.strokeStyle = "rgba(220,231,223,.045)";
    for (let angle = 0; angle < Math.PI * 2; angle += Math.PI / 8) {
      ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx + Math.cos(angle) * radius, cy + Math.sin(angle) * radius); ctx.stroke();
    }
    const sweep = this.phase % (Math.PI * 2);
    const gradient = ctx.createLinearGradient(cx, cy, cx + Math.cos(sweep) * radius, cy + Math.sin(sweep) * radius);
    gradient.addColorStop(0, "rgba(157,246,174,0)"); gradient.addColorStop(1, "rgba(157,246,174,.28)");
    ctx.strokeStyle = gradient; ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx + Math.cos(sweep) * radius, cy + Math.sin(sweep) * radius); ctx.stroke();

    state.flows.slice(0, 55).forEach((record, index) => {
      const flow = record.payload || {};
      const angle = this.hash(flow.destination_ip || index) * Math.PI * 2;
      const distance = radius * (0.25 + this.hash(flow.flow_id || index) * 0.72);
      const x = cx + Math.cos(angle) * distance;
      const y = cy + Math.sin(angle) * distance;
      const malicious = state.decisions[index]?.payload?.evidence?.classifier_label !== "benign" && state.decisions[index];
      const color = malicious ? "255,107,87" : "157,246,174";
      ctx.strokeStyle = `rgba(${color},.16)`;
      ctx.beginPath(); ctx.moveTo(cx, cy); ctx.quadraticCurveTo(cx + Math.sin(angle) * 60, cy - Math.cos(angle) * 60, x, y); ctx.stroke();
      ctx.fillStyle = `rgba(${color},${0.45 + (index % 3) * .18})`;
      ctx.beginPath(); ctx.arc(x, y, malicious ? 3.2 : 2, 0, Math.PI * 2); ctx.fill();
      const progress = (this.phase * 0.23 + index * .17) % 1;
      const px = cx + (x - cx) * progress;
      const py = cy + (y - cy) * progress;
      ctx.fillStyle = `rgba(${color},.9)`; ctx.fillRect(px - 1, py - 1, 2, 2);
    });
    ctx.fillStyle = "#9df6ae"; ctx.beginPath(); ctx.arc(cx, cy, 4, 0, Math.PI * 2); ctx.fill();
    this.phase += 0.008;
    requestAnimationFrame(this.draw);
  }
}

async function refresh() {
  if (state.paused) return;
  try {
    const [health, overview, flows, alerts, decisions, evidence, nim, models, metrics, status] = await Promise.all([
      getJSON("/health"), getJSON("/overview"), getJSON("/flows?limit=60"), getJSON("/alerts?limit=20"),
      getJSON("/decisions?limit=60"), getJSON("/evidence?limit=30"), getJSON("/nim?limit=20"),
      getJSON("/models"), getJSON("/metrics"), getJSON("/status"),
    ]);
    Object.assign(state, { overview, flows, alerts, decisions, evidence, nim, models, metrics, status });
    state.lastRefresh = new Date();
    $("live-dot").className = "live-dot is-live";
    text("system-state", health.status === "ok" ? "System live" : health.status);
    text("last-sync", `Synced ${state.lastRefresh.toLocaleTimeString()}`);
    renderSummary(); renderFlows(); renderClasses(); renderIncidents(); renderEvidence(); renderModels(); renderRuntime();
  } catch (error) {
    $("live-dot").className = "live-dot is-error";
    text("system-state", "API unavailable");
    text("last-sync", error.message);
  }
}

document.querySelectorAll(".nav-chip").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".nav-chip").forEach((item) => item.classList.toggle("is-active", item === button));
    $("dashboard").dataset.view = button.dataset.view;
  });
});

$("pause-button").addEventListener("click", () => {
  state.paused = !state.paused;
  $("pause-button").setAttribute("aria-pressed", String(state.paused));
  text("pause-button", state.paused ? "Resume feed" : "Pause feed");
  text("system-state", state.paused ? "Feed paused" : "System live");
  if (!state.paused) refresh();
});

setInterval(() => text("field-clock", new Date().toLocaleTimeString("en-GB")), 1000);
new NetworkField($("network-canvas"));
refresh();
setInterval(refresh, 4000);
