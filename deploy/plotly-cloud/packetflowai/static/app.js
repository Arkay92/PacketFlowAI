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
  forensicCases: [],
  forensicIndex: 0,
  v3: null,
  worldNodeId: null,
  simulationAction: null,
  v4: null,
  v5: null,
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

function forensicSeverity(decision) {
  const level = safeNumber(decision.policy_level);
  const risk = safeNumber(decision.risk_score);
  return level >= 4 || risk >= 65 ? "red" : "orange";
}

function buildForensicCases() {
  const flows = new Map(state.flows.map((record) => [record.flow_id, record]));
  const alerts = new Map(state.alerts.map((record) => [record.event_id, record]));
  const nim = new Map(state.nim.map((record) => [record.event_id, record]));
  const evidenceByEvent = new Map();
  state.evidence.forEach((record) => {
    const entries = evidenceByEvent.get(record.event_id) || [];
    entries.push(record);
    evidenceByEvent.set(record.event_id, entries);
  });
  state.forensicCases = state.decisions.flatMap((record) => {
    const decision = record.payload || {};
    const eventId = decision.event_id || record.event_id;
    const label = String(decision.evidence?.classifier_label || "unknown").toLowerCase();
    const level = safeNumber(decision.policy_level);
    const risk = safeNumber(decision.risk_score);
    if (label === "benign" || (level < 2 && risk < 35)) return [];
    return [{
      eventId,
      decision,
      decisionRecord: record,
      flowRecord: flows.get(eventId) || null,
      alertRecord: alerts.get(eventId) || null,
      nimRecord: nim.get(eventId) || null,
      evidenceRecords: evidenceByEvent.get(eventId) || [],
      severity: forensicSeverity(decision),
    }];
  }).sort((left, right) => safeNumber(right.decision.risk_score) - safeNumber(left.decision.risk_score));
  state.forensicIndex = Math.min(state.forensicIndex, Math.max(0, state.forensicCases.length - 1));
}

function forensicValue(value, fallback = "--") {
  return value === null || value === undefined || value === "" ? fallback : value;
}

function renderForensicEvidence(caseFile) {
  const target = $("forensic-evidence-grid");
  const evidence = caseFile?.decision.evidence || {};
  const channels = [
    ["Local classifier", evidence.calibrated_confidence ?? evidence.classifier_confidence, "Calibrated confidence"],
    ["HDC prototype", evidence.prototype_similarity, "Cosine similarity"],
    ["Anomaly model", evidence.anomaly_score, "Deviation score"],
    ["NIM context", evidence.nim_reasoning_strength, caseFile?.nimRecord ? "Shadow assessment" : "No assessment"],
  ];
  target.replaceChildren(...channels.map(([name, value, note]) => {
    const numeric = value === null || value === undefined ? 0 : safeNumber(value);
    const card = create("article", "forensic-evidence-card");
    const meter = create("div", "evidence-meter");
    const fill = create("i");
    fill.style.setProperty("--score", `${Math.max(0, Math.min(100, numeric <= 1 ? numeric * 100 : numeric))}%`);
    meter.append(fill);
    card.append(create("span", "", name), create("strong", "", percent(value)), meter, create("small", "", note));
    return card;
  }));
}

function renderForensicMetadata(flow) {
  const target = $("forensic-metadata");
  const metadata = flow?.protocol_metadata || {};
  const entries = Object.entries(metadata);
  if (!entries.length) entries.push(["Protocol metadata", "Not captured"]);
  target.replaceChildren(...entries.map(([key, value]) => {
    const row = create("div");
    row.append(create("dt", "", String(key).replaceAll("_", " ")), create("dd", "", String(value)));
    return row;
  }));
}

function renderForensics() {
  buildForensicCases();
  const cases = state.forensicCases;
  const caseFile = cases[state.forensicIndex];
  text("forensic-case-count", cases.length);
  text("forensic-case-position", caseFile ? `CASE ${state.forensicIndex + 1} / ${cases.length}` : "No case selected");
  $("forensic-previous").disabled = cases.length < 2;
  $("forensic-next").disabled = cases.length < 2;

  const strip = $("forensic-case-strip");
  strip.replaceChildren(...cases.map((item, index) => {
    const button = create("button", `forensic-case-button ${item.severity}${index === state.forensicIndex ? " is-active" : ""}`);
    button.type = "button";
    button.dataset.index = String(index);
    const copy = create("span");
    copy.append(
      create("strong", "", String(item.decision.evidence?.classifier_label || "unknown").replaceAll("_", " ")),
      create("small", "", `${Math.round(safeNumber(item.decision.risk_score))} risk / ${item.eventId}`),
    );
    button.append(create("i"), copy);
    button.addEventListener("click", () => selectForensicCase(index));
    return button;
  }));

  if (!caseFile) {
    text("forensic-map-hint", "Awaiting an orange or red policy signal.");
    renderForensicEvidence(null);
    renderForensicMetadata(null);
    $("forensic-raw-json").textContent = JSON.stringify({ status: "awaiting_case" }, null, 2);
    return;
  }

  const flow = caseFile.flowRecord?.payload || {};
  const decision = caseFile.decision;
  const alert = caseFile.alertRecord?.payload || {};
  const evidence = decision.evidence || {};
  const dossier = document.querySelector(".forensic-dossier");
  document.querySelector(".forensics-workspace").classList.toggle("is-red", caseFile.severity === "red");
  dossier.classList.toggle("is-red", caseFile.severity === "red");
  text("forensic-severity", caseFile.severity === "red" ? "RED / HIGH CONFIDENCE" : "ORANGE / SUSPICIOUS");
  text("forensic-case-id", caseFile.eventId);
  text("forensic-label", String(evidence.classifier_label || "unknown").replaceAll("_", " "));
  text("forensic-explanation", decision.explanation || alert.reason || "Evidence threshold crossed.");
  text("forensic-source", forensicValue(flow.source_ip));
  text("forensic-source-port", `PORT ${forensicValue(flow.source_port)}`);
  text("forensic-destination", forensicValue(flow.destination_ip));
  text("forensic-destination-port", `PORT ${forensicValue(flow.destination_port)}`);
  text("forensic-protocol", forensicValue(flow.protocol));
  text("forensic-state", forensicValue(flow.state));
  text("forensic-risk", `${safeNumber(decision.risk_score).toFixed(1)} / 100`);
  text("forensic-policy", policyName(decision.policy_level));
  text("forensic-created", new Date(caseFile.decisionRecord.created_at).toLocaleString());
  text("forensic-duration", `${safeNumber(flow.duration_seconds).toFixed(3)} sec`);
  text("forensic-action", String(decision.action || alert.action || "monitor").toUpperCase());
  text("forensic-reason", decision.reason || alert.reason || "Local policy remains authoritative.");
  text("forensic-packets", number.format(flow.packet_count || 0));
  text("forensic-direction", `${flow.forward_packets || 0} FWD / ${flow.reverse_packets || 0} REV`);
  text("forensic-bytes", bytes(flow.byte_count));
  text("forensic-byte-rate", `${bytes(flow.bytes_per_second)} / S`);
  text("forensic-packet-rate", safeNumber(flow.packets_per_second).toFixed(1));
  text("forensic-retransmits", flow.retransmission_count || 0);
  text("forensic-tcp-flags", `SYN ${flow.syn_count || 0} / RST ${flow.rst_count || 0}`);
  text("forensic-burst", safeNumber(flow.burstiness).toFixed(2));
  text("forensic-hosts", flow.unique_destination_hosts || 0);
  text("forensic-ports", `${flow.unique_destination_ports || 0} DEST PORTS`);
  text("forensic-forward-bytes", bytes(flow.forward_bytes));
  text("forensic-reverse-bytes", bytes(flow.reverse_bytes));
  const totalDirectionalBytes = safeNumber(flow.forward_bytes) + safeNumber(flow.reverse_bytes) || 1;
  $("forensic-forward-bar").parentElement.style.setProperty(
    "--forward", `${safeNumber(flow.forward_bytes) / totalDirectionalBytes * 100}%`,
  );
  text("forensic-map-hint", `${caseFile.severity.toUpperCase()} SIGNAL / ${caseFile.eventId} / CLICK NODES TO NAVIGATE`);
  renderForensicEvidence(caseFile);
  renderForensicMetadata(flow);
  $("forensic-raw-json").textContent = JSON.stringify({
    flow,
    decision,
    alert: caseFile.alertRecord?.payload || null,
    evidence: caseFile.evidenceRecords.map((record) => ({ channel: record.channel, payload: record.payload })),
    nim: caseFile.nimRecord?.payload || null,
  }, null, 2);
}

function selectForensicCase(index) {
  if (!state.forensicCases.length) return;
  state.forensicIndex = (index + state.forensicCases.length) % state.forensicCases.length;
  renderForensics();
}

class ForensicField {
  constructor(canvas) {
    this.canvas = canvas;
    this.context = canvas.getContext("2d");
    this.points = [];
    this.phase = 0;
    this.resize = this.resize.bind(this);
    this.draw = this.draw.bind(this);
    new ResizeObserver(this.resize).observe(canvas);
    canvas.addEventListener("click", (event) => this.select(event));
    requestAnimationFrame(this.draw);
  }

  resize() {
    const ratio = window.devicePixelRatio || 1;
    this.width = this.canvas.clientWidth;
    this.height = this.canvas.clientHeight;
    if (!this.width || !this.height) return;
    this.canvas.width = this.width * ratio;
    this.canvas.height = this.height * ratio;
    this.context.setTransform(ratio, 0, 0, ratio, 0, 0);
  }

  hash(value) {
    let result = 2166136261;
    for (const char of String(value)) result = Math.imul(result ^ char.charCodeAt(0), 16777619);
    return (result >>> 0) / 4294967295;
  }

  select(event) {
    const bounds = this.canvas.getBoundingClientRect();
    const x = event.clientX - bounds.left;
    const y = event.clientY - bounds.top;
    const nearest = this.points.reduce((best, point) => {
      const distance = Math.hypot(point.x - x, point.y - y);
      return !best || distance < best.distance ? { ...point, distance } : best;
    }, null);
    if (nearest && nearest.distance < 24) selectForensicCase(nearest.index);
  }

  draw() {
    if (!this.width) this.resize();
    if (!this.width || !this.height) {
      requestAnimationFrame(this.draw);
      return;
    }
    const ctx = this.context;
    const cx = this.width / 2;
    const cy = this.height / 2;
    const radius = Math.min(this.width, this.height) * .39;
    ctx.clearRect(0, 0, this.width, this.height);
    ctx.strokeStyle = "rgba(255,191,105,.075)";
    ctx.lineWidth = 1;
    [0.25, 0.5, 0.75, 1].forEach((scale) => {
      ctx.beginPath(); ctx.arc(cx, cy, radius * scale, 0, Math.PI * 2); ctx.stroke();
    });
    for (let angle = 0; angle < Math.PI * 2; angle += Math.PI / 12) {
      ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx + Math.cos(angle) * radius, cy + Math.sin(angle) * radius); ctx.stroke();
    }
    const sweep = this.phase % (Math.PI * 2);
    const sweepGradient = ctx.createLinearGradient(cx, cy, cx + Math.cos(sweep) * radius, cy + Math.sin(sweep) * radius);
    sweepGradient.addColorStop(0, "rgba(255,191,105,0)");
    sweepGradient.addColorStop(1, "rgba(255,191,105,.38)");
    ctx.strokeStyle = sweepGradient;
    ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx + Math.cos(sweep) * radius, cy + Math.sin(sweep) * radius); ctx.stroke();

    this.points = state.forensicCases.map((caseFile, index) => {
      const angle = this.hash(caseFile.eventId) * Math.PI * 2;
      const risk = safeNumber(caseFile.decision.risk_score);
      const distance = radius * (.3 + Math.min(1, risk / 100) * .62);
      return { index, x: cx + Math.cos(angle) * distance, y: cy + Math.sin(angle) * distance, caseFile };
    });
    this.points.forEach((point) => {
      const selected = point.index === state.forensicIndex;
      const red = point.caseFile.severity === "red";
      const color = red ? "255,107,87" : "255,191,105";
      ctx.strokeStyle = `rgba(${color},${selected ? .5 : .15})`;
      ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(point.x, point.y); ctx.stroke();
      if (selected) {
        ctx.strokeStyle = `rgba(${color},${.35 + Math.sin(this.phase * 3) * .12})`;
        ctx.beginPath(); ctx.arc(point.x, point.y, 13, 0, Math.PI * 2); ctx.stroke();
      }
      ctx.fillStyle = `rgb(${color})`;
      ctx.shadowColor = `rgba(${color},.8)`;
      ctx.shadowBlur = selected ? 14 : 6;
      ctx.beginPath(); ctx.arc(point.x, point.y, selected ? 5 : 3, 0, Math.PI * 2); ctx.fill();
      ctx.shadowBlur = 0;
    });
    ctx.fillStyle = "#ffbf69";
    ctx.beginPath(); ctx.arc(cx, cy, 4, 0, Math.PI * 2); ctx.fill();
    this.phase += .009;
    requestAnimationFrame(this.draw);
  }
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

function renderTimeSnapshot(index) {
  const snapshots = state.v3?.time_machine?.snapshots || [];
  if (!snapshots.length) return;
  const snapshot = snapshots[Math.max(0, Math.min(index, snapshots.length - 1))];
  text("time-value", new Date(snapshot.as_of).toLocaleString());
  text("time-known-flows", snapshot.known.flows);
  text("time-known-decisions", snapshot.known.decisions);
  text("time-future", snapshot.not_yet_known.flows + snapshot.not_yet_known.decisions);
  text("time-policy", policyName(snapshot.known.policy));
}

function selectSimulation(action) {
  state.simulationAction = action;
  renderCommand();
}

function renderCommand() {
  const data = state.v3;
  if (!data) return;
  const model = data.world_model || { nodes: [], edges: [], counts: {} };
  const campaign = data.campaigns?.[0];
  const assessment = data.predictions?.[0];
  const simulation = data.simulation || { alternatives: [] };
  text("command-posture", data.disagreements?.length ? "ANALYST ATTENTION" : campaign ? "PREDICTIVE WATCH" : "NOMINAL WATCH");
  text("command-generated", `GRAPH SYNTHESIS ${ago(data.generated_at)}`);
  text("world-counts", `${model.counts?.nodes || 0} nodes / ${model.counts?.edges || 0} edges`);
  text("campaign-title", campaign?.title || "NO CAMPAIGN");
  text("campaign-summary", campaign?.summary || "Waiting for correlated activity.");
  const predictionList = $("prediction-list");
  predictionList.replaceChildren();
  (assessment?.predictions || []).forEach((prediction) => {
    const item = create("article", "prediction");
    const heading = create("div");
    heading.append(create("strong", "", prediction.label), create("span", "", percent(prediction.probability)));
    const meter = create("i"); meter.style.setProperty("--probability", `${prediction.probability * 100}%`);
    item.append(heading, meter, create("small", "", `${prediction.technique_id} / ${prediction.time_horizon}`));
    predictionList.append(item);
  });
  if (!assessment) predictionList.append(empty("No progression sequence", "More evidence is required.", true));
  text("forecast-uncertainty", assessment ? percent(assessment.uncertainty) : "--");
  text("simulation-target", simulation.target || "--");
  text("simulation-rationale", simulation.rationale || "No active simulation.");
  const options = $("simulation-options"); options.replaceChildren();
  const activeAction = state.simulationAction || simulation.recommended_action;
  (simulation.alternatives || []).forEach((alternative) => {
    const button = create("button", `simulation-card${alternative.action === activeAction ? " is-active" : ""}`);
    button.type = "button";
    button.append(create("span", "", alternative.action === simulation.recommended_action ? "RECOMMENDED" : "ALTERNATIVE"), create("strong", "", alternative.action.replaceAll("_", " ")));
    const facts = create("dl");
    [["Risk reduction", percent(alternative.risk_reduction)], ["Business impact", alternative.business_impact], ["Blast radius", alternative.blast_radius]].forEach(([label, value]) => {
      const row = create("div"); row.append(create("dt", "", label), create("dd", "", value)); facts.append(row);
    });
    button.append(facts, create("small", "", `AUTHORITY / ${alternative.authority_required.toUpperCase()}`));
    button.addEventListener("click", () => selectSimulation(alternative.action));
    options.append(button);
  });
  const selected = (simulation.alternatives || []).find((item) => item.action === activeAction) || simulation.alternatives?.[0];
  text("twin-paths", selected?.threat_paths_removed || 0); text("twin-action", selected?.action?.replaceAll("_", " ") || "OBSERVE");
  text("twin-disrupted", `${selected?.legitimate_flows_disrupted || 0} flows`); text("twin-dependency", selected?.critical_dependency_affected ? "AT RISK" : "CLEAR");
  const snapshots = data.time_machine?.snapshots || [];
  const slider = $("time-slider"); slider.max = String(Math.max(0, snapshots.length - 1));
  if (Number(slider.value) > snapshots.length - 1 || slider.dataset.initialized !== "true") slider.value = String(Math.max(0, snapshots.length - 1));
  slider.dataset.initialized = "true"; renderTimeSnapshot(Number(slider.value));
  const integrity = data.integrity || {};
  text("integrity-state", integrity.verified ? "CHAIN VERIFIED" : "CHAIN FAILURE"); text("integrity-events", `${integrity.events || 0} SEALED EVENTS`); text("merkle-root", integrity.merkle_root || "No Merkle root");
  const checks = $("integrity-checks"); checks.replaceChildren();
  ["evidence_chain", "decision_record", "model_artifact", "policy_version"].forEach((key) => {
    const row = create("div", "integrity-check"); row.append(create("span", "", key.replaceAll("_", " ")), create("b", "", integrity[key] || "PENDING")); checks.append(row);
  });
  const ladder = $("authority-ladder"); ladder.replaceChildren();
  (data.authority?.rules || []).forEach((rule) => {
    const row = create("div", "authority-step"); row.append(create("i", "", rule.level), create("strong", "", rule.action.replaceAll("_", " ")), create("span", "", rule.authority_scope), create("em", "", rule.autonomous ? "AUTONOMOUS" : rule.approver_role)); ladder.append(row);
  });
  const rail = $("capability-rail"); rail.replaceChildren();
  (data.capabilities || []).forEach((capability) => {
    const row = create("article", "capability"); const copy = create("div"); copy.append(create("strong", "", capability.name), create("small", "", typeof capability.detail === "string" ? capability.detail : "Multiple acceleration tiers")); row.append(create("i"), copy, create("span", "", capability.status)); rail.append(row);
  });
}

function renderPlatform() {
  const data = state.v4;
  if (!data) return;
  const causal = data.causal_v2 || {};
  text("causal-link-count", causal.links?.length || 0);
  text("causal-root", causal.root_cause ? `ROOT / ${causal.root_cause}` : "Root cause unresolved");
  text("earliest-intervention", causal.earliest_intervention ? new Date(causal.earliest_intervention).toLocaleTimeString() : "--");
  text("missed-opportunity", causal.missed_opportunity?.detected ? "AUTHORITY DELAY DETECTED" : "No missed opportunity detected");
  const minimum = data.intervention?.minimum_intervention;
  text("minimum-intervention", minimum?.action?.replaceAll("_", " ") || "OBSERVE");
  text("residual-risk", `Residual risk ${safeNumber(minimum?.residual_risk).toFixed(1)}`);
  const coverage = data.explainability?.expected_source_coverage || {};
  text("evidence-coverage", percent(coverage.score || 0));
  text("missing-context", coverage.missing?.length ? `MISSING / ${coverage.missing.join(" / ")}` : "All contract sources present");
  const domains = $("platform-domains"); domains.replaceChildren();
  (data.platform_domains || []).forEach((domain) => {
    const item = create("article", "platform-domain");
    const copy = create("div"); copy.append(create("strong", "", domain.domain), create("small", "", domain.detail));
    item.append(copy, create("span", "", domain.status)); domains.append(item);
  });
  text("adaptive-batch", data.runtime_v2?.adaptive_batch || 1);
  text("capture-paths", data.runtime_v2?.capture_backends?.length || 0);
  text("hindsight-leakage", data.time_machine_v2?.hindsight_leakage?.length || 0);
}

function renderAssurance() {
  const data = state.v5;
  if (!data) return;
  text("assurance-level", data.assurance_level || "A0");
  text("assurance-profile", `${data.observed_sources || 0} / ${data.expected_sources || 0} CONTRACT SOURCES`);
  text("assurance-threat-risk", data.threat_risk || "UNKNOWN");
  text("assurance-risk", data.assurance_risk || "UNKNOWN");
  text("omission-risk", data.unknown_omission_risk || "UNKNOWN");
  text("assurance-limitation", data.limitation || "Unknown omission risk cannot be eliminated.");

  const vector = $("assurance-vector"); vector.replaceChildren();
  [
    ["Integrity", data.integrity],
    ["Inclusion proofs", data.inclusion],
    ["Log continuity", data.sequence_continuity],
    ["Expected source coverage", `${data.observed_sources} / ${data.expected_sources}`],
    ["Sequence coverage", percent(data.sequence_coverage)],
    ["Sensor liveness", data.sensor_liveness],
    ["Producer attestation", `${data.producer_attestation?.verified || 0} / ${data.producer_attestation?.expected || 0}`],
    ["External anchoring", data.external_anchoring],
    ["Independent re-derivation", data.independent_rederivation],
    ["Unexplained gaps", data.unexplained_gaps],
  ].forEach(([label, value]) => {
    const row = create("div", `vector-row${["PARTIAL", "UNKNOWN"].includes(String(value)) || Number(value) > 0 ? " partial" : ""}`);
    row.append(create("span", "", label), create("strong", "", value)); vector.append(row);
  });

  const proof = $("proof-path"); proof.replaceChildren();
  (data.proof_path || []).forEach((label, index) => {
    const button = create("button", `proof-step${index === 0 ? " is-active" : ""}`, label); button.type = "button";
    button.addEventListener("click", () => {
      proof.querySelectorAll("button").forEach((item) => item.classList.toggle("is-active", item === button));
      text("proof-active", label.toUpperCase());
      text("proof-digest", `${data.epoch_manifests?.[0]?.merkle_root || "no-root"} / step ${index + 1}`);
    });
    proof.append(button);
  });

  const heatmap = $("assurance-heatmap"); heatmap.replaceChildren(create("span", "heat-cell label", "TIME"));
  (data.assurance_heatmap?.sources || []).forEach((source) => heatmap.append(create("span", "heat-cell label", source.toUpperCase())));
  (data.assurance_heatmap?.rows || []).forEach((row) => {
    heatmap.append(create("span", "heat-cell label", row.time));
    row.cells.forEach((cell) => {
      const node = create("button", `heat-cell${cell.status === "DARK" ? " dark" : ""}`, cell.status === "DARK" ? "GAP" : "LIVE");
      node.type = "button";
      if (cell.status === "DARK") node.addEventListener("click", () => {
        const period = (data.dark_periods || []).find((item) => item.source === cell.source);
        $("dark-period-detail").replaceChildren(
          create("span", "", `${cell.source.toUpperCase()} SENSOR GAP`),
          create("strong", "", period ? `${period.start} - ${period.end} / ${period.duration_seconds}s` : `${row.time} interval`),
          create("p", "", period ? `Reason: ${period.reason}. Assurance impact: ${period.impact}.` : "No signed heartbeat was observed."),
        );
      });
      heatmap.append(node);
    });
  });

  const chain = $("observation-chain"); chain.replaceChildren();
  (data.observation_world?.nodes || []).forEach((node) => chain.append(create("span", "", `${node.kind} / ${node.label}`)));
  const path = $("recording-path"); path.replaceChildren();
  (data.recording_path || []).forEach((stage, index) => {
    const item = create("article", `path-stage${stage.status === "LOSS" ? " loss" : ""}`);
    item.append(create("i", "", index + 1), create("span", "", stage.stage), create("strong", "", number.format(stage.count)));
    path.append(item);
  });

  const claims = $("formal-claims"); claims.replaceChildren();
  (data.formal_claims || []).forEach((claim) => {
    const row = create("article", `claim-row${["PARTIAL", "UNKNOWN"].includes(claim.status) ? " partial" : ""}`);
    row.append(create("span", "", claim.id), create("p", "", claim.statement), create("strong", "", claim.status)); claims.append(row);
  });
  [["what-we-know", data.what_we_know, "+"], ["what-we-cannot-prove", data.what_we_cannot_prove, "-"]].forEach(([id, values, symbol]) => {
    const list = $(id); list.replaceChildren();
    (values || []).forEach((value) => { const row = create("article", "boundary-item"); row.append(create("i", "", symbol), create("p", "", value)); list.append(row); });
  });

  const rederivation = $("rederivation-status"); rederivation.replaceChildren();
  (data.rederivation || []).forEach((item) => {
    const row = create("article", `rederive-row${item.classification.includes("NOT_REPRODUCIBLE") ? " recorded" : ""}`);
    row.append(create("span", "", item.component), create("p", "", item.classification.replaceAll("_", " ")), create("strong", "", item.status)); rederivation.append(row);
  });
  const authority = $("assurance-authority"); authority.replaceChildren();
  (data.authority || []).forEach((item) => { const row = create("article", "authority-assurance-row"); row.append(create("strong", "", item.action.replaceAll("_", " ")), create("span", "", item.decision.replaceAll("_", " "))); authority.append(row); });
  const debts = data.assurance_debt || [];
  $("assurance-debt").replaceChildren(create("strong", "", `ASSURANCE DEBT / ${debts.length}`), create("span", "", debts.length ? debts.map((item) => item.source).join(" / ") : "No open evidence debt"));

  const attacks = $("assurance-attack-lab"); attacks.replaceChildren();
  (data.attack_lab || []).forEach((item) => { const row = create("article", "attack-row"); row.append(create("strong", "", item.attack.replaceAll("_", " ")), create("span", "", item.result), create("small", "", item.detected_by)); attacks.append(row); });
  const contract = data.contract || {}; const contractBox = $("evidence-contract"); contractBox.replaceChildren();
  const identity = create("div", "contract-identity"); identity.append(create("strong", "", contract.contract_id || "NO CONTRACT"), create("span", "", `SIGNED / ${contract.version || "--"}`));
  const sources = create("div", "contract-sources");
  (contract.expected_sources || []).forEach((source) => sources.append(create("span", (data.missing_expected_sources || []).includes(source) ? "missing" : "", source.toUpperCase())));
  contractBox.append(identity, sources, create("div", "contract-meta", `${contract.environment || "--"}<br>${contract.valid_from || "--"} TO ${contract.valid_until || "--"}<br>HASH / ${contract.contract_hash || "--"}`));
  const witness = data.witness_reconciliation || {}; $("witness-status").replaceChildren(create("span", "", `WITNESSES / ${(witness.witnesses || []).length}`), create("span", "", witness.status || "UNKNOWN"), create("span", "", `SERVICES / ${(witness.services || []).length}`));
}

class ObservationField {
  constructor(canvas) { this.canvas = canvas; this.context = canvas.getContext("2d"); this.resize = this.resize.bind(this); this.draw = this.draw.bind(this); new ResizeObserver(this.resize).observe(canvas); requestAnimationFrame(this.draw); }
  resize() { const ratio = window.devicePixelRatio || 1; this.width = this.canvas.clientWidth; this.height = this.canvas.clientHeight; if (!this.width || !this.height) return; this.canvas.width = this.width * ratio; this.canvas.height = this.height * ratio; this.context.setTransform(ratio, 0, 0, ratio, 0, 0); }
  draw() { if (!this.width) this.resize(); if (!this.width || !this.height) { requestAnimationFrame(this.draw); return; } const ctx = this.context; const nodes = state.v5?.observation_world?.nodes || []; const edges = state.v5?.observation_world?.edges || []; const points = nodes.map((node, index) => ({ node, x: 40 + index * ((this.width - 80) / Math.max(1, nodes.length - 1)), y: this.height / 2 + Math.sin(index * 1.7) * 55 })); const byId = new Map(points.map((point) => [point.node.id, point])); ctx.clearRect(0, 0, this.width, this.height); edges.forEach((edge) => { const left = byId.get(edge.source); const right = byId.get(edge.target); if (!left || !right) return; ctx.strokeStyle = "rgba(214,255,103,.35)"; ctx.beginPath(); ctx.moveTo(left.x, left.y); ctx.lineTo(right.x, right.y); ctx.stroke(); ctx.fillStyle = "rgba(214,255,103,.7)"; ctx.font = "7px monospace"; ctx.fillText(edge.relationship, (left.x + right.x) / 2 - 24, (left.y + right.y) / 2 - 8); }); points.forEach((point) => { ctx.fillStyle = point.node.kind === "HOST" ? "#ffbf69" : "#d6ff67"; ctx.shadowColor = ctx.fillStyle; ctx.shadowBlur = 12; ctx.beginPath(); ctx.arc(point.x, point.y, 5, 0, Math.PI * 2); ctx.fill(); ctx.shadowBlur = 0; ctx.fillStyle = "#aebbb3"; ctx.font = "8px monospace"; ctx.fillText(point.node.label, point.x - 25, point.y + 22); }); requestAnimationFrame(this.draw); }
}

class WorldField {
  constructor(canvas) {
    this.canvas = canvas; this.context = canvas.getContext("2d"); this.points = []; this.phase = 0;
    this.resize = this.resize.bind(this); this.draw = this.draw.bind(this);
    new ResizeObserver(this.resize).observe(canvas); canvas.addEventListener("click", (event) => this.select(event)); requestAnimationFrame(this.draw);
  }
  resize() { const ratio = window.devicePixelRatio || 1; this.width = this.canvas.clientWidth; this.height = this.canvas.clientHeight; if (!this.width || !this.height) return; this.canvas.width = this.width * ratio; this.canvas.height = this.height * ratio; this.context.setTransform(ratio, 0, 0, ratio, 0, 0); }
  hash(value) { let result = 2166136261; for (const char of String(value)) result = Math.imul(result ^ char.charCodeAt(0), 16777619); return (result >>> 0) / 4294967295; }
  select(event) { const bounds = this.canvas.getBoundingClientRect(); const x = event.clientX - bounds.left; const y = event.clientY - bounds.top; const nearest = this.points.reduce((best, point) => { const distance = Math.hypot(point.x - x, point.y - y); return !best || distance < best.distance ? { ...point, distance } : best; }, null); if (nearest?.distance < 28) { state.worldNodeId = nearest.node.node_id; const selection = $("world-selection"); selection.replaceChildren(create("span", "", "ACTIVE ENTITY"), create("strong", "", `${nearest.node.kind} / ${nearest.node.label}`), create("small", "", `${(state.v3.world_model.edges || []).filter((edge) => edge.source === nearest.node.node_id || edge.target === nearest.node.node_id).length} evidence relationships linked to this entity.`)); } }
  draw() { if (!this.width) this.resize(); if (!this.width || !this.height) { requestAnimationFrame(this.draw); return; } const ctx = this.context; const nodes = (state.v3?.world_model?.nodes || []).slice(0, 70); const edges = state.v3?.world_model?.edges || []; const cx = this.width * .46; const cy = this.height * .5; const radius = Math.min(this.width, this.height) * .42; ctx.clearRect(0, 0, this.width, this.height); this.points = nodes.map((node, index) => { const angle = this.hash(node.node_id) * Math.PI * 2; const ring = node.kind === "FLOW" ? .25 : node.kind === "TECHNIQUE" ? .9 : .48 + this.hash(index) * .35; return { node, x: cx + Math.cos(angle) * radius * ring, y: cy + Math.sin(angle) * radius * ring }; }); const byId = new Map(this.points.map((point) => [point.node.node_id, point])); edges.slice(0, 140).forEach((edge) => { const left = byId.get(edge.source); const right = byId.get(edge.target); if (!left || !right) return; const selected = state.worldNodeId && (edge.source === state.worldNodeId || edge.target === state.worldNodeId); ctx.strokeStyle = selected ? "rgba(255,191,105,.7)" : "rgba(113,201,255,.12)"; ctx.beginPath(); ctx.moveTo(left.x, left.y); ctx.lineTo(right.x, right.y); ctx.stroke(); }); this.points.forEach((point) => { const selected = point.node.node_id === state.worldNodeId; const palette = { SOURCE: "255,107,87", HOST: "255,191,105", ACCOUNT: "211,157,255", TECHNIQUE: "113,201,255", FLOW: "157,246,174" }; const color = palette[point.node.kind] || "205,226,211"; ctx.fillStyle = `rgb(${color})`; ctx.shadowColor = `rgba(${color},.8)`; ctx.shadowBlur = selected ? 18 : 5; ctx.beginPath(); ctx.arc(point.x, point.y, selected ? 6 : point.node.kind === "FLOW" ? 2 : 3.5, 0, Math.PI * 2); ctx.fill(); ctx.shadowBlur = 0; if (selected || point.node.kind === "TECHNIQUE") { ctx.fillStyle = `rgba(${color},.8)`; ctx.font = "8px monospace"; ctx.fillText(point.node.label.slice(0, 22), point.x + 9, point.y + 3); } }); this.phase += .01; requestAnimationFrame(this.draw); }
}

async function refresh() {
  if (state.paused) return;
  try {
    const [health, overview, flows, alerts, decisions, evidence, nim, models, metrics, status, v4, v5] = await Promise.all([
      getJSON("/health"), getJSON("/overview"), getJSON("/flows?limit=60"), getJSON("/alerts?limit=20"),
      getJSON("/decisions?limit=60"), getJSON("/evidence?limit=30"), getJSON("/nim?limit=20"),
      getJSON("/models"), getJSON("/metrics"), getJSON("/status"), getJSON("/v4/overview"), getJSON("/v5/overview"),
    ]);
    Object.assign(state, { overview, flows, alerts, decisions, evidence, nim, models, metrics, status, v3: v4, v4, v5 });
    state.lastRefresh = new Date();
    $("live-dot").className = "live-dot is-live";
    text("system-state", health.status === "ok" ? "System live" : health.status);
    text("last-sync", `Synced ${state.lastRefresh.toLocaleTimeString()}`);
    renderSummary(); renderFlows(); renderClasses(); renderIncidents(); renderEvidence(); renderModels(); renderRuntime();
    renderForensics();
    renderCommand();
    renderPlatform();
    renderAssurance();
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
    if (button.dataset.view === "forensics") renderForensics();
    if (button.dataset.view === "command") { renderCommand(); renderPlatform(); }
    if (button.dataset.view === "assurance") renderAssurance();
  });
});

$("forensic-previous").addEventListener("click", () => selectForensicCase(state.forensicIndex - 1));
$("forensic-next").addEventListener("click", () => selectForensicCase(state.forensicIndex + 1));
$("time-slider").addEventListener("input", (event) => renderTimeSnapshot(Number(event.target.value)));
document.addEventListener("keydown", (event) => {
  if ($("dashboard").dataset.view !== "forensics" || event.target.matches("input, textarea, pre")) return;
  if (event.key === "ArrowLeft") selectForensicCase(state.forensicIndex - 1);
  if (event.key === "ArrowRight") selectForensicCase(state.forensicIndex + 1);
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
new ForensicField($("forensic-canvas"));
const worldField = new WorldField($("world-canvas"));
new ObservationField($("assurance-world-canvas"));
refresh();
setInterval(refresh, 4000);
