# PacketFlowAI v2 Implementation Plan

## Purpose

This document turns the proposed change list into an execution plan grounded in the current repository state as of August 10, 2026.

Current baseline:

- The project is primarily a single-file prototype in [main.py](C:\Users\pc006\Documents\GitHub\PacketFlowAI\main.py).
- Training, inference, packet capture, response behavior, state management, and dataset parsing are tightly coupled.
- Runtime artifacts such as `packet_hv_model.pth`, `exceptions.log`, and `packet_feedback.txt` live in the repository root.
- The current system performs packet-level classification rather than flow-level reasoning.

The goal of this plan is to move PacketFlowAI from a prototype HDC packet classifier into a modular, flow-centric, confidence-aware detection and reasoning system with safe response controls and optional NVIDIA NIM escalation.

## Planning Principles

- Fix correctness and safety issues before expanding capability.
- Create stable typed schemas before refactoring model logic.
- Separate local deterministic detection from cloud reasoning.
- Keep enforcement conservative until provenance, rollback, and allowlists exist.
- Avoid deep feature expansion until the ingestion and evaluation pipelines are canonical.
- Structure the migration as a series of small, testable PRs rather than a single rewrite.
- Treat dataset-native ground truth as authoritative where it exists; weak or heuristic labels must be explicitly marked and isolated.
- Treat NVIDIA NIM as a bounded reasoning assistant, never as the direct authority for enforcement or automatic training labels.
- Keep local detection functional when all cloud reasoning is disabled or unavailable.

## Current-State Constraints

The plan assumes the current codebase has the following characteristics:

- Global mutable state for bans, malicious counts, encoder instances, and shutdown handling
- Mixed training and inference feature extraction logic
- Random HDC encoder state that is not fully checkpointed
- Regex-based training feature extraction
- Packet-level Scapy capture with limited TCP/UDP support
- Basic queueing with limited lifecycle separation
- A mismatch between declared attack taxonomy and actual model/classification behavior

These constraints directly shape the implementation order.

## Target v2 Architecture

The intended end-state is:

```text
Network / PCAP
  ->
Capture Backend
  ->
Flow Engine
  ->
Canonical Feature Extraction
  ->
HDC Representation + Behavioural Features + Protocol Metadata
  ->
Local Detection Stack
  - HDC Prototype Classifier
  - Neural Head
  - OOD / Unknown Detector
  - Anomaly Model
  - Confidence Calibration
  ->
Uncertainty Gate
  ->
Optional NVIDIA NIM Reasoning
  - NIM Evidence Sanitizer
  - Read-only Reasoning Tools
  - Shadow / Influence Modes
  ->
Evidence Fusion
  ->
Threat Taxonomy + Deterministic MITRE Mapping
  ->
Threat Score + Policy Level
  ->
Deterministic Response Engine
  - Alert / Webhook / SIEM
  - Mirror / Rate Limit
  - Temporary Block / Quarantine
  ->
Evidence Store + Analyst Feedback + Active Learning + Drift Detection + Telemetry
  ->
Model Registry + Promotion / Rollback
```

## Milestone Overview

The work should be executed in four release milestones.

### v2.0 Correctness and Architecture

Focus:

- Eliminate training-serving skew
- Fix unsafe response logic
- Make HDC deterministic
- Split the monolith into maintainable modules
- Establish full model/checkpoint provenance and safe artifact handling

Included phases:

- Phase 0: Baseline Audit and Freeze
- Phase 1: Correctness and Safety Hardening
- Phase 2: Repository Restructure

### v2.1 Flow Intelligence

Focus:

- Move from packet classification to flow classification
- Add temporal, host, and protocol-aware features
- Build a stronger local detector and real benchmark pipeline
- Add deterministic ATT&CK mapping and clustering of UNKNOWN/OOD traffic into emerging behavior families

Included phases:

- Phase 3: Flow-Based Detection Core
- Phase 4: Local Detection Stack
- Phase 5: Dataset and Evaluation Framework

### v2.2 NVIDIA NIM and Fusion

Focus:

- Escalate only uncertain or abnormal events
- Keep cloud reasoning optional and bounded
- Preserve inspectable evidence rather than opaque “AI says so” outputs
- Add NIM shadow mode, cloud-data sanitization, read-only reasoning tools, and effectiveness measurement

Included phases:

- Phase 6: NVIDIA NIM Reasoning Layer
- Phase 7: Fusion and Policy Engine

### v2.3 Operational Productization

Focus:

- Make the system reliable under load
- Add persistence, APIs, feedback workflows, dashboarding, and CI
- Add model registry, candidate/active/previous promotion, rollback, active learning, and SIEM-oriented response adapters

Included phases:

- Phase 8: Runtime and Performance Engineering
- Phase 9: Storage, API, Telemetry, and Feedback
- Phase 10: Tooling, Tests, CI, Docs, and Dashboard

## Phase Plan

## Phase 0: Baseline Audit and Freeze

### Objectives

- Establish a clear map of current behavior before changing architecture
- Preserve a small set of regression fixtures from the current implementation
- Document where current behavior is intentionally kept versus replaced

### Tasks

- Audit [main.py](C:\Users\pc006\Documents\GitHub\PacketFlowAI\main.py) responsibilities and identify refactor boundaries
- Record current checkpoint contents and loading behavior
- Capture sample dataset parsing behavior from the current regex extractor
- Capture sample live packet preprocessing behavior from the current Scapy path
- Document the current class mapping and decision flow
- Create golden fixtures for:
  - Dataset row -> extracted features
  - Packet -> extracted features
  - Features -> encoded hypervector shape
  - Prediction -> reported attack label

### Deliverables

- Current-state architecture note
- Refactor map for `main.py`
- Golden test fixtures and expected outputs

### Exit Criteria

- Team can identify exactly which current behaviors are bugs, which are temporary, and which must remain stable through v2.0

## Phase 1: Correctness and Safety Hardening

### Objectives

- Fix the P0/P1 issues that invalidate current results or create unsafe runtime behavior

### Workstreams

#### 1. HDC Determinism and Checkpoint Integrity

Tasks:

- Persist the complete encoder state with the model checkpoint, or derive all role and level vectors from `seed + schema_version`
- Add an explicit encoder schema version
- Ensure the same input produces the same encoded output across process restarts
- Store checkpoint metadata for:
  - model weights
  - encoder state or seed
  - schema version
  - label map
  - feature schema
  - model ID and model version
  - encoder version and encoder seed
  - taxonomy version
  - model weights hash and encoder hash
  - training dataset IDs and dataset fingerprints/hashes
  - training capture/session IDs where available
  - training start/completion timestamps
  - Git commit and build ID
  - validation metrics and final test metrics
  - calibration method and calibration artifact reference
  - OOD and decision thresholds
  - Python and PyTorch versions
  - artifact creation timestamp
- Introduce a versioned `ModelManifest` schema that is stored with every checkpoint and validated at load time

Acceptance criteria:

- A saved model and a restarted process encode identical inputs identically
- Checkpoints fail fast on schema mismatch rather than silently loading
- Every loaded model can be traced to its encoder, feature schema, taxonomy, training data fingerprints, calibration artifact, thresholds, Git commit, and evaluation metrics

#### 2. Canonical Feature Schema

Tasks:

- Introduce typed feature models for packet-level input
- Define one canonical representation shared by dataset parsing and Scapy parsing
- Replace dual extraction paths with a common normalization pipeline
- Introduce explicit missing/invalid feature handling

Acceptance criteria:

- Dataset and live packet paths produce the same typed schema for equivalent traffic
- Missing or malformed fields are surfaced explicitly rather than silently defaulted

#### 3. HDC Encoding Semantics

Tasks:

- Bind feature identity into numerical encoding using deterministic role hypervectors
- Add dedicated missing-value hypervectors
- Clamp quantization to `[0, NUM_LEVELS - 1]`
- Make categorical and numerical encoding behavior explicit and testable

Acceptance criteria:

- Different numerical features with the same scalar value no longer collapse to the same structural encoding
- Invalid values cannot index outside level bounds

#### 4. TCP Flags and Port Semantics

Tasks:

- Create one canonical TCP flag encoding used by training and inference
- Stop mixing ordinal flag encoding in one path with bitmask encoding in another
- Preserve numeric ports as numeric ports
- Map service names to correct numeric ports without collapsing unknowns to `9999`

Acceptance criteria:

- Equivalent TCP packets encode flags identically in training and live inference
- Unknown ports do not all collapse to the same arbitrary feature value

#### 5. Labels, Taxonomy, and Leakage Removal

Tasks:

- Remove label derivation from explanation keywords
- Stop using text explanations as classifier input where they directly encode the label
- Treat dataset-native ground truth labels as authoritative where available
- Route every external dataset label through a versioned taxonomy normalization mapper
- Permit heuristic or weak labels only for explicitly marked synthetic/weakly-supervised experiments; never mix them silently with authoritative labels
- Replace the binary/multiclass mismatch with a staged taxonomy plan
- Introduce an internal attack taxonomy and a normalization layer

Acceptance criteria:

- The classifier no longer trains on a circular label path
- Prediction indexes are valid and semantically consistent
- Production benchmark labels are sourced from dataset-native ground truth or are explicitly identified as synthetic/weak labels

#### 6. Response Safety

Tasks:

- Move malicious IP count updates to post-classification only
- Replace lifetime count bans with decaying risk scores and timestamps
- Add allowlists for protected infrastructure and trusted networks
- Replace “redirect_packet” behavior with explicit response adapters
- Define response adapters for `AlertAction`, `WebhookAction`, `SIEMAction`, `MirrorAction`, `RateLimitAction`, `TemporaryBlockAction`, and `QuarantineAction`
- Give every action capability metadata such as `reversible`, `requires_confirmation`, `default_ttl`, and `minimum_policy_level`
- Set default action policy to alert-only
- Make all future containment actions reversible and TTL-based where the underlying enforcement mechanism permits it

Acceptance criteria:

- Benign packets cannot create bans before classification
- The runtime never rewrites packets as a substitute for real enforcement
- Containment defaults remain disabled until explicit policy support exists

### Deliverables

- Deterministic encoder module
- Canonical packet feature schema
- Safe local policy v1
- Versioned checkpoint format
- Unit tests for the corrected behaviors

### Exit Criteria

- The current system is no longer invalidated by encoder drift, parser skew, label leakage, or unsafe pre-classification banning

## Phase 2: Repository Restructure

### Objectives

- Replace the single-file architecture with modular services and typed boundaries

### Proposed Module Layout

```text
packetflowai/
  capture/
  features/
  hdc/
  models/
  flows/
  anomaly/
  taxonomy/
    attack_taxonomy.py
    mitre_mapper.py
  reasoning/
    nim.py
    sanitizer.py
    tools/
  fusion/
  policy/
  actions/
  feedback/
    active_learning.py
  registry/
    manifests.py
    promotion.py
    rollback.py
  telemetry/
  datasets/
  api/
  cli/
  storage/
  config/
  tests/
```

### Tasks

- Move configuration out of `main.py`
- Introduce a package layout and CLI entrypoints
- Move globals behind service objects with lifecycle control
- Add typed domain models:
  - `PacketFeatures`
  - `FlowFeatures`
  - `LocalPrediction`
  - `NIMAssessment`
  - `ThreatAssessment`
  - `ResponseDecision`
  - `FeedbackRecord`
- Separate training, replay, and live capture commands
- Move generated artifacts out of the project root
- Add explicit interfaces for taxonomy mapping, MITRE mapping, evidence sanitization, read-only reasoning tools, active-learning selection, and model registry operations

### Deliverables

- Initial package structure
- Central configuration system
- Service interfaces replacing globals
- Clean CLI surface

### Exit Criteria

- `main.py` is reduced to an entrypoint or removed entirely
- Core logic is importable and testable without live packet capture

## Phase 3: Flow-Based Detection Core

### Objectives

- Re-center the product around flows rather than isolated packets

### Workstreams

#### 1. Flow Tracking

Tasks:

- Implement bidirectional 5-tuple flow identity
- Maintain flow windows and lifecycle transitions
- Add start, update, close, and timeout behavior

Acceptance criteria:

- Flows can be replayed deterministically from packet sequences

#### 2. Temporal Flow Features

Tasks:

- Add duration, packet count, byte count, packets/sec, bytes/sec
- Add forward and reverse counts
- Add packet length statistics
- Add inter-arrival statistics
- Add SYN, ACK, FIN, RST, retransmission, burstiness, and state features

Acceptance criteria:

- Flow windows expose stable typed temporal summaries suitable for both local inference and NIM escalation

#### 3. Host Behaviour Features

Tasks:

- Track rolling unique destination hosts and ports
- Track fan-out, connection frequency, failure indicators, protocol mix, inbound/outbound ratios, and entropy

Acceptance criteria:

- Host context is available without leaking unbounded state or permanent labels

#### 4. Protocol Coverage

Tasks:

- Add proper UDP support
- Add proper IPv6 support
- Add optional DNS, TLS, and plaintext HTTP metadata extraction
- Keep payload-derived metadata separated from core telemetry

Acceptance criteria:

- UDP and IPv6 are first-class citizens in the canonical pipeline

#### 5. Temporal HDC

Tasks:

- Add sequence-aware HDC operations such as permutation or rotation
- Encode ordered event patterns such as handshake progression or burst structure

Acceptance criteria:

- The HDC representation carries temporal information instead of being a flat bundle of independent packet attributes

### Deliverables

- Flow engine
- Flow feature extractor
- Temporal HDC encoder
- Replay-ready ingestion path

### Exit Criteria

- The product can classify flow windows consistently in both replay and live modes

## Phase 4: Local Detection Stack

### Objectives

- Build a robust local detector before introducing external reasoning

### Workstreams

#### 1. Prototype-Based HDC Classification

Tasks:

- Maintain per-class prototype hypervectors
- Score flows by similarity to class prototypes
- Preserve similarity scores as first-class evidence

#### 2. Neural Head

Tasks:

- Retain the neural classifier as a secondary local model
- Benchmark HDC prototype versus HDC + MLP versus hybrid approaches

#### 3. Unknown and OOD Detection

Tasks:

- Add thresholds for low similarity, low margin, high entropy, or abnormal energy
- Support explicit `UNKNOWN` or `SUSPICIOUS` outcomes

#### 4. Anomaly Model

Tasks:

- Build a baseline model of expected host or environment traffic
- Score deviations independently of supervised class labels

#### 5. Confidence Calibration

Tasks:

- Add validation-based calibration
- Track calibration quality via Expected Calibration Error and Brier score

#### 6. Taxonomy Design

Tasks:

- Introduce a two-stage taxonomy:
  - Stage A: `benign / malicious / unknown`
  - Stage B: attack family
- Normalize external dataset labels into the internal taxonomy
- Add a deterministic ATT&CK mapper from internal attack families/evidence to candidate techniques
- Treat NIM ATT&CK suggestions as corroborating/challenging evidence only; NIM is not the authoritative ATT&CK mapper

#### 7. UNKNOWN / OOD Clustering

Tasks:

- Cluster UNKNOWN and OOD flow representations using HDC similarity plus selected behavioral features
- Track stable emerging-cluster IDs, sample counts, first/last seen times, source diversity, internal similarity, and common characteristics
- Escalate sufficiently coherent or high-risk UNKNOWN clusters for analyst review and optional NIM cluster-level reasoning
- Keep cluster hypotheses separate from confirmed attack-family labels until reviewed or otherwise validated

Acceptance criteria:

- Repeated structurally similar UNKNOWN events can be grouped into an emerging behavior family instead of remaining isolated alerts
- Cluster membership and hypotheses are stored as evidence and never silently become supervised labels

### Deliverables

- Prototype classifier
- Neural classifier
- OOD detector
- Anomaly model
- Calibration artifacts
- Internal taxonomy mapper
- Deterministic MITRE ATT&CK mapper
- UNKNOWN/OOD clustering module

### Exit Criteria

- Local detection exposes at least:
  - calibrated classifier confidence
  - prototype similarity
  - anomaly score
  - unknown / OOD indication
  - emerging UNKNOWN/OOD cluster ID when applicable
  - deterministic ATT&CK candidates with mapping provenance where applicable

## Phase 5: Dataset and Evaluation Framework

### Objectives

- Replace weak data handling with a reproducible benchmarking framework

### Tasks

- Add train, validation, and untouched test splits
- Prefer session-, source-, or capture-aware splits over random row splits
- Create dataset adapters for multiple intrusion datasets
- Initial benchmark suite should include:
  - CIC-IDS2017
  - CSE-CIC-IDS2018
  - UNSW-NB15
  - the existing packet-tag-explanation dataset as a legacy/comparison input
- Map external labels into one PacketFlowAI taxonomy
- Preserve the original/native dataset label alongside the normalized PacketFlowAI label for provenance
- Add benchmark and report generation commands
- Define concrete CLI surfaces such as:
  - `packetflowai benchmark --dataset cicids2017 --model candidate-v2 --output results/cicids2017-v2.json`
  - `packetflowai benchmark compare hdc-prototype hdc-mlp hybrid`
- Define replay commands such as:
  - `packetflowai replay example.pcap`
  - `packetflowai replay example.pcap --realtime`
  - `packetflowai replay example.pcap --speed 10`
  - `packetflowai replay example.pcap --model candidate-v2 --output results.json`
- Enforce the invariant that replay and live capture use the exact same downstream flow, feature, detection, fusion, and policy pipeline
- Report security-relevant metrics:
  - macro-F1
  - per-class precision, recall, F1
  - false-positive rate
  - false-negative rate
  - confusion matrix
  - PR-AUC
  - ROC-AUC where appropriate
  - unknown detection quality
  - calibration error

### Deliverables

- Dataset adapters
- Evaluation framework
- Benchmark CLI
- Machine-readable reports

### Exit Criteria

- Benchmark results are reproducible and test data remains untouched until final evaluation
- Dataset-native labels and normalized labels are both traceable in reports
- Replay and live capture pipeline equivalence is covered by integration tests

## Phase 6: NVIDIA NIM Reasoning Layer

### Objectives

- Add optional reasoning escalation without making cloud dependence a requirement

### Tasks

- Introduce a `ReasoningProvider` abstraction
- Implement an NVIDIA NIM provider using OpenAI-compatible APIs
- Make the model name configurable
- Read `NVIDIA_API_KEY`, `NIM_BASE_URL`, and `NIM_MODEL` from environment/secret management only
- Never log, serialize into snapshots, store in checkpoints, or commit the NVIDIA API key; exclude `.env` files and add CI secret scanning
- Escalate per flow, event window, or coherent UNKNOWN cluster, not per packet
- Send structured evidence only
- Add a `NIMEvidenceSanitizer` boundary before cloud calls
- Disable raw payload transmission by default
- Support configurable redaction/truncation for internal IPs, hostnames, usernames, URLs, DNS/TLS/HTTP-derived strings, and other potentially sensitive values
- Treat all network-derived strings as hostile data and isolate them from prompt instructions
- Define strict output schema with fields such as:
  - verdict
  - attack_family
  - nim_assessed_confidence or nim_reasoning_strength
  - evidence
  - contradictions
  - unknown_indicators
  - mitre_techniques
  - recommended_action
  - reason
- Validate and reject malformed responses
- Protect prompts from network-derived prompt injection
- Explicitly document that NIM self-reported confidence is not a calibrated classifier probability
- Add timeout, retry, concurrency caps, caching, and circuit breaking
- Add NIM modes:
  - `disabled`: no cloud reasoning
  - `shadow`: record NIM assessment but do not let it influence the final decision
  - `influence`: allow validated NIM evidence to affect fusion within configured policy bounds
- Introduce optional read-only reasoning tools for historical flow lookup, threat-intelligence lookup, deterministic MITRE lookup, and asset context
- Prohibit all NIM tools from exposing enforcement operations such as block, kill connection, or quarantine
- Keep the provider read-only and non-authoritative for enforcement
- Track detailed telemetry for request count, failures, timeouts, latency, input/output tokens, cache hits/misses, escalation rate, local/NIM agreement, disagreement, decision changes, and post-adjudication correctness
- Measure whether NIM reduces false positives/false negatives on uncertain traffic rather than treating integration alone as success

### Deliverables

- NIM provider
- Escalation policy
- Structured schema validator
- NIM telemetry and effectiveness report
- NIM evidence sanitizer
- Read-only NIM reasoning tool interfaces
- Shadow-mode evaluator

### Exit Criteria

- Local classification continues when NIM is disabled or unavailable
- NIM cannot directly trigger containment
- Raw packet payloads do not leave the host by default
- NIM shadow mode can run at production traffic volume without changing real decisions
- NIM self-reported confidence remains separately labeled from calibrated local-model confidence

## Phase 7: Fusion and Policy Engine

### Objectives

- Convert multiple evidence sources into a transparent decision pipeline

### Tasks

- Expose at least:
  - classifier confidence
  - prototype similarity
  - anomaly score
  - NIM assessed confidence/reasoning strength and advice
  - final risk score
- Keep calibrated local classifier confidence, prototype similarity, anomaly score, and NIM reasoning strength as semantically separate evidence channels
- Never numerically average NIM self-reported confidence with calibrated classifier probability as if they were equivalent quantities
- Start with deterministic fusion rules
- Add a trainable fusion/calibration model later once adjudicated data exists
- Define policy levels:
  - `NORMAL`
  - `OBSERVE`
  - `SUSPICIOUS`
  - `LIKELY_MALICIOUS`
  - `HIGH_CONFIDENCE_ATTACK`
  - `CONTAIN`
- Require reversible actions with TTL and expiry metadata
- Preserve full decision provenance
- Build an explanation layer for analysts
- Add response adapters for alert, webhook, SIEM/syslog/CEF/JSON export, mirror, rate-limit, temporary block, and quarantine
- Keep the policy engine independent from the enforcement technology so adapters can be replaced without changing detection logic

### Deliverables

- Fusion engine
- Policy engine
- Explanation engine
- Reversible action model
- Operational response adapters including SIEM/SOC-friendly outputs

### Exit Criteria

- Every decision has inspectable evidence and versioned provenance
- NIM evidence can influence decisions only in configured influence mode and can never bypass deterministic policy gates

## Phase 8: Runtime and Performance Engineering

### Objectives

- Make the runtime resilient under real traffic conditions

### Tasks

- Replace unbounded queues with bounded queues and explicit policies
- Separate capture from downstream processing
- Add batch inference
- Add capture filtering where appropriate
- Keep Scapy as one backend, but define a pluggable `CaptureBackend` interface
- Implement/retain `ScapyCaptureBackend` and `PcapReplayBackend` behind that interface
- Document future high-throughput backends such as AF_PACKET, eBPF, and XDP so core logic does not acquire Scapy-specific assumptions
- Improve shutdown handling with explicit lifecycle control
- Add queue-drain semantics
- Measure:
  - packets/sec
  - flows/sec
  - feature latency
  - HDC latency
  - inference p50/p95/p99
  - dropped packets
  - queue depth
  - memory and CPU/GPU utilization

### Deliverables

- Runtime pipeline redesign
- Backend abstraction
- Performance telemetry
- Load-test harness

### Exit Criteria

- The runtime degrades predictably under pressure rather than failing silently or consuming unbounded memory

## Phase 9: Storage, API, Telemetry, and Feedback

### Objectives

- Add the operational systems required for investigation, feedback, and monitoring

### Tasks

- Add a local SQLite event database
- Store flows, alerts, decisions, evidence, NIM assessments, and analyst feedback
- Replace `packet_feedback.txt` with structured feedback records
- Explicitly prohibit NIM assessments from automatically becoming supervised training labels
- Require analyst-confirmed/adjudicated labels before records enter the supervised feedback dataset
- Preserve model prediction, NIM assessment, disagreement state, analyst decision, timestamp, and provenance in every feedback record
- Add an `ActiveLearningSelector` that prioritizes analyst review for:
  - high anomaly with low classifier confidence
  - HDC/neural disagreement
  - local/NIM disagreement
  - new OOD or UNKNOWN clusters
  - high-risk UNKNOWN events
  - novel protocol/behavior patterns
  - high-impact containment candidates
- Add structured JSON logging
- Add metrics export compatible with Prometheus or OpenTelemetry-style collection
- Add read-only APIs:
  - `/health`
  - `/metrics`
  - `/flows`
  - `/alerts`
  - `/models`
  - `/feedback`
  - `/status`
  - `/config`
- Add a model registry with `candidate`, `active`, and `previous` lifecycle states
- Add atomic model promotion and one-command rollback
- Require offline evaluation before promotion and support shadow deployment of candidate models before activation
- Add CLI operations such as:
  - `packetflowai model list`
  - `packetflowai model evaluate candidate-v27`
  - `packetflowai model promote candidate-v27`
  - `packetflowai model rollback`
- Store model/encoder/calibration artifacts outside normal source history and keep only source, configs, schemas, and manifests in Git
- Add drift detection over:
  - feature distributions
  - class distributions
  - unknown rate
  - confidence distributions
  - prototype similarity

### Deliverables

- Event database
- Feedback store
- API service
- Structured telemetry
- Drift detection module
- Active-learning review queue
- Model registry with promotion, shadowing, and rollback

### Exit Criteria

- Analysts can inspect past decisions and feed corrections back into the system without reading log files directly
- Only adjudicated labels can enter supervised retraining datasets
- Candidate models can be evaluated/shadowed, promoted atomically, and rolled back without overwriting the active model in place

## Phase 10: Tooling, Tests, CI, Docs, and Dashboard

### Objectives

- Make the repository reproducible, governable, and demonstrable

### Tasks

- Migrate from `requirements.txt` to `pyproject.toml`
- Define supported Python versions
- Remove unused dependencies such as `torchvision` if still unused
- Add linting, formatting, typing, and import checks
- Add unit tests for:
  - deterministic HDC output
  - role binding
  - quantization boundaries
  - parser parity
  - TCP flag parity
  - UDP and IPv6 handling
  - label mapping
  - calibration
  - risk decay
  - policy gates
  - NIM schema validation
  - NIM failure behavior
  - NIM shadow/influence mode separation
  - NIM evidence sanitization and redaction
  - prohibition on NIM-generated training labels
  - deterministic MITRE mapping provenance
  - UNKNOWN/OOD cluster stability
  - candidate model promotion and rollback
- Add PCAP integration tests
- Add adversarial tests for malformed packets, floods, odd flags, fragmentation, strange DNS/TLS metadata, prompt injection strings, and API failures
- Add GitHub Actions for lint, tests, typing, dependency audit, benchmark smoke tests, and secret scanning
- Add `.gitignore` coverage for runtime artifacts, `.env` files, model outputs, caches, SQLite databases, and temporary PCAPs
- Define artifact ownership explicitly:
  - Git: source, configs, schemas, manifests
  - Model registry/releases: weights, encoders, calibration artifacts, benchmark reports
  - Runtime storage: logs, SQLite databases, caches, temporary PCAPs
- Rewrite the README around the actual architecture
- Document the threat model and limitations
- Publish reproducible benchmark methodology
- Add a basic dashboard for live flows, alerts, risk, uncertainty, agreement, and active containment

### Deliverables

- CI pipeline
- Expanded test suite
- Updated docs
- Dashboard
- Clean artifact management

### Exit Criteria

- The project is maintainable, test-gated, documented, and demo-ready

## Cross-Phase Dependency Graph

The most important implementation dependencies are:

1. Canonical feature schema before flow intelligence
2. Deterministic encoder before checkpoint versioning and benchmark credibility
3. Flow engine before NIM escalation design
4. Local unknown/anomaly signals before uncertainty gating
5. Evaluation framework before benchmark claims
6. Provenance and policy layers before automated containment
7. Storage and feedback before active learning or drift response
8. Dataset-native labels and taxonomy versioning before credible multiclass benchmark claims
9. Deterministic MITRE mapping before using ATT&CK as production evidence
10. NIM sanitizer and shadow mode before NIM can influence production decisions
11. Analyst adjudication before feedback enters supervised retraining
12. Model registry and provenance before candidate promotion/rollback

## Recommended PR Sequence

The following PR sequence keeps changes reviewable.

1. Project skeleton, package layout, config system, typed models, and artifact boundaries
2. Deterministic HDC encoder, expanded `ModelManifest`, and versioned checkpoint format
3. Canonical packet feature schema and dataset/live parser parity
4. TCP flag fix, quantization fix, missing-value support, port handling fix
5. Dataset-native label handling, taxonomy normalization, and leakage removal
6. Safe policy v1 with alert-only default, allowlists, explicit response adapters, and risk decay scaffolding
7. Flow engine and canonical flow feature extraction
8. UDP and IPv6 support
9. Temporal HDC and host behavioral features
10. Prototype classifier, neural head cleanup, OOD detector, anomaly model, and confidence calibration
11. Deterministic MITRE mapper and UNKNOWN/OOD clustering
12. Multi-dataset evaluation and benchmark framework with named benchmark suite
13. Replay CLI and live/replay equivalence integration fixtures
14. NIM evidence sanitizer, provider, secret handling, uncertainty gate, caching, and circuit breaker
15. NIM shadow mode, read-only reasoning tools, and detailed effectiveness telemetry
16. Fusion engine, policy levels, explanation engine, and SIEM/SOC response outputs
17. Storage, structured feedback, analyst adjudication workflow, active-learning selector, and drift detection
18. Model registry, candidate shadow deployment, promotion, and rollback
19. Performance hardening, capture backend abstraction, queue redesign, batching, and shutdown cleanup
20. API and dashboard
21. CI, secret scanning, docs, packaging, artifact hygiene, and benchmark publication
22. Production acceptance run covering model/NIM disagreement, rollback, UNKNOWN clustering, and end-to-end provenance

## Backlog by Priority

### P0: Must Fix Before Claiming Correctness

- Deterministic encoder persistence or seeded schema-based derivation
- Feature identity binding in HDC numerical encoding
- Canonical TCP flag encoding
- Canonical dataset/live feature pipeline
- Quantization clamping and missing values
- Typed parser replacing fragile regex logic
- Correct port handling
- Binary versus multiclass consistency
- Remove label leakage and weak keyword labels
- Enforce dataset-native labels as authoritative where available
- Expand checkpoint provenance with model/data/build/calibration fingerprints
- Post-classification risk updates only
- Alert-only default response policy

### P1: Required for v2 Local Detection Credibility

- Flow-based classification
- Temporal and behavioral features
- UDP and IPv6 support
- Prototype classifier
- OOD and anomaly detection
- Calibration
- Proper train/validation/test evaluation
- Dataset normalization across multiple sources
- Benchmark and replay modes
- Model and encoder versioning
- Deterministic MITRE mapping
- UNKNOWN/OOD clustering
- Named multi-dataset benchmark suite

### P2: Required for NIM-Backed Reasoning

- Reasoning provider abstraction
- Uncertainty gate
- Structured NIM output
- Prompt injection defenses
- NIM caching and circuit breaking
- NIM evidence sanitization and cloud-data redaction
- NIM shadow mode
- Read-only reasoning tools
- Explicit separation between NIM reasoning strength and calibrated classifier confidence
- Detailed NIM effectiveness telemetry
- Fusion engine
- Provenance and explanation engine

### P3: Required for Productization

- Runtime hardening
- Event database
- API
- Dashboard
- Feedback workflow
- Analyst-only promotion of feedback into supervised labels
- Active-learning review queue
- Model registry with candidate/active/previous lifecycle
- Atomic promotion and rollback
- SIEM/SOC response outputs
- Explicit artifact ownership and secret scanning
- Drift detection
- CI/CD and repository hygiene
- Documentation and benchmark publication

## Acceptance Standard for v2.0

v2.0 should not be declared complete until:

- The encoder is deterministic across restarts
- Training and live paths use the same typed schema
- TCP flags and ports are encoded consistently
- Labels are derived from a defined taxonomy rather than explanation leakage
- The runtime no longer performs unsafe pre-classification bans
- Response behavior defaults to alert-only
- The monolith is split into importable, testable modules
- Every checkpoint carries complete model/encoder/schema/taxonomy/data/build/calibration provenance
- Dataset-native ground truth is used where available and weak labels are explicitly isolated

## Acceptance Standard for v2.1

v2.1 should not be declared complete until:

- Flow windows are the primary classification object
- UDP and IPv6 are supported through the canonical path
- Local detection exposes calibrated confidence, anomaly score, and prototype similarity
- Benchmarking uses validation and untouched test sets
- Replay uses the same pipeline as live capture
- Deterministic MITRE mapping exists independently of NIM
- Repeated UNKNOWN/OOD events can be clustered into emerging behavior families with stored provenance
- The initial CIC-IDS2017, CSE-CIC-IDS2018, UNSW-NB15, and legacy comparison adapters are benchmarkable

## Acceptance Standard for v2.2

v2.2 should not be declared complete until:

- NIM is optional and cleanly isolated
- Only uncertain or abnormal flows are escalated
- Structured NIM outputs are validated
- NIM cannot directly enforce containment
- Fusion preserves separate evidence channels rather than collapsing them into one opaque number
- NIM shadow mode is supported before influence mode
- Raw payloads remain local by default and cloud evidence passes through the sanitizer
- NIM self-reported confidence is not treated as a calibrated classifier probability
- NIM can use only approved read-only reasoning tools
- NIM effectiveness can be measured against post-adjudication outcomes

## Acceptance Standard for v2.3

v2.3 should not be declared complete until:

- Runtime load behavior is bounded and measurable
- Historical evidence is queryable
- Feedback and drift workflows exist
- CI gates quality
- Documentation reflects the real architecture
- The system is demonstrable through API and dashboard surfaces
- NIM assessments cannot automatically become supervised training labels
- Active-learning prioritization and analyst adjudication workflows exist
- Candidate models support shadow evaluation, atomic promotion, and rollback
- Operational outputs include webhook/SIEM-friendly adapters
- Source, model-registry, and runtime artifacts are cleanly separated

## Non-Negotiable Safety and Learning Rules

The following rules apply across all milestones:

- NIM assessments are evidence, not ground truth.
- NIM output must never automatically create supervised training labels.
- Only analyst-confirmed/adjudicated labels may enter supervised feedback datasets.
- NIM cannot directly execute containment or access mutating enforcement tools.
- Raw packet payloads remain local by default; cloud evidence is structured, minimized, sanitized, and configurable.
- NIM self-reported confidence is not a calibrated probability and must remain separately identified in fusion and telemetry.
- New local models and NIM influence must support shadow evaluation before they affect production decisions.
- Automated containment remains deterministic, policy-gated, allowlist-aware, provenance-backed, and reversible/TTL-based where supported.
- ATT&CK mapping must have deterministic provenance; NIM may corroborate or challenge but is not the authoritative mapper.
- UNKNOWN/OOD clusters remain hypotheses until validated and do not silently become attack-family labels.

## Model and Artifact Lifecycle

The intended lifecycle is:

```text
TRAIN
  ->
CANDIDATE MODEL
  ->
OFFLINE EVALUATION
  ->
SHADOW DEPLOYMENT
  ->
PROMOTION
  ->
ACTIVE MODEL
  ->
PREVIOUS MODEL
  ->
ROLLBACK IF REQUIRED
```

Artifact boundaries:

- Git stores source, configs, schemas, and manifests.
- The model registry/release store contains model weights, encoder artifacts, calibration artifacts, and benchmark reports.
- Runtime storage contains logs, SQLite databases, caches, feedback state, and temporary PCAP/replay artifacts.
- Model promotion must be atomic and must never rely on blindly overwriting a single production `.pth` file.

## NIM Effectiveness Standard

NIM should be judged by whether it improves uncertain-event handling, not by whether an LLM is present in the architecture.

Track at minimum:

- NIM request, failure, timeout, latency, and token metrics
- Cache hit/miss rate and escalation rate
- Local/NIM agreement and disagreement
- Cases where NIM changed, increased, or decreased risk
- Local model correctness after analyst adjudication
- NIM assessment correctness after analyst adjudication
- False-positive and false-negative changes for escalated traffic
- Cost/latency per useful corrected or clarified decision

NIM should move from `shadow` to `influence` mode only after measured evidence shows that the configured fusion policy improves outcomes without unacceptable latency, privacy, or false-positive costs.

## Immediate Next Step

The best first implementation slice is:

1. Introduce the package skeleton and central config
2. Implement deterministic encoder state and the expanded versioned `ModelManifest`/checkpoint metadata
3. Define the canonical packet feature schema
4. Unify dataset and live parsing onto that schema
5. Fix TCP flags, quantization, and port encoding
6. Replace weak label derivation with dataset-native ground-truth handling and a versioned taxonomy mapper
7. Replace unsafe banning logic with alert-only policy scaffolding and explicit response-adapter interfaces
8. Add unit tests covering all of the above

This slice resolves the most serious correctness problems while creating the foundation for the flow-centric redesign.

## Num code

```markdown

import requests

invoke_url = "https://integrate.api.nvidia.com/v1/chat/completions"
stream = False

headers = {
    "Authorization": "Bearer $NVIDIA_API_KEY",
    "Accept": "text/event-stream" if stream else "application/json",
}

payload = {
  "model": "minimaxai/minimax-m3",
  "messages": [
    {
      "role": "user",
      "content": ""
    }
  ],
  "temperature": 1,
  "top_p": 0.95,
  "max_tokens": 8192,
  "stream": stream
}

response = requests.post(invoke_url, headers=headers, json=payload, stream=stream)
if stream:
    for line in response.iter_lines():
        if line:
            print(line.decode("utf-8"))
else:
    print(response.json())

```