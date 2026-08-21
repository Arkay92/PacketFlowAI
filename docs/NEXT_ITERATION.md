# PacketFlowAI v4 Architecture

PacketFlowAI v4 groups the master backlog into testable capability families rather than 156 disconnected switches.

## Trust Boundary

`packetflow_verifier` is a separate standard-library-only package. It does not import `packetflowai`. Portable `.pfcase` bundles contain canonical records, model reproducibility metadata, policy and authority records, a sequential chain, per-file hashes, a Merkle root, witness attestations, and an optional append-only anchor receipt. Verification, evidence diffing, and recorded-input/output decision replay are deterministic.

Witness HMAC is suitable for deployments with independently managed shared secrets. Production public-key witnesses or transparency services implement the same detached-root boundary without changing bundle contents.

## Intelligence Plane

- Temporal campaigns and causal links retain alternatives rather than declaring causation from proximity.
- Predictions expose 5-minute, 1-hour, and 24-hour horizons, confidence decomposition, calibration, pruning, and conformal-style prediction sets.
- The digital twin models assets, dependencies, trust, reachability, exposure, criticality, and infrastructure changes.
- The intervention solver chooses the most reversible action that crosses a configured residual-risk threshold without a known critical dependency.
- The time machine reconstructs all supplied state categories into `KNOWN THEN` and `LEARNED LATER` and reports hindsight leakage.
- Authority is data: expiring grants, relationship scopes, two-person approval, break-glass records, policy simulation, and diffs.

## Learning And Federation

Threat memory supports provisional concepts, analyst promotion, prototype lineage, forks, decay, and per-asset seasonal baselines. Federation exchanges fingerprints and similarity signatures, applies site reputation, rejects outliers and impossible claims, and never accepts raw traffic.

## SOC Fabric

Native, Zeek, Suricata, EDR, identity, cloud, DNS, and application adapters converge on `CanonicalEvidenceEvent`. The same evidence can be mapped to OCSF, STIX 2.1, TAXII, CEF, syslog, JSON SIEM payloads, OpenTelemetry log events, or case-management records. Sigma candidates remain shadow-only until repository tests pass.

## Runtime Boundary

Scapy remains portable. Linux deployments can use the operational `AF_PACKET` receive path and attach a supplied XDP object when privileges permit. AF_XDP, GPU, FPGA, GNN, and temporal-graph integrations are explicit optional backends; PacketFlowAI never reports them active without their platform/runtime prerequisites.

Adaptive batching, risk-aware load shedding, sensor budgets, dynamic sampling, deception weighting, evasion tests, feedback/federation poisoning checks, grounding scores, and prompt-injection regression hooks protect the runtime under pressure and adversarial input.

## API

Existing `/v3/*` contracts remain available. `/v4/*` adds overview, predictions, causal analysis, intervention, digital twin, complete time machine, authority, explainability, runtime, and domain status resources.
