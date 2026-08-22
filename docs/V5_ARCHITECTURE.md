# PacketFlowAI v5 Architecture

## Product Principle

PacketFlowAI produces the record but is not the only component capable of proving it. Integrity is not completeness. Known gaps remain visible, unknown omissions are never declared impossible, redaction and destruction leave evidence, and consequential authority weakens when assurance weakens.

## Evidence Plane

Sensors emit source identity, epoch, monotonic sequence, time provenance, signed heartbeats, and evidence. Ingest returns signed receipts retained by producer journals. Critical deployments can dual-write to an independent sink and appraise TPM, TEE, confidential-VM, or signed-collector evidence through a RATS-style attester/verifier boundary.

Epoch manifests bind event count, first/last sequence, first/last timestamp, Merkle root, and previous root. Witnesses observe checkpoints through multiple transparency services and reconcile views. Split roots are a security event. Historical trust roots preserve old verification across rotation, expiry, personnel change, and replacement.

## Assurance Plane

The applicable signed Evidence Contract defines expected sources before the incident. Producer continuity, source contribution, heartbeats, dark periods, recording-path counts, receipt journals, clock skew, cross-source asymmetry, contradictions, omissions, redactions, and retention receipts produce an assurance vector. Unknown omission risk remains `NOT_ELIMINATED`.

Threat risk and evidence assurance are independent. Policy sets minimum evidence channels and desired context. Missing required evidence can deny or elevate authority; missing desired context creates assurance debt. Sudden assurance loss and sensor suppression influence investigation ranking.

## Reproducibility Plane

Lifecycle provenance binds input hashes, algorithm/configuration versions, output hashes, and timestamps. Decision capsules include source evidence, features, model, calibration, taxonomy, policy, authority, and result. Deterministic local operations are re-derived. NIM output receives an integrity-protected reasoning receipt and remains explicitly non-deterministic.

## Verification Boundary

`packetflow_verifier` is standard-library-only and imports no `packetflowai` modules. It validates PFCASE schema, file commitments, Merkle inclusion, chain continuity, source coverage, external checkpoints, decision replay, and formal claims. Challenge mode maps deliberate evidence attacks to the control that detects them. Its browser app performs local verification without an application runtime or evidence upload.

The audit API exposes only manifest, public keys, schema, inclusion/consistency proof metadata, checkpoints, and Evidence Contracts. It does not expose operational application state.

## Disclosure

Confidential evidence and public commitments are separate. SOC, auditor, customer, and regulator disclosures may reveal different records while retaining one committed root. Withheld content uses declared redacted leaves. Lifecycle-controlled destruction uses signed receipts. Verification never equates the highest assurance level with omniscient observation.
