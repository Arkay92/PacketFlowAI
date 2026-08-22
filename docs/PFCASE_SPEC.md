# PFCASE Specification 1.0

Status: PacketFlowAI v5 published interoperability profile.

PFCASE is a ZIP container whose verification does not require PacketFlowAI. A conforming verifier must reject path traversal, duplicate file names, undeclared files when strict mode is enabled, hash mismatches, invalid Merkle roots, and invalid supplied signatures.

## Required Layout

```text
manifest.json
case.json
evidence/events.jsonl
evidence/observations.jsonl
evidence/sources.json
commitments/merkle-roots.json
commitments/epochs.json
commitments/sequence-roots.json
contracts/evidence-contract.json
omissions/ledger.jsonl
redactions/leaves.json
verification.json
```

Optional directories are `decisions/`, `authority/`, `policies/`, `models/`, `attestations/`, `anchors/`, and `signatures/`. Unknown files are reported and may be rejected by a strict verifier.

## Canonical Serialization

PF-CANONICAL-JSON-1 is UTF-8 JSON with keys sorted lexicographically, no insignificant whitespace, JSON scalar spellings, and ASCII escaping enabled. JSON Lines files contain one canonical object per line and end with LF when non-empty.

All digests are lowercase SHA-256 hexadecimal strings. `manifest.files` maps every file except `manifest.json` to its digest. The case Merkle tree uses file digests ordered by filename. Parent nodes hash the ASCII concatenation of left and right hexadecimal digests. An odd final node is duplicated.

## Signatures And Trust

`manifest_signature` is detached from the signed body: verify the canonical manifest after removing `manifest_signature`. This reference implementation supports HMAC-SHA256 for independently administered test/deployment trust domains. Public-key implementations must identify the algorithm and key ID without changing canonicalization or commitment rules.

Trust material is historical. Verification uses the key valid at the signed timestamp, including expired or subsequently rotated keys, and rejects keys revoked before that timestamp.

## Evidence Semantics

Producer evidence includes `producer_id`, `epoch_id`, `sequence_number`, event/receive/commit timestamps, clock provenance, payload, and evidence hash. Sequence continuity is only a statement about the committed range for one producer epoch.

An Evidence Contract is signed, versioned, hash-addressed, valid for a bounded interval, and declares expected sources and incident-specific requirements. Coverage means only that a contract-declared source contributed. It is not global completeness.

Epoch commitments bind producer, epoch, count, first/last sequence, first/last timestamp, root, and previous epoch root. Count anchoring detects removal of committed records. It does not prove that every real-world event was generated before commitment.

## Omission And Redaction

Controlled absence must create an immutable omission record. Defined kinds are `EVENT_REDACTED`, `EVENT_EXCLUDED`, `EVENT_REJECTED`, `EVENT_EXPIRED`, `EVENT_DESTROYED`, `SOURCE_DISABLED`, `SOURCE_UNAVAILABLE`, and `SOURCE_FILTER_CHANGED`.

A redacted leaf publishes the original commitment, reason, authority, and timestamp while withholding content. Selective disclosure remains linked to the committed root. Retention destruction similarly records range, policy, authority, timestamp, and previous commitment.

Unknown omission risk is always reported. No PFCASE assurance level means that an unobserved event could not have existed.

## Reproducibility

Each claim is classified as `DETERMINISTICALLY_REPRODUCIBLE`, `EXTERNALLY_VERIFIABLE`, `ATTESTED_BUT_NOT_REPRODUCIBLE`, or `NON_DETERMINISTIC_SUPPORTING_ANALYSIS`. NIM receipts prove the recorded request/output used at the time; they do not promise identical regenerated prose.

## Verifier Algorithm

1. Parse the ZIP safely and validate required paths.
2. Canonicalize and validate the manifest schema.
3. Hash every declared file and report missing, modified, and unexpected files.
4. recompute the case Merkle root and any requested inclusion proof.
5. Verify manifest, producer, epoch, receipt, witness, and anchor signatures against timestamped trust roots.
6. Check producer sequence ranges, epoch count/root links, receipt journals, and transparency consistency.
7. Evaluate expected-source coverage, heartbeats, dark periods, omissions, and redactions.
8. Re-derive deterministic decisions and label recorded-only analysis accurately.
9. Emit formal claims, limitations, and `unknown_omission_risk: NOT_ELIMINATED`.

## Time

Evidence preserves event, receive, and commit time independently. Sources declare clock source, estimated skew, and confidence. Ordering claims must identify which timestamp they use and surface skew that can change causal interpretation.

## Encryption And Disclosure

Evidence contents may be encrypted separately from public commitments. Sanitized, auditor, and SOC disclosures must resolve to the same committed root. Encryption does not change digest, redaction, omission, or assurance semantics.

