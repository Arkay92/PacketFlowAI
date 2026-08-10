# Threat Model and Limitations

## Protected Assets

- Model and encoder integrity
- Analyst feedback provenance
- Network evidence and identifiers
- Enforcement controls
- NVIDIA API credentials

## Trust Boundaries

Network packets, payload metadata, PCAPs, dataset strings, and NIM output are untrusted. Dataset labels are accepted only through explicit adapters. NIM evidence passes through redaction and output validation; raw payloads are excluded by default. API endpoints are read-only and bind to localhost by default.

## Safety Invariants

- NIM output is evidence, never ground truth or an automatic training label.
- Only identified analyst-adjudicated feedback enters supervised data.
- NIM has no mutating tools and cannot directly request enforcement.
- Containment is disabled by default and requires deterministic local gates, confirmation, reversibility, and TTL.
- Legacy or mismatched checkpoints fail closed during manifest validation.

## Known Limitations

- Scapy is suitable for prototyping, not line-rate capture.
- Encrypted payload content is not inspected.
- Protocol metadata parsing is intentionally shallow and bounded.
- The included neural architecture requires environment-specific training and calibration before operational use.
- Benchmark quality depends on capture-aware splits and correct native-label mappings.
