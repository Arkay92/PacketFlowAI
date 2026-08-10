# Architecture

PacketFlowAI uses one downstream pipeline for live capture and PCAP replay:

```text
CaptureBackend -> bounded RuntimeService -> FlowEngine -> FlowFeatures
  -> local HDC/neural evidence -> uncertainty gate -> optional sanitized NIM
  -> deterministic fusion -> policy -> response adapters -> SQLite evidence store
```

`ScapyCaptureBackend` and `PcapReplayBackend` are replaceable. Future AF_PACKET, eBPF, or XDP backends must emit the same packet callback and must not leak backend-specific assumptions into flow logic.

Evidence channels are intentionally separate. Calibrated classifier confidence, prototype similarity, anomaly score, and NIM reasoning strength are not interchangeable probabilities. NIM is disabled by default and cannot invoke enforcement.

Artifact ownership:

- Git: source, schemas, configuration examples, documentation, CI, and tests.
- Registry/release storage: model weights, encoder/calibration artifacts, evaluation reports.
- Runtime storage: SQLite databases, logs, caches, and temporary PCAPs.
