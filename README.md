# PacketFlowAI

PacketFlowAI is a predictive cyber-defence prototype built around a temporal threat world model, deterministic hyperdimensional encoding, local neural/prototype/anomaly evidence, optional bounded NVIDIA NIM reasoning, and an explicit authority graph. Live capture and PCAP replay use the same flow, detection, fusion, policy, evidence, and campaign pipeline.

## Features

- Deterministic HDC encoding with checkpoint integrity validation
- Bidirectional IPv4/IPv6 TCP/UDP flow tracking with temporal and host features
- Temporal HDC, prototype similarity, OOD detection, anomaly scoring, and calibration
- Dataset-native label normalization without explanation leakage
- Named dataset adapters and machine-readable security benchmark reports
- Optional NIM disabled/shadow/influence modes with evidence sanitization
- Deterministic fusion, ATT&CK provenance, alert-only defaults, and TTL containment gates
- SQLite evidence, analyst-adjudicated feedback, drift checks, and active learning
- Candidate/active/previous model registry with promotion and rollback
- Read-only API, Prometheus metrics, structured logs, and operations dashboard
- Persistent campaign graph connecting flows, hosts, accounts, services, alerts, ATT&CK techniques, and cases
- Next-move prediction with uncertainty, evidence references, and time horizons
- Counterfactual response simulation and digital-twin blast-radius analysis
- Evidence time reconstruction, hash-chain sealing, Merkle verification, and explicit action authority
- Multi-sensor adapters, threat memory, guarded continual prototypes, federated consensus, and bounded playbooks

## Installation
1. Clone the repository.

```bash
git clone https://github.com/Arkay92/PacketFlowAI.git
```

2. Install dependencies.

```bash
cd PacketFlowAI
pip install -e .
```

## Usage

List available commands:

```bash
python -m packetflowai --help
```

Train against a dataset containing authoritative labels:

```bash
python -m packetflowai train --dataset rdpahalavan/packet-tag-explanation --split train --epochs 10
```

List interfaces and start live capture:

```bash
python -m packetflowai interfaces
python -m packetflowai capture --interface <interface_name>
```

Replay a PCAP through the same flow pipeline:

```bash
python -m packetflowai replay traffic.pcap
python -m packetflowai replay traffic.pcap --realtime --speed 10 --output artifacts/replay.json
```

Benchmark, model lifecycle, API/dashboard, and load testing:

```bash
python -m packetflowai benchmark run --dataset cicids2017 --input test.csv \
  --predictions predictions.json --model candidate-v2 --output artifacts/benchmarks/report.json
python -m packetflowai model list
python -m packetflowai model promote packet-hv-mlp:2.3.0
python -m packetflowai api --host 127.0.0.1 --port 8080
python -m packetflowai loadtest --flows 10000 --packets-per-flow 4
```

The API command starts the **Signal Room** at `http://127.0.0.1:8080`. The live dashboard visualizes flow activity, threat pressure, classifications, alerts, evidence channels, model state, and runtime telemetry. It refreshes every four seconds and exposes a pause control for incident review.

The **Forensics** view isolates orange and red packet conversations for deeper analysis. Select cases from the threat constellation or case strip, use the arrow keys to move between incidents, and inspect route identity, packet statistics, directional volume, model evidence, protocol metadata, policy provenance, and the correlated read-only record.

The **Command** view is the v3 predictive-defence cell. Navigate the temporal threat graph, inspect campaign and next-move assessments, compare counterfactual responses, rewind the evidence timeline, verify the sealed forensic chain, and inspect exactly where autonomous authority ends.

Create the guided Signal Room presentation video with Playwright:

```bash
pip install -e ".[presentation]"
playwright install chromium
python scripts/create_ui_video.py
```

The recorder uses isolated presentation data and writes `artifacts/presentation/packetflowai-signal-room-tour.webm`. Each explanatory card and dashboard view is held for at least six seconds.

Record the interactive Forensic War Room analyst journey with:

```bash
python scripts/create_forensics_video.py
```

This walkthrough uses a visible guided cursor, real clicks, smooth scrolling, case navigation, and the same UI-styled explanatory cards. It writes `artifacts/presentation/packetflowai-forensics-war-room-tour.webm`.

Record the complete v3 journey with:

```bash
python scripts/create_v3_video.py
```

The v3 walkthrough uses real navigation, graph selection, response comparisons, timeline controls, Forensic War Room movement, and UI-styled cards held for at least five seconds. It writes `artifacts/presentation/packetflowai-v3-predictive-defence-tour.webm`.

`python main.py ...` remains available as a compatibility launcher. Live capture requires appropriate packet-capture permissions and Npcap on Windows.

## Configuration

Configuration is defined in `packetflowai/config.py`. Common runtime overrides are available through environment variables:

- `PACKETFLOWAI_ARTIFACT_DIR`
- `PACKETFLOWAI_HV_DIMENSION`
- `PACKETFLOWAI_NUM_LEVELS`
- `PACKETFLOWAI_ENCODER_SEED`
- `PACKETFLOWAI_QUEUE_SIZE`
- `PACKETFLOWAI_RISK_HALF_LIFE`
- `PACKETFLOWAI_NIM_MODE` (`disabled`, `shadow`, or `influence`)
- `NIM_BASE_URL`
- `NIM_MODEL`
- `NVIDIA_API_KEY` (environment/secret manager only)

NIM is disabled by default. Its self-reported reasoning strength is not a calibrated probability, it cannot invoke enforcement, and its assessments cannot become training labels. Generated checkpoints, registry state, databases, reports, and logs live under `artifacts/` and are excluded from Git. Legacy state-dict-only checkpoints intentionally fail manifest validation.

See [architecture](docs/ARCHITECTURE.md), [threat model](docs/THREAT_MODEL.md), and [benchmark methodology](docs/BENCHMARKS.md).

## Contributing

Run the test suite with:

```bash
python -m unittest discover -v
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
