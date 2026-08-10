# PacketFlowAI

PacketFlowAI is a flow-centric network detection prototype built around deterministic hyperdimensional encoding, local neural/prototype/anomaly evidence, optional bounded NVIDIA NIM reasoning, and a conservative policy engine. Live capture and PCAP replay use the same flow, detection, fusion, policy, and evidence pipeline.

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
