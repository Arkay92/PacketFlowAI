# Reproducible Benchmark Methodology

Supported adapters are `cicids2017`, `cse-cic-ids2018`, `unsw-nb15`, and `packet-tag-explanation`. Adapters preserve native labels and normalize them into the PacketFlowAI taxonomy.

Use capture, session, flow, or source groups for deterministic train/validation/untouched-test splits. Fit models, prototypes, anomaly baselines, and calibration only on training/validation data. Open the test split once for final reporting.

Reports include macro-F1, per-class precision/recall/F1, false-positive and false-negative rates, confusion matrix, PR-AUC, ROC-AUC where valid, unknown detection rate, and calibration error.

```bash
packetflowai benchmark run --dataset cicids2017 --input data.csv \
  --predictions predictions.json --model candidate-v2 --output artifacts/benchmarks/cicids2017-v2.json

packetflowai benchmark compare artifacts/benchmarks/*.json
```

`predictions.json` is an ordered JSON array of `{"label": "...", "malicious_score": 0.0}` records generated from the untouched test rows. Reports and model manifests should be retained together.
