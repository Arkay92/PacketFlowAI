"""PacketFlowAI command-line entrypoints."""

import argparse
import json
import logging
import time
from collections.abc import Sequence
from dataclasses import asdict, replace
from pathlib import Path

from .api import APIServer
from .benchmark import ADAPTERS, evaluate_predictions
from .capture import available_interfaces
from .config import AppConfig
from .flows import FlowEngine
from .hdc import HypervectorEncoder
from .inference import FlowInferenceService
from .loadtest import run_load_test
from .manifests import load_checkpoint
from .modeling import build_model, default_device
from .orchestrator import DetectionOrchestrator
from .reasoning import NIMProvider
from .registry import FilesystemModelRegistry
from .runtime import FlowRuntime, PcapReplayBackend, RuntimeService, ScapyCaptureBackend
from .storage import EventStore
from .telemetry import MetricsRegistry, configure_logging
from .training import TrainingService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="packetflowai", description="Flow-centric network detection runtime")
    parser.add_argument("--verbose", action="store_true")
    subcommands = parser.add_subparsers(dest="command", required=True)

    train = subcommands.add_parser("train", help="train a local model")
    train.add_argument("--dataset")
    train.add_argument("--split")
    train.add_argument("--epochs", type=int)

    capture = subcommands.add_parser("capture", help="classify live flows")
    capture.add_argument("--interface", required=True)
    capture.add_argument("--filter", dest="capture_filter")

    replay = subcommands.add_parser("replay", help="classify PCAP flows through the live pipeline")
    replay.add_argument("pcap", type=Path)
    replay.add_argument("--limit", type=int)
    replay.add_argument("--realtime", action="store_true")
    replay.add_argument("--speed", type=float, default=1.0)
    replay.add_argument("--output", type=Path)

    benchmark = subcommands.add_parser("benchmark", help="generate or compare benchmark reports")
    benchmark_commands = benchmark.add_subparsers(dest="benchmark_command", required=True)
    benchmark_run = benchmark_commands.add_parser("run")
    benchmark_run.add_argument("--dataset", choices=sorted(ADAPTERS), required=True)
    benchmark_run.add_argument("--input", type=Path, required=True)
    benchmark_run.add_argument("--predictions", type=Path, required=True)
    benchmark_run.add_argument("--model", required=True)
    benchmark_run.add_argument("--output", type=Path, required=True)
    benchmark_compare = benchmark_commands.add_parser("compare")
    benchmark_compare.add_argument("reports", nargs="+", type=Path)
    benchmark_compare.add_argument("--output", type=Path)

    model = subcommands.add_parser("model", help="manage candidate/active/previous models")
    model_commands = model.add_subparsers(dest="model_command", required=True)
    model_commands.add_parser("list")
    register = model_commands.add_parser("register")
    register.add_argument("key", help="model-id:version")
    register.add_argument("artifact", type=Path)
    evaluate = model_commands.add_parser("evaluate")
    evaluate.add_argument("key")
    evaluate.add_argument("--report", type=Path, required=True)
    evaluate.add_argument("--shadow-validated", action="store_true")
    promote = model_commands.add_parser("promote")
    promote.add_argument("key")
    promote.add_argument("--skip-shadow-requirement", action="store_true")
    model_commands.add_parser("rollback")

    api = subcommands.add_parser("api", help="serve the read-only API and dashboard")
    api.add_argument("--host", default="127.0.0.1")
    api.add_argument("--port", type=int, default=8080)

    loadtest = subcommands.add_parser("loadtest", help="run the synthetic flow-engine load harness")
    loadtest.add_argument("--flows", type=int, default=10_000)
    loadtest.add_argument("--packets-per-flow", type=int, default=4)

    subcommands.add_parser("interfaces", help="list capture interfaces")
    return parser


def _components(config: AppConfig):
    device = default_device()
    encoder = HypervectorEncoder(
        config.model.hv_dimension,
        num_levels=config.model.num_levels,
        seed=config.model.encoder_seed,
    )
    model = build_model(config.model).to(device)
    return device, encoder, model


def _operational_services(config: AppConfig):
    config.artifacts.create()
    device, encoder, model = _components(config)
    registry = FilesystemModelRegistry(config.artifacts.registry)
    try:
        checkpoint = Path(registry.active_model()["artifact"])
    except RuntimeError:
        checkpoint = config.artifacts.model_checkpoint
    manifest = load_checkpoint(str(checkpoint), model, encoder, map_location=device)
    inference = FlowInferenceService(config, encoder, model, device, manifest.model_id, manifest.model_version)
    store = EventStore(config.artifacts.event_database)
    metrics = MetricsRegistry()
    reasoning = NIMProvider(config.nim) if config.nim.mode != "disabled" else None
    orchestrator = DetectionOrchestrator(
        inference, store, metrics, reasoning=reasoning, nim_mode=config.nim.mode
    )
    flow_runtime = FlowRuntime(FlowEngine(), orchestrator.handle_flow)
    return flow_runtime, orchestrator, store, metrics


def run_train(config: AppConfig) -> int:
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError("training requires the Hugging Face datasets package") from error
    dataset = load_dataset(config.training.dataset_id)[config.training.dataset_split]
    device, encoder, model = _components(config)
    manifest = TrainingService(config, encoder, model, device).fit(
        dataset,
        dataset_id=config.training.dataset_id,
        dataset_fingerprint=getattr(dataset, "_fingerprint", "unknown"),
    )
    FilesystemModelRegistry(config.artifacts.registry).register_candidate(
        manifest.model_id, manifest.model_version, config.artifacts.model_checkpoint
    )
    logging.info("Saved model %s version %s to %s", manifest.model_id, manifest.model_version,
                 config.artifacts.model_checkpoint)
    return 0


def _run_runtime(service: RuntimeService) -> None:
    service.start()
    try:
        while service.backend_thread and service.backend_thread.is_alive():
            time.sleep(0.2)
    except KeyboardInterrupt:
        logging.info("Shutdown requested")
    finally:
        service.stop(drain=True)
        service.join()


def run_capture(config: AppConfig, interface: str, capture_filter: str | None) -> int:
    pipeline, _, store, _ = _operational_services(config)
    service = RuntimeService(
        ScapyCaptureBackend(interface, capture_filter, config.runtime.capture_poll_seconds),
        pipeline,
        queue_size=config.runtime.queue_size,
    )
    try:
        logging.info("Capturing flows on %s; press Ctrl+C to stop", interface)
        _run_runtime(service)
        logging.info("Capture stopped metrics=%s", service.metrics())
    finally:
        store.close()
    return 0


def run_replay(config: AppConfig, path: Path, limit: int | None, realtime: bool,
               speed: float, output: Path | None) -> int:
    if not path.is_file():
        raise FileNotFoundError(f"PCAP file not found: {path}")
    pipeline, orchestrator, store, _ = _operational_services(config)
    service = RuntimeService(
        PcapReplayBackend(path, realtime=realtime, speed=speed, limit=limit),
        pipeline,
        queue_size=config.runtime.queue_size,
        overflow_policy="block",
    )
    try:
        _run_runtime(service)
        result = {"decisions": orchestrator.last_decisions, "metrics": service.metrics()}
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        logging.info("Replay complete metrics=%s", result["metrics"])
    finally:
        store.close()
    return 0


def run_benchmark(args: argparse.Namespace) -> int:
    if args.benchmark_command == "compare":
        reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.reports]
        comparison = sorted(
            ({"dataset_id": report["dataset_id"], "model_id": report["model_id"], "macro_f1": report["macro_f1"]}
             for report in reports),
            key=lambda item: -item["macro_f1"],
        )
        serialized = json.dumps(comparison, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(serialized, encoding="utf-8")
        else:
            print(serialized)
        return 0
    adapter = ADAPTERS[args.dataset]()
    records = adapter.from_csv(args.input)
    raw_predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    labels = [item["label"] if isinstance(item, dict) else item for item in raw_predictions]
    scores = [float(item.get("malicious_score", 0.0)) if isinstance(item, dict) else 0.0 for item in raw_predictions]
    report = evaluate_predictions(
        args.dataset, args.model,
        [record.native_label for record in records],
        [record.normalized_label for record in records],
        labels, scores,
    )
    report.write(args.output)
    print(json.dumps(asdict(report), indent=2))
    return 0


def run_model(config: AppConfig, args: argparse.Namespace) -> int:
    registry = FilesystemModelRegistry(config.artifacts.registry)
    if args.model_command == "list":
        print(json.dumps(registry.list_models(), indent=2))
    elif args.model_command == "register":
        model_id, version = args.key.split(":", 1)
        print(json.dumps(asdict(registry.register_candidate(model_id, version, args.artifact)), indent=2))
    elif args.model_command == "evaluate":
        print(json.dumps(registry.mark_evaluated(args.key, args.report, args.shadow_validated), indent=2))
    elif args.model_command == "promote":
        print(json.dumps(registry.promote(args.key, not args.skip_shadow_requirement), indent=2))
    else:
        print(json.dumps(registry.rollback(), indent=2))
    return 0


def run_api(config: AppConfig, host: str, port: int) -> int:
    config.artifacts.create()
    store = EventStore(config.artifacts.event_database)
    registry = FilesystemModelRegistry(config.artifacts.registry)
    server = APIServer(config, store, MetricsRegistry(), registry, host, port)
    logging.info("API and dashboard listening on http://%s:%s", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.stop()
    finally:
        store.close()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = AppConfig.from_env()
    if args.command == "train":
        config = replace(config, training=replace(
            config.training,
            dataset_id=args.dataset or config.training.dataset_id,
            dataset_split=args.split or config.training.dataset_split,
            epochs=args.epochs or config.training.epochs,
        ))
    configure_logging(config, args.verbose)
    if args.command == "train":
        return run_train(config)
    if args.command == "capture":
        return run_capture(config, args.interface, args.capture_filter)
    if args.command == "replay":
        return run_replay(config, args.pcap, args.limit, args.realtime, args.speed, args.output)
    if args.command == "benchmark":
        return run_benchmark(args)
    if args.command == "model":
        return run_model(config, args)
    if args.command == "api":
        return run_api(config, args.host, args.port)
    if args.command == "loadtest":
        print(json.dumps(asdict(run_load_test(args.flows, args.packets_per_flow)), indent=2))
        return 0
    for interface in available_interfaces():
        print(interface)
    return 0
