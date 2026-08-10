import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from packetflowai.capture import ReplayService
from packetflowai.cli import build_parser
from packetflowai.config import AppConfig, ArtifactPaths, ModelConfig
from packetflowai.domain import LocalPrediction, ThreatAssessment
from packetflowai.features import packet_features_from_description
from packetflowai.hdc import HypervectorEncoder
from packetflowai.inference import PacketInferenceService, packet_features_from_scapy
from packetflowai.modeling import build_model


class ConfigurationTests(unittest.TestCase):
    def test_artifact_paths_are_outside_repository_root_files(self):
        paths = ArtifactPaths(Path("build-artifacts"))
        self.assertEqual(paths.model_checkpoint, Path("build-artifacts/models/packet_hv_model.pth"))
        self.assertEqual(paths.error_log, Path("build-artifacts/runtime/exceptions.log"))

    def test_environment_configuration(self):
        with patch.dict(os.environ, {"PACKETFLOWAI_HV_DIMENSION": "256", "PACKETFLOWAI_QUEUE_SIZE": "12"}):
            config = AppConfig.from_env()
        self.assertEqual(config.model.hv_dimension, 256)
        self.assertEqual(config.runtime.queue_size, 12)


class CliTests(unittest.TestCase):
    def test_commands_are_separate(self):
        parser = build_parser()
        self.assertEqual(parser.parse_args(["train"]).command, "train")
        self.assertEqual(parser.parse_args(["capture", "--interface", "eth0"]).command, "capture")
        self.assertEqual(parser.parse_args(["replay", "traffic.pcap"]).command, "replay")


class ScapyPipelineTests(unittest.TestCase):
    def test_live_packet_matches_dataset_schema(self):
        from scapy.layers.inet import IP, TCP

        packet = IP(version=4, len=60, src="192.0.2.1", dst="198.51.100.1") / TCP(
            sport=49152, dport=443, flags="SA"
        )
        live, packet_id, source_ip = packet_features_from_scapy(packet)
        dataset = packet_features_from_description(
            "IP version: 4, IP len: 60, TCP sport: 49152, TCP dport: https, TCP flags: SA"
        )
        self.assertEqual(live, dataset)
        self.assertEqual(source_ip, "192.0.2.1")
        self.assertEqual(len(packet_id), 64)

    def test_inference_returns_typed_boundaries(self):
        from scapy.layers.inet import IP, TCP

        config = AppConfig(model=ModelConfig(hv_dimension=64, hidden_dimensions=(16, 8), dropout=0.0))
        encoder = HypervectorEncoder(64, seed=config.model.encoder_seed)
        model = build_model(config.model)
        for parameter in model.parameters():
            torch.nn.init.zeros_(parameter)
        packet = IP(version=4, len=40, src="192.0.2.5", dst="198.51.100.2") / TCP(
            sport=12345, dport=80, flags="S"
        )
        service = PacketInferenceService(config, encoder, model, torch.device("cpu"), "test", "1")
        assessment, decisions = service.process(packet)
        self.assertIsInstance(assessment, ThreatAssessment)
        self.assertIsInstance(assessment.local_prediction, LocalPrediction)
        self.assertEqual(assessment.local_prediction.label, "benign")
        self.assertEqual(decisions, ())

    def test_replay_uses_the_inference_handler(self):
        from scapy.layers.inet import IP, UDP
        from scapy.utils import wrpcap

        received = []
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.pcap"
            wrpcap(str(path), [IP(src="192.0.2.1", dst="198.51.100.1") / UDP(sport=53, dport=53000)])
            replay = ReplayService(pipeline=None, handler=received.append)
            processed = replay.run(path)
        self.assertEqual(processed, 1)
        self.assertEqual(len(received), 1)


class EntrypointTests(unittest.TestCase):
    def test_main_is_only_a_compatibility_launcher(self):
        source = Path("main.py").read_text(encoding="utf-8")
        self.assertNotIn("class HVModel", source)
        self.assertIn("packetflowai.cli import main", source)


if __name__ == "__main__":
    unittest.main()
