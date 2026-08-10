import os
import tempfile
import unittest

import numpy as np
import torch

from packetflowai.features import (
    PacketFeatures,
    canonical_tcp_flags,
    packet_features_from_description,
    packet_features_from_mapping,
    parse_port,
)
from packetflowai.hdc import HypervectorEncoder
from packetflowai.manifests import load_checkpoint, save_checkpoint
from packetflowai.policy import AlertOnlyPolicy, RiskTracker
from packetflowai.taxonomy import extract_authoritative_label


class HypervectorEncoderTests(unittest.TestCase):
    def setUp(self):
        self.features = PacketFeatures(
            ip_version=4,
            ip_len=60,
            protocol="TCP",
            tcp_sport=49152,
            tcp_dport=443,
            tcp_flags=18,
        )

    def test_encoding_is_deterministic_across_instances(self):
        first = HypervectorEncoder(512, seed=42).encode_packet(self.features)
        second = HypervectorEncoder(512, seed=42).encode_packet(self.features)
        np.testing.assert_array_equal(first, second)

    def test_feature_identity_is_bound_to_numerical_value(self):
        encoder = HypervectorEncoder(2048, seed=42)
        source = encoder.encode_numerical("tcp_sport", 443, 0, 65535)
        destination = encoder.encode_numerical("tcp_dport", 443, 0, 65535)
        self.assertFalse(np.array_equal(source, destination))

    def test_quantization_clamps_both_boundaries(self):
        encoder = HypervectorEncoder(128, num_levels=10)
        self.assertEqual(encoder.quantize(-100, 0, 100), 0)
        self.assertEqual(encoder.quantize(1000, 0, 100), 9)

    def test_missing_value_has_a_stable_distinct_encoding(self):
        encoder = HypervectorEncoder(512, seed=42)
        missing = encoder.encode_numerical("tcp_sport", None, 0, 65535)
        zero = encoder.encode_numerical("tcp_sport", 0, 0, 65535)
        self.assertFalse(np.array_equal(missing, zero))


class CanonicalFeatureTests(unittest.TestCase):
    def test_description_and_mapping_paths_have_parity(self):
        description = packet_features_from_description(
            "IP version: 4, IP len: 60, TCP sport: 49152, TCP dport: https, TCP flags: SA"
        )
        live = packet_features_from_mapping({
            "ip_version": 4,
            "ip_len": 60,
            "protocol": "TCP",
            "tcp_sport": 49152,
            "tcp_dport": 443,
            "tcp_flags": 18,
        })
        self.assertEqual(description, live)

    def test_flags_use_a_canonical_bitmask(self):
        self.assertEqual(canonical_tcp_flags("SA"), 18)
        self.assertEqual(canonical_tcp_flags(18), 18)
        self.assertEqual(canonical_tcp_flags("AS"), 18)

    def test_ports_preserve_numbers_and_known_services(self):
        self.assertEqual(parse_port("54321"), 54321)
        self.assertEqual(parse_port("https"), 443)
        self.assertIsNone(parse_port("unregistered-service"))

    def test_missing_fields_remain_explicit(self):
        features = packet_features_from_description("IP version: 4, TCP flags: S")
        self.assertIsNone(features.ip_len)
        self.assertIsNone(features.tcp_dport)


class LabelTests(unittest.TestCase):
    def test_dataset_native_label_is_normalized_with_provenance(self):
        label = extract_authoritative_label({"label": "port scan", "Explanation": "ignore me"})
        self.assertEqual(label.normalized_value, "port_scan")
        self.assertEqual(label.provenance, "dataset-native")

    def test_explanation_is_never_used_as_a_label(self):
        with self.assertRaisesRegex(ValueError, "no authoritative label"):
            extract_authoritative_label({"Explanation": "This describes a DDoS attack"})


class ManifestTests(unittest.TestCase):
    def test_checkpoint_round_trip_and_encoder_validation(self):
        model = torch.nn.Linear(16, 2)
        encoder = HypervectorEncoder(16, seed=7)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "model.pth")
            saved = save_checkpoint(path, model, encoder, "test-model", "1.0")
            loaded_model = torch.nn.Linear(16, 2)
            loaded = load_checkpoint(path, loaded_model, encoder, map_location="cpu")
        self.assertEqual(saved.model_weights_hash, loaded.model_weights_hash)
        for expected, actual in zip(model.parameters(), loaded_model.parameters(), strict=True):
            self.assertTrue(torch.equal(expected, actual))

    def test_checkpoint_rejects_encoder_drift(self):
        model = torch.nn.Linear(16, 2)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "model.pth")
            save_checkpoint(path, model, HypervectorEncoder(16, seed=7), "test-model", "1.0")
            with self.assertRaisesRegex(ValueError, "encoder configuration mismatch"):
                load_checkpoint(path, model, HypervectorEncoder(16, seed=8), map_location="cpu")


class PolicyTests(unittest.TestCase):
    def test_default_policy_only_alerts_on_non_benign_results(self):
        policy = AlertOnlyPolicy()
        self.assertEqual(policy.respond({"label": "benign"}), [])
        self.assertEqual(policy.respond({"label": "malware"})[0]["action"], "alert")

    def test_risk_updates_after_malicious_classification_and_decays(self):
        tracker = RiskTracker(half_life_seconds=10)
        self.assertEqual(tracker.score("192.0.2.10", now=0), 0)
        self.assertEqual(tracker.update_after_classification("192.0.2.10", "malware", now=0), 1)
        self.assertAlmostEqual(tracker.score("192.0.2.10", now=10), 0.5)

    def test_allowlisted_sources_do_not_gain_risk(self):
        tracker = RiskTracker(allowlist=("10.0.0.0/8",))
        self.assertEqual(tracker.update_after_classification("10.1.2.3", "malware", now=0), 0)


class TrainingBoundaryTests(unittest.TestCase):
    def test_single_item_evaluation_batch_is_supported(self):
        from packetflowai.training import evaluate

        model = torch.nn.Linear(4, 2)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.ones(1, 4), torch.zeros(1, dtype=torch.long)),
            batch_size=1,
        )
        metrics = evaluate(model, torch.device("cpu"), loader)
        self.assertGreaterEqual(metrics.f1, 0.0)


if __name__ == "__main__":
    unittest.main()
