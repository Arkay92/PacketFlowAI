"""Deterministic hyperdimensional packet encoding."""

import hashlib
import json
from collections.abc import Mapping
from typing import Any

import numpy as np

from .features import PacketFeatures

ENCODER_SCHEMA_VERSION = "hdc-v1"

FEATURE_RANGES = {
    "ip_version": (0, 6),
    "ip_len": (0, 65535),
    "tcp_sport": (0, 65535),
    "tcp_dport": (0, 65535),
    "tcp_flags": (0, 255),
    "udp_sport": (0, 65535),
    "udp_dport": (0, 65535),
}


class HypervectorEncoder:
    def __init__(self, dimension: int, num_levels: int = 100, seed: int = 20260810,
                 schema_version: str = ENCODER_SCHEMA_VERSION):
        if dimension <= 0 or num_levels <= 1:
            raise ValueError("dimension must be positive and num_levels must exceed one")
        self.dimension = dimension
        self.num_levels = num_levels
        self.seed = seed
        self.schema_version = schema_version

    def _vector(self, namespace: str, key: Any) -> np.ndarray:
        identity = f"{self.seed}|{self.schema_version}|{namespace}|{key}".encode()
        derived_seed = int.from_bytes(hashlib.sha256(identity).digest()[:8], "big")
        rng = np.random.default_rng(derived_seed)
        return rng.choice(np.array([-1, 1], dtype=np.int8), size=self.dimension)

    def quantize(self, value: float, min_value: float, max_value: float) -> int:
        if max_value <= min_value:
            raise ValueError("max_value must be greater than min_value")
        scaled = (float(value) - min_value) / (max_value - min_value)
        return int(np.clip(round(scaled * (self.num_levels - 1)), 0, self.num_levels - 1))

    def encode_numerical(self, feature_name: str, value: float | None,
                         min_value: float, max_value: float) -> np.ndarray:
        role = self._vector("role", feature_name)
        value_vector = self._vector("missing", feature_name) if value is None else self._vector(
            "level", self.quantize(value, min_value, max_value)
        )
        return role * value_vector

    def encode_categorical(self, feature_name: str, value: Any) -> np.ndarray:
        role = self._vector("role", feature_name)
        value_vector = self._vector("missing", feature_name) if value is None else self._vector(
            f"category:{feature_name}", str(value)
        )
        return role * value_vector

    def bundle(self, vectors: list[np.ndarray]) -> np.ndarray:
        if not vectors:
            raise ValueError("cannot bundle an empty vector list")
        summed = np.sum(vectors, axis=0)
        return np.where(summed >= 0, 1, -1).astype(np.int8)

    def permute(self, vector: np.ndarray, steps: int = 1) -> np.ndarray:
        """Encode sequence position using a deterministic cyclic permutation."""
        return np.roll(vector, int(steps))

    def encode_sequence(self, vectors: list[np.ndarray]) -> np.ndarray:
        if not vectors:
            raise ValueError("cannot encode an empty sequence")
        return self.bundle([self.permute(vector, index) for index, vector in enumerate(vectors)])

    def encode_packet(self, features: PacketFeatures | Mapping[str, Any]) -> np.ndarray:
        values = features.as_mapping() if isinstance(features, PacketFeatures) else dict(features)
        vectors = [self.encode_categorical("protocol", values.get("protocol"))]
        for name, (minimum, maximum) in FEATURE_RANGES.items():
            vectors.append(self.encode_numerical(name, values.get(name), minimum, maximum))
        return self.bundle(vectors)

    def configuration(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "num_levels": self.num_levels,
            "schema_version": self.schema_version,
            "seed": self.seed,
        }

    def configuration_hash(self) -> str:
        payload = json.dumps(self.configuration(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
