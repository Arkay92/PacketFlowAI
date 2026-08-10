"""Stable evidence-only clustering for UNKNOWN/OOD vectors."""

import hashlib
from dataclasses import dataclass, field

import numpy as np


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator else 0.0


@dataclass
class UnknownCluster:
    cluster_id: str
    centroid: np.ndarray
    sample_count: int
    first_seen: float
    last_seen: float
    source_ips: set[str] = field(default_factory=set)
    similarity_sum: float = 0.0
    common_characteristics: dict[str, int] = field(default_factory=dict)
    hypothesis: str | None = None

    @property
    def internal_similarity(self) -> float:
        return self.similarity_sum / max(1, self.sample_count - 1)


class UnknownClusterer:
    def __init__(self, similarity_threshold: float = 0.75):
        self.similarity_threshold = similarity_threshold
        self.clusters: dict[str, UnknownCluster] = {}

    def _new_id(self, vector: np.ndarray) -> str:
        packed = np.packbits(np.asarray(vector) > 0).tobytes()
        return "unknown-" + hashlib.sha256(packed).hexdigest()[:12]

    def assign(self, vector: np.ndarray, timestamp: float, source_ip: str,
               characteristics: dict[str, str] | None = None) -> UnknownCluster:
        vector = np.asarray(vector, dtype=np.float64)
        matches = [
            (cosine_similarity(vector, cluster.centroid), cluster)
            for cluster in self.clusters.values()
        ]
        similarity, cluster = max(
            matches,
            default=(-1.0, None),
            key=lambda item: (item[0], item[1].cluster_id if item[1] else ""),
        )
        if cluster is None or similarity < self.similarity_threshold:
            cluster_id = self._new_id(vector)
            suffix = 1
            while cluster_id in self.clusters:
                suffix += 1
                cluster_id = f"{self._new_id(vector)}-{suffix}"
            cluster = UnknownCluster(cluster_id, vector.copy(), 1, timestamp, timestamp, {source_ip})
            self.clusters[cluster_id] = cluster
        else:
            cluster.centroid = (cluster.centroid * cluster.sample_count + vector) / (cluster.sample_count + 1)
            cluster.sample_count += 1
            cluster.last_seen = timestamp
            cluster.source_ips.add(source_ip)
            cluster.similarity_sum += similarity
        for key, value in (characteristics or {}).items():
            token = f"{key}={value}"
            cluster.common_characteristics[token] = cluster.common_characteristics.get(token, 0) + 1
        return cluster
