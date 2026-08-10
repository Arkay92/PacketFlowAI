"""Versioned model checkpoint manifests and integrity validation."""

import hashlib
import platform
import subprocess
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

import torch

from .features import FEATURE_SCHEMA_VERSION
from .hdc import ENCODER_SCHEMA_VERSION, HypervectorEncoder
from .taxonomy import ATTACK_TYPES, TAXONOMY_VERSION

MANIFEST_SCHEMA_VERSION = "model-manifest-v1"


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def state_dict_hash(state_dict: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


@dataclass
class ModelManifest:
    model_id: str
    model_version: str
    model_weights_hash: str
    encoder_hash: str
    encoder_seed: int
    encoder_version: str = ENCODER_SCHEMA_VERSION
    feature_schema_version: str = FEATURE_SCHEMA_VERSION
    taxonomy_version: str = TAXONOMY_VERSION
    label_map: dict[str, int] = field(default_factory=lambda: {name: i for i, name in enumerate(ATTACK_TYPES)})
    schema_version: str = MANIFEST_SCHEMA_VERSION
    training_dataset_ids: list[str] = field(default_factory=list)
    training_dataset_fingerprints: dict[str, str] = field(default_factory=dict)
    training_capture_ids: list[str] = field(default_factory=list)
    training_started_at: str | None = None
    training_completed_at: str | None = None
    git_commit: str | None = field(default_factory=_git_commit)
    build_id: str | None = None
    validation_metrics: dict[str, float] = field(default_factory=dict)
    test_metrics: dict[str, float] = field(default_factory=dict)
    calibration_method: str | None = None
    calibration_artifact: str | None = None
    ood_thresholds: dict[str, float] = field(default_factory=dict)
    decision_thresholds: dict[str, float] = field(default_factory=dict)
    python_version: str = field(default_factory=platform.python_version)
    pytorch_version: str = field(default_factory=lambda: str(torch.__version__))
    artifact_created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "ModelManifest":
        return cls(**dict(values))

    def validate(self, encoder: HypervectorEncoder, state_dict: Mapping[str, torch.Tensor]) -> None:
        expected = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "encoder_version": encoder.schema_version,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "taxonomy_version": TAXONOMY_VERSION,
        }
        for field_name, expected_value in expected.items():
            actual = getattr(self, field_name)
            if actual != expected_value:
                raise ValueError(f"checkpoint {field_name} mismatch: {actual!r} != {expected_value!r}")
        if self.encoder_seed != encoder.seed or self.encoder_hash != encoder.configuration_hash():
            raise ValueError("checkpoint encoder configuration mismatch")
        if self.label_map != {name: i for i, name in enumerate(ATTACK_TYPES)}:
            raise ValueError("checkpoint label map mismatch")
        if self.model_weights_hash != state_dict_hash(state_dict):
            raise ValueError("checkpoint model weights failed integrity validation")


def save_checkpoint(path: str, model: torch.nn.Module, encoder: HypervectorEncoder,
                    model_id: str, model_version: str, **metadata: Any) -> ModelManifest:
    state_dict = model.state_dict()
    manifest = ModelManifest(
        model_id=model_id,
        model_version=model_version,
        model_weights_hash=state_dict_hash(state_dict),
        encoder_hash=encoder.configuration_hash(),
        encoder_seed=encoder.seed,
        **metadata,
    )
    torch.save({"manifest": asdict(manifest), "model_state_dict": state_dict}, path)
    return manifest


def load_checkpoint(path: str, model: torch.nn.Module, encoder: HypervectorEncoder,
                    map_location: Any = None) -> ModelManifest:
    try:
        checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=map_location)
    if not isinstance(checkpoint, dict) or "manifest" not in checkpoint or "model_state_dict" not in checkpoint:
        raise ValueError("legacy or malformed checkpoint: a Phase 1 manifest is required")
    manifest = ModelManifest.from_dict(checkpoint["manifest"])
    manifest.validate(encoder, checkpoint["model_state_dict"])
    model.load_state_dict(checkpoint["model_state_dict"])
    return manifest
