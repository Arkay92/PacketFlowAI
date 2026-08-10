"""Central application configuration."""

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ArtifactPaths:
    root: Path = Path("artifacts")

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def runtime(self) -> Path:
        return self.root / "runtime"

    @property
    def model_checkpoint(self) -> Path:
        return self.models / "packet_hv_model.pth"

    @property
    def error_log(self) -> Path:
        return self.runtime / "exceptions.log"

    @property
    def event_database(self) -> Path:
        return self.runtime / "packetflowai.sqlite3"

    @property
    def registry(self) -> Path:
        return self.root / "registry"

    @property
    def benchmark_reports(self) -> Path:
        return self.root / "benchmarks"

    def create(self) -> None:
        self.models.mkdir(parents=True, exist_ok=True)
        self.runtime.mkdir(parents=True, exist_ok=True)
        self.registry.mkdir(parents=True, exist_ok=True)
        self.benchmark_reports.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class ModelConfig:
    hv_dimension: int = 10_000
    num_levels: int = 100
    encoder_seed: int = 20_260_810
    hidden_dimensions: tuple[int, int] = (512, 256)
    dropout: float = 0.5
    model_id: str = "packet-hv-mlp"
    model_version: str = "2.3.0"


@dataclass(frozen=True)
class RuntimeConfig:
    queue_size: int = 1_000
    risk_half_life_seconds: float = 300.0
    allowlist: tuple[str, ...] = ("127.0.0.0/8", "::1/128")
    capture_poll_seconds: float = 1.0


@dataclass(frozen=True)
class TrainingConfig:
    dataset_id: str = "rdpahalavan/packet-tag-explanation"
    dataset_split: str = "train"
    epochs: int = 10
    batch_size: int = 16
    test_size: float = 0.2
    random_seed: int = 42
    learning_rate: float = 0.001
    num_workers: int = 0


@dataclass(frozen=True)
class NIMConfig:
    mode: str = "disabled"
    base_url: str = "https://integrate.api.nvidia.com/v1"
    model: str = "minimaxai/minimax-m3"
    timeout_seconds: float = 15.0
    retries: int = 1
    concurrency: int = 2
    cache_ttl_seconds: float = 300.0
    circuit_failure_threshold: int = 3
    circuit_reset_seconds: float = 60.0
    redact_internal_ips: bool = True
    redact_network_strings: bool = True
    maximum_string_length: int = 256


@dataclass(frozen=True)
class AppConfig:
    artifacts: ArtifactPaths = field(default_factory=ArtifactPaths)
    model: ModelConfig = field(default_factory=ModelConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    nim: NIMConfig = field(default_factory=NIMConfig)

    @classmethod
    def from_env(cls) -> "AppConfig":
        artifacts = ArtifactPaths(Path(os.getenv("PACKETFLOWAI_ARTIFACT_DIR", "artifacts")))
        model = ModelConfig(
            hv_dimension=int(os.getenv("PACKETFLOWAI_HV_DIMENSION", "10000")),
            num_levels=int(os.getenv("PACKETFLOWAI_NUM_LEVELS", "100")),
            encoder_seed=int(os.getenv("PACKETFLOWAI_ENCODER_SEED", "20260810")),
        )
        runtime = RuntimeConfig(
            queue_size=int(os.getenv("PACKETFLOWAI_QUEUE_SIZE", "1000")),
            risk_half_life_seconds=float(os.getenv("PACKETFLOWAI_RISK_HALF_LIFE", "300")),
        )
        nim = NIMConfig(
            mode=os.getenv("PACKETFLOWAI_NIM_MODE", "disabled").lower(),
            base_url=os.getenv("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1"),
            model=os.getenv("NIM_MODEL", "minimaxai/minimax-m3"),
        )
        if nim.mode not in {"disabled", "shadow", "influence"}:
            raise ValueError("PACKETFLOWAI_NIM_MODE must be disabled, shadow, or influence")
        return cls(artifacts=artifacts, model=model, runtime=runtime, nim=nim)
