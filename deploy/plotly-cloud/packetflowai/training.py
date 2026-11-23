"""Training and evaluation service."""

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from .config import AppConfig
from .dataset_pipeline import HypervectorDataset, preprocess_dataset
from .domain import FlowFeatures
from .flows import TemporalFlowEncoder
from .hdc import HypervectorEncoder
from .manifests import ModelManifest, save_checkpoint
from .modeling import HVModel
from .taxonomy import ATTACK_TYPES


@dataclass(frozen=True)
class EvaluationMetrics:
    precision: float
    recall: float
    f1: float
    accuracy: float

    def as_dict(self) -> dict[str, float]:
        return {"precision": self.precision, "recall": self.recall, "f1": self.f1, "accuracy": self.accuracy}


def train_epoch(model: nn.Module, device: torch.device, loader: DataLoader,
                optimizer: optim.Optimizer, epoch: int) -> float:
    model.train()
    total_loss = 0.0
    for batch_index, (vectors, targets) in enumerate(loader):
        vectors, targets = vectors.to(device), targets.to(device)
        optimizer.zero_grad()
        loss = nn.CrossEntropyLoss()(model(vectors), targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if batch_index % 10 == 0:
            logging.info("Train epoch %s batch %s loss %.6f", epoch, batch_index, loss.item())
    return total_loss / max(1, len(loader))


def evaluate(model: nn.Module, device: torch.device, loader: DataLoader) -> EvaluationMetrics:
    model.eval()
    predictions: list[int] = []
    targets: list[int] = []
    with torch.no_grad():
        for vectors, batch_targets in loader:
            outputs = model(vectors.to(device))
            predictions.extend(outputs.argmax(dim=1).cpu().tolist())
            targets.extend(batch_targets.tolist())
    if not targets:
        raise ValueError("cannot evaluate an empty dataset")
    return EvaluationMetrics(
        precision=precision_score(targets, predictions, average="weighted", zero_division=0),
        recall=recall_score(targets, predictions, average="weighted", zero_division=0),
        f1=f1_score(targets, predictions, average="weighted", zero_division=0),
        accuracy=float((np.asarray(predictions) == np.asarray(targets)).mean()),
    )


class TrainingService:
    def __init__(self, config: AppConfig, encoder: HypervectorEncoder, model: HVModel,
                 device: torch.device):
        self.config = config
        self.encoder = encoder
        self.model = model
        self.device = device

    def fit(self, rows: Iterable[Mapping[str, Any]], dataset_id: str,
            dataset_fingerprint: str = "unknown") -> ModelManifest:
        vectors, targets = preprocess_dataset(rows, self.encoder)
        return self._fit_vectors(vectors, targets, dataset_id, dataset_fingerprint)

    def fit_flows(self, samples: Iterable[tuple[FlowFeatures, str, tuple[str, ...]]], dataset_id: str,
                  dataset_fingerprint: str = "unknown") -> ModelManifest:
        """Train on canonical flow windows; packet-row training remains a legacy adapter."""
        temporal = TemporalFlowEncoder(self.encoder)
        vectors = []
        targets = []
        for flow, label, event_tokens in samples:
            if label not in ATTACK_TYPES:
                raise ValueError(f"flow label is outside the taxonomy: {label}")
            vectors.append(temporal.encode(flow, event_tokens))
            targets.append(ATTACK_TYPES.index(label))
        if not vectors:
            raise ValueError("flow training set is empty")
        return self._fit_vectors(
            torch.tensor(np.stack(vectors), dtype=torch.float32),
            torch.tensor(targets, dtype=torch.long),
            dataset_id,
            dataset_fingerprint,
        )

    def _fit_vectors(self, vectors: torch.Tensor, targets: torch.Tensor, dataset_id: str,
                     dataset_fingerprint: str) -> ModelManifest:
        train_vectors, test_vectors, train_targets, test_targets = train_test_split(
            vectors,
            targets,
            test_size=self.config.training.test_size,
            random_state=self.config.training.random_seed,
            stratify=targets if len(torch.unique(targets)) > 1 else None,
        )
        train_loader = DataLoader(
            HypervectorDataset(train_vectors, train_targets),
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=self.config.training.num_workers,
        )
        test_loader = DataLoader(
            HypervectorDataset(test_vectors, test_targets),
            batch_size=self.config.training.batch_size,
            num_workers=self.config.training.num_workers,
        )
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.training.learning_rate)
        started_at = datetime.now(UTC).isoformat()
        best_f1 = -1.0
        manifest: ModelManifest | None = None
        self.config.artifacts.create()
        for epoch in range(1, self.config.training.epochs + 1):
            loss = train_epoch(self.model, self.device, train_loader, optimizer, epoch)
            metrics = evaluate(self.model, self.device, test_loader)
            logging.info("Epoch %s loss %.4f validation %s", epoch, loss, metrics.as_dict())
            if metrics.f1 > best_f1:
                best_f1 = metrics.f1
                manifest = save_checkpoint(
                    str(self.config.artifacts.model_checkpoint),
                    self.model,
                    self.encoder,
                    model_id=self.config.model.model_id,
                    model_version=self.config.model.model_version,
                    training_dataset_ids=[dataset_id],
                    training_dataset_fingerprints={dataset_id: dataset_fingerprint},
                    training_started_at=started_at,
                    training_completed_at=datetime.now(UTC).isoformat(),
                    validation_metrics=metrics.as_dict(),
                )
        if manifest is None:
            raise RuntimeError("training completed without producing a checkpoint")
        return manifest
